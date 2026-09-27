"""
Objective 1 - Dual-Branch Hybrid Deep Learning Model for student study-pathway prediction.

Architecture
    Branch A (static)   : MLP over one-hot / scaled socioeconomic features       -> z_static   (64)
    Branch B (temporal) : Causal dilated TCN block + uni-directional LSTM over
                          Saber 11 / score features reshaped to (B, 3, 3)          -> z_temporal (64)
    Fusion head         : [z_static || z_temporal] (128) -> Dense(64) -> ReLU -> Dropout -> logits

Targets evaluated (same train/val/test rows for both, for a direct comparison)
    1. MACRO_TRACK       (4 broad tracks)       -> checkpoint: best_objective1_model.pt
    2. ACADEMIC_PROGRAM  (21 granular programs) -> checkpoint: best_objective1_model_program.pt

Usage
    python objective1_hybrid_model.py            # expects dataset.csv in the current directory
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
SEED = 42
DATA_PATH = "dataset.csv"
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2
TOP_K = 3
MACRO_CHECKPOINT = "best_objective1_model.pt"
PROGRAM_CHECKPOINT = "best_objective1_model_program.pt"

STATIC_CATEGORICAL = [
    "GENDER", "STRATUM", "SISBEN", "SCHOOL_TYPE", "SCHOOL_NAT",
    "EDU_FATHER", "EDU_MOTHER", "OCC_FATHER", "OCC_MOTHER", "REVENUE",
    "INTERNET", "COMPUTER", "CAR",
]
STATIC_NUMERIC = ["PEOPLE_HOUSE"]

# 9 features -> 3 time steps x 3 features per step (order defines the sequence).
TEMPORAL_FEATURES = [
    "MAT_S11", "CR_S11", "CC_S11",          # step 1: Saber 11 core subjects
    "BIO_S11", "ENG_S11", "G_SC",           # step 2: Saber 11 bio/english + global score
    "PERCENTILE", "2ND_DECILE", "QUARTILE", # step 3: rank-based standing
]
TIME_STEPS = 3
FEATURES_PER_STEP = len(TEMPORAL_FEATURES) // TIME_STEPS

MACRO_TRACK_MAP = {
    "Industrial & Management": [
        "INDUSTRIAL ENGINEERING", "PRODUCTION ENGINEERING",
        "PRODUCTIVITY AND QUALITY ENGINEERING",
    ],
    "Civil & Infrastructure": [
        "CIVIL ENGINEERING", "CATASTRAL ENGINEERING AND GEODESY", "TOPOGRAPHIC ENGINEERY",
        "CIVIL CONSTRUCTIONS", "TRANSPORTATION AND ROAD ENGINEERING",
    ],
    "Mechanical, Electrical & Tech": [
        "MECHANICAL ENGINEERING", "ELECTRONIC ENGINEERING", "ELECTRIC ENGINEERING",
        "MECHATRONICS ENGINEERING", "ELECTRIC ENGINEERING AND TELECOMMUNICATIONS",
        "AERONAUTICAL ENGINEERING", "ELECTROMECHANICAL ENGINEERING",
        "INDUSTRIAL AUTOMATIC ENGINEERING", "CONTROL ENGINEERING", "AUTOMATION ENGINEERING",
        "INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING",
    ],
    "Chemical & Process": ["CHEMICAL ENGINEERING", "TEXTILE ENGINEERING"],
}
PROGRAM_TO_TRACK = {prog: track for track, progs in MACRO_TRACK_MAP.items() for prog in progs}

PEOPLE_HOUSE_MAP = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7,
    "EIGHT": 8, "NINE": 9, "NUEVE": 9, "TEN": 10, "ELEVEN": 11, "ONCE": 11,
    "TWELVE OR MORE": 12,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------------------------
# 1. Loading & cleaning
# ----------------------------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Could not find '{path}' in the current directory.")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # Column names: strip whitespace, drop empty/unnamed columns.
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df = df.dropna(axis=1, how="all")

    # String values: strip whitespace; '0' and blanks are the dataset's missing-value codes.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"0": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})

    df["ACADEMIC_PROGRAM"] = df["ACADEMIC_PROGRAM"].str.upper().str.replace(r"\s+", " ", regex=True)
    df = df.dropna(subset=["ACADEMIC_PROGRAM"])

    for col in STATIC_CATEGORICAL:
        df[col] = df[col].fillna("Unknown")

    df["PEOPLE_HOUSE"] = df["PEOPLE_HOUSE"].str.upper().map(PEOPLE_HOUSE_MAP).astype(float)

    for col in TEMPORAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def add_macro_track(df: pd.DataFrame) -> pd.DataFrame:
    df["MACRO_TRACK"] = df["ACADEMIC_PROGRAM"].map(PROGRAM_TO_TRACK)
    unmapped = df.loc[df["MACRO_TRACK"].isna(), "ACADEMIC_PROGRAM"].unique()
    if len(unmapped) > 0:
        print(f"[WARN] Dropping {len(unmapped)} unmapped program(s): {list(unmapped)}")
        df = df.dropna(subset=["MACRO_TRACK"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------------------------
# 2. Stratified 80/10/10 split (shared by both targets)
# ----------------------------------------------------------------------------------------------
def _safe_strata(labels: np.ndarray, min_count: int = 2) -> np.ndarray:
    """Merge classes too small to stratify into the majority stratum so sklearn never fails."""
    labels = np.asarray(labels, dtype=object).copy()
    values, counts = np.unique(labels, return_counts=True)
    majority = values[np.argmax(counts)]
    too_small = set(values[counts < min_count])
    if too_small:
        labels[np.isin(labels, list(too_small))] = majority
    return labels


def stratified_split(df: pd.DataFrame, strat_col: str, seed: int):
    """
    80/10/10 stratified split on the granular program label. Because MACRO_TRACK is a
    deterministic function of ACADEMIC_PROGRAM, this is also stratified for MACRO_TRACK.
    Programs with < 3 students cannot appear in all three splits, so they are kept in train.
    """
    counts = df[strat_col].value_counts()
    rare = counts[counts < 3].index
    rare_idx = df.index[df[strat_col].isin(rare)].to_numpy()
    main_idx = df.index[~df[strat_col].isin(rare)].to_numpy()
    if len(rare) > 0:
        print(f"[INFO] Programs with <3 samples kept in train only: "
              f"{ {p: int(counts[p]) for p in rare} }")

    y_main = df.loc[main_idx, strat_col].to_numpy()
    train_idx, temp_idx = train_test_split(
        main_idx, test_size=0.20, random_state=seed, stratify=_safe_strata(y_main)
    )
    y_temp = df.loc[temp_idx, strat_col].to_numpy()
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=_safe_strata(y_temp)
    )
    train_idx = np.concatenate([train_idx, rare_idx])
    return train_idx, val_idx, test_idx


# ----------------------------------------------------------------------------------------------
# 3. Feature preprocessing (fit on train only -> no leakage)
# ----------------------------------------------------------------------------------------------
class FeaturePreprocessor:
    def __init__(self):
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.static_ct = ColumnTransformer(
            [("cat", ohe, STATIC_CATEGORICAL), ("num", StandardScaler(), STATIC_NUMERIC)]
        )
        self.temporal_scaler = StandardScaler()
        self.static_medians = None
        self.temporal_medians = None

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[STATIC_NUMERIC] = df[STATIC_NUMERIC].fillna(self.static_medians)
        df[TEMPORAL_FEATURES] = df[TEMPORAL_FEATURES].fillna(self.temporal_medians)
        return df

    def fit(self, df_train: pd.DataFrame) -> "FeaturePreprocessor":
        self.static_medians = df_train[STATIC_NUMERIC].median()
        self.temporal_medians = df_train[TEMPORAL_FEATURES].median()
        df_train = self._impute(df_train)
        self.static_ct.fit(df_train[STATIC_CATEGORICAL + STATIC_NUMERIC])
        self.temporal_scaler.fit(df_train[TEMPORAL_FEATURES].to_numpy(dtype=np.float64))
        return self

    def transform(self, df: pd.DataFrame):
        df = self._impute(df)
        x_static = self.static_ct.transform(df[STATIC_CATEGORICAL + STATIC_NUMERIC]).astype(np.float32)
        x_temp = self.temporal_scaler.transform(df[TEMPORAL_FEATURES].to_numpy(dtype=np.float64))
        x_temp = x_temp.astype(np.float32).reshape(-1, TIME_STEPS, FEATURES_PER_STEP)
        return x_static, x_temp


# ----------------------------------------------------------------------------------------------
# 4. PyTorch Dataset
# ----------------------------------------------------------------------------------------------
class StudentPathwayDataset(Dataset):
    def __init__(self, x_static: np.ndarray, x_temporal: np.ndarray, y: np.ndarray):
        self.x_static = torch.as_tensor(x_static, dtype=torch.float32)
        self.x_temporal = torch.as_tensor(x_temporal, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_static[idx], self.x_temporal[idx], self.y[idx]


# ----------------------------------------------------------------------------------------------
# 5. Model
# ----------------------------------------------------------------------------------------------
class StaticMLP(nn.Module):
    """Branch A: Dense(128) -> BN -> ReLU -> Dropout(0.2) -> Dense(64) -> ReLU."""

    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding so output at t depends only on inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):  # x: (B, C, T)
        return self.conv(nn.functional.pad(x, (self.left_pad, 0)))


class CausalTCNBlock(nn.Module):
    """Causal dilated conv (k=2, d=1) -> ReLU -> Dropout, with a residual (1x1 projection) path."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 2, dilation: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.out_relu = nn.ReLU()

    def forward(self, x):  # x: (B, C, T)
        out = self.dropout(self.relu(self.conv(x)))
        return self.out_relu(out + self.residual(x))


class TemporalEncoder(nn.Module):
    """Branch B: Causal TCN block -> uni-directional LSTM(64) -> last hidden state."""

    def __init__(self, features_per_step: int, tcn_channels: int = 64, hidden_size: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.tcn = CausalTCNBlock(features_per_step, tcn_channels, kernel_size=2, dilation=1,
                                  dropout=dropout)
        self.lstm = nn.LSTM(input_size=tcn_channels, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=False)

    def forward(self, x):  # x: (B, T, F)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        _, (h_n, _) = self.lstm(h)
        return h_n[-1]  # (B, hidden)


class DualBranchHybridModel(nn.Module):
    def __init__(self, static_dim: int, features_per_step: int, num_classes: int,
                 dropout: float = 0.2):
        super().__init__()
        self.static_branch = StaticMLP(static_dim, dropout)
        self.temporal_branch = TemporalEncoder(features_per_step, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_static, x_temporal):
        z_static = self.static_branch(x_static)
        z_temporal = self.temporal_branch(x_temporal)
        fused = torch.cat([z_static, z_temporal], dim=1)
        return self.head(fused)


# ----------------------------------------------------------------------------------------------
# 6. Training / evaluation
# ----------------------------------------------------------------------------------------------
def inverse_frequency_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """c_y = N / (K * N_y). Classes absent from train get weight N/K (they never contribute)."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    weights = len(y_train) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xs, xt, y in loader:
            xs, xt, y = xs.to(device), xt.to(device), y.to(device)
            logits = model(xs, xt)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, labels = [], []
    for xs, xt, y in loader:
        logits = model(xs.to(device), xt.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_and_evaluate(df, target_col, split, x_arrays, checkpoint_path, device):
    train_idx, val_idx, test_idx = split
    (xs_tr, xt_tr), (xs_va, xt_va), (xs_te, xt_te) = x_arrays

    encoder = LabelEncoder().fit(df[target_col])
    class_names = list(encoder.classes_)
    num_classes = len(class_names)
    y_all = encoder.transform(df[target_col])
    y_tr, y_va, y_te = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    print("\n" + "=" * 90)
    print(f"TARGET: {target_col}  ({num_classes} classes)")
    print("=" * 90)
    print(f"Train/Val/Test sizes: {len(y_tr)} / {len(y_va)} / {len(y_te)}")

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(StudentPathwayDataset(xs_tr, xt_tr, y_tr), batch_size=BATCH_SIZE,
                              shuffle=True, generator=g)
    val_loader = DataLoader(StudentPathwayDataset(xs_va, xt_va, y_va), batch_size=BATCH_SIZE)
    test_loader = DataLoader(StudentPathwayDataset(xs_te, xt_te, y_te), batch_size=BATCH_SIZE)

    class_weights = inverse_frequency_weights(y_tr, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    set_seed(SEED)
    model = DualBranchHybridModel(xs_tr.shape[1], FEATURES_PER_STEP, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        flag = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": va_loss,
                "target": target_col,
                "class_names": class_names,
                "static_dim": xs_tr.shape[1],
                "features_per_step": FEATURES_PER_STEP,
            }, checkpoint_path)
            flag = "  <-- best (saved)"
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc*100:6.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:6.2f}%{flag}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded best checkpoint '{checkpoint_path}' (epoch {ckpt['epoch']}, "
          f"val loss {ckpt['val_loss']:.4f})")

    probs, y_true = predict_proba(model, test_loader, device)
    y_pred = probs.argmax(1)
    all_labels = np.arange(num_classes)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)
    k = TOP_K
    top_k = top_k_accuracy_score(y_true, probs, k=k, labels=all_labels)
    majority_baseline = (y_true == np.bincount(y_tr, minlength=num_classes).argmax()).mean()

    print(f"\n--- Test results: {target_col} ---")
    print(f"Majority-class baseline accuracy : {majority_baseline*100:.2f}%")
    print(f"Overall Test Accuracy            : {acc*100:.2f}%")
    print(f"Macro F1-Score                   : {macro_f1:.4f}  (over classes present in test)")
    print(f"Top-{k} Recommendation Accuracy   : {top_k*100:.2f}%")
    print("\nClassification Report:")
    present = np.unique(np.concatenate([y_true, y_pred]))
    print(classification_report(y_true, y_pred, labels=present,
                                target_names=[class_names[i] for i in present],
                                digits=4, zero_division=0))

    return {"target": target_col, "classes": num_classes, "accuracy": acc,
            "macro_f1": macro_f1, "top_k_acc": top_k, "top_k": k,
            "majority_baseline": majority_baseline}


# ----------------------------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = add_macro_track(load_and_clean(DATA_PATH))
    print(f"Loaded {len(df)} records, {df.shape[1]} columns.")
    print("\nMACRO_TRACK distribution:")
    print(df["MACRO_TRACK"].value_counts().to_string())
    print(f"\nACADEMIC_PROGRAM classes: {df['ACADEMIC_PROGRAM'].nunique()}")

    split = stratified_split(df, "ACADEMIC_PROGRAM", SEED)
    train_idx, val_idx, test_idx = split

    prep = FeaturePreprocessor().fit(df.loc[train_idx])
    x_arrays = tuple(prep.transform(df.loc[idx]) for idx in (train_idx, val_idx, test_idx))
    print(f"Static feature dim (after one-hot): {x_arrays[0][0].shape[1]} | "
          f"Temporal tensor shape: (B, {TIME_STEPS}, {FEATURES_PER_STEP})")

    results = [
        train_and_evaluate(df, "MACRO_TRACK", split, x_arrays, MACRO_CHECKPOINT, device),
        train_and_evaluate(df, "ACADEMIC_PROGRAM", split, x_arrays, PROGRAM_CHECKPOINT, device),
    ]

    print("\n" + "=" * 90)
    print("SUMMARY: Macro-Track vs Granular Program (held-out test set)")
    print("=" * 90)
    print(f"{'Target':<20}{'Classes':>8}{'Majority':>11}{'Accuracy':>11}{'Macro-F1':>11}{'Top-k Acc':>12}")
    for r in results:
        print(f"{r['target']:<20}{r['classes']:>8}{r['majority_baseline']*100:>10.2f}%"
              f"{r['accuracy']*100:>10.2f}%{r['macro_f1']:>11.4f}"
              f"{r['top_k_acc']*100:>10.2f}% (k={r['top_k']})")


if __name__ == "__main__":
    main()
