# -*- coding: utf-8 -*-
"""
Objective 1 - Student Study Pathway Prediction
================================================
Benchmark of a Dual-Branch Hybrid Deep Learning model (Static MLP + Causal TCN +
Uni-LSTM) against traditional ML, single-branch deep, and hybrid baselines.

Protocol
--------
* Stratified 5-fold CV on ACADEMIC_PROGRAM (random_state=42).
* ALL preprocessing (imputation, one-hot encoding, scaling) is fitted on the
  training fold only and then applied to the validation fold (no leakage).
* Every model is trained on the 21-class ACADEMIC_PROGRAM target. MACRO_TRACK
  predictions are obtained by summing the granular class probabilities that
  belong to each track and taking the argmax, so a single model yields both the
  granular recommendation and the macro-track prediction consistently.
* Metrics per fold: Macro-Track Top-1 Acc and Macro-F1, Granular Top-1 Acc,
  Granular Top-3 Acc, Granular Macro-F1. Paired t-tests (scipy.stats.ttest_rel)
  on fold-wise Macro-F1: Proposed model vs. every other model.
* A majority-class model is included as a reference floor.
* `run_experiment` is parameterised (training label, class weighting, temporal
  features); `objective1_ablations.py` uses it for the follow-up experiments.

Usage (Google Colab)
--------------------
1. Upload `dataset.csv` to the working directory (/content).
2. Run this file (`!python objective1_experiments.py`) or paste it in a cell.
3. Results are printed and exported to `objective1_rigorous_results.csv`
   (summary) and `objective1_fold_results.csv` (raw fold-wise metrics).
"""

import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION & REPRODUCIBILITY
# =============================================================================
DATA_PATH = "dataset.csv"
SUMMARY_CSV = "objective1_rigorous_results.csv"
FOLD_CSV = "objective1_fold_results.csv"

SEED = 42
N_SPLITS = 5
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2
DROPOUT = 0.2
TIME_STEPS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

TARGET = "ACADEMIC_PROGRAM"
MACRO_TARGET = "MACRO_TRACK"

MACRO_TRACK_MAP = {
    "Industrial & Management": [
        "INDUSTRIAL ENGINEERING",
        "PRODUCTION ENGINEERING",
        "PRODUCTIVITY AND QUALITY ENGINEERING",
    ],
    "Civil & Infrastructure": [
        "CIVIL ENGINEERING",
        "CATASTRAL ENGINEERING AND GEODESY",
        "TOPOGRAPHIC ENGINEERY",
        "CIVIL CONSTRUCTIONS",
        "TRANSPORTATION AND ROAD ENGINEERING",
    ],
    "Mechanical, Electrical & Tech": [
        "MECHANICAL ENGINEERING",
        "ELECTRONIC ENGINEERING",
        "ELECTRIC ENGINEERING",
        "MECHATRONICS ENGINEERING",
        "ELECTRIC ENGINEERING AND TELECOMMUNICATIONS",
        "AERONAUTICAL ENGINEERING",
        "ELECTROMECHANICAL ENGINEERING",
        "INDUSTRIAL AUTOMATIC ENGINEERING",
        "CONTROL ENGINEERING",
        "AUTOMATION ENGINEERING",
        "INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING",
    ],
    "Chemical & Process": [
        "CHEMICAL ENGINEERING",
        "TEXTILE ENGINEERING",
    ],
}
PROGRAM_TO_TRACK = {p: t for t, progs in MACRO_TRACK_MAP.items() for p in progs}

STATIC_FEATURES = [
    "GENDER", "STRATUM", "SISBEN", "SCHOOL_TYPE", "SCHOOL_NAT", "EDU_FATHER",
    "EDU_MOTHER", "OCC_FATHER", "OCC_MOTHER", "REVENUE", "PEOPLE_HOUSE",
    "INTERNET", "COMPUTER", "CAR",
]

# Ordered so that reshaping to (3, 4) gives:
#   step 1: [MAT, CR, CC, BIO]          - subject scores
#   step 2: [ENG, G_SC, PERCENTILE, 2ND_DECILE] - global standing
#   step 3: [QUARTILE, STEM_AVG, HUM_AVG, STEM_HUM_RATIO] - derived profile
TEMPORAL_FEATURES = [
    "MAT_S11", "CR_S11", "CC_S11", "BIO_S11",
    "ENG_S11", "G_SC", "PERCENTILE", "2ND_DECILE",
    "QUARTILE", "STEM_AVG", "HUM_AVG", "STEM_HUM_RATIO",
]
assert len(TEMPORAL_FEATURES) % TIME_STEPS == 0
FEATS_PER_STEP = len(TEMPORAL_FEATURES) // TIME_STEPS

# =============================================================================
# 1. DATA LOADING, TARGET MAPPING & FEATURE ENGINEERING
# =============================================================================


def load_and_prepare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found in {os.getcwd()}. Upload dataset.csv to the "
            "Colab working directory (/content) and re-run."
        )
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:  # the original Saber 11/Pro export is Latin-1
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    print(f"[Data] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.")

    required = set(STATIC_FEATURES + [TARGET, "MAT_S11", "CR_S11", "CC_S11", "BIO_S11",
                                      "ENG_S11", "G_SC", "PERCENTILE", "2ND_DECILE", "QUARTILE"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    # Filter missing targets and normalise program names
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
    df = df[df[TARGET] != ""]

    # Macro-track mapping
    df[MACRO_TARGET] = df[TARGET].map(PROGRAM_TO_TRACK)
    unmapped = df.loc[df[MACRO_TARGET].isna(), TARGET].value_counts()
    if len(unmapped) > 0:
        print("[Data] WARNING - programs not covered by the MACRO_TRACK map "
              "(rows dropped):")
        print(unmapped.to_string())
        df = df[df[MACRO_TARGET].notna()].copy()

    # Numeric coercion for academic variables
    for c in ["MAT_S11", "CR_S11", "CC_S11", "BIO_S11", "ENG_S11", "G_SC",
              "PERCENTILE", "2ND_DECILE", "QUARTILE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Feature engineering
    df["STEM_AVG"] = (df["MAT_S11"] + df["BIO_S11"]) / 2.0
    df["HUM_AVG"] = (df["CR_S11"] + df["CC_S11"]) / 2.0
    df["STEM_HUM_RATIO"] = df["STEM_AVG"] / (df["HUM_AVG"] + 1e-5)
    df["STEM_HUM_RATIO"] = df["STEM_HUM_RATIO"].replace([np.inf, -np.inf], np.nan)

    df = df.reset_index(drop=True)
    print(f"[Data] {len(df):,} rows after filtering | "
          f"{df[TARGET].nunique()} programs | {df[MACRO_TARGET].nunique()} macro-tracks")
    print("[Data] Macro-track distribution:")
    print(df[MACRO_TARGET].value_counts().to_string())
    return df


def split_static_types(df: pd.DataFrame):
    """Numeric static columns -> continuous (scaled); everything else -> nominal (one-hot)."""
    continuous, nominal = [], []
    for c in STATIC_FEATURES:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2:
            continuous.append(c)
        else:
            nominal.append(c)
    return continuous, nominal


def build_static_transformer(continuous, nominal) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformers = []
    if continuous:
        transformers.append(("cont", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), continuous))
    if nominal:
        transformers.append(("nom", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", ohe),
        ]), nominal))
    return ColumnTransformer(transformers, remainder="drop")


def build_temporal_transformer() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])


def preprocess_fold(df_tr, df_va, continuous, nominal,
                    temporal_features=TEMPORAL_FEATURES, time_steps=TIME_STEPS):
    """Fit transformers on training fold only; return arrays for both folds."""
    static_tf = build_static_transformer(continuous, nominal)
    temporal_tf = build_temporal_transformer()

    tr_static = df_tr[STATIC_FEATURES].copy()
    va_static = df_va[STATIC_FEATURES].copy()
    for frame in (tr_static, va_static):  # uniform string dtype for the encoder
        for c in nominal:
            frame[c] = frame[c].where(frame[c].notna(), "MISSING").astype(str).str.strip()

    Xs_tr = static_tf.fit_transform(tr_static).astype(np.float32)
    Xs_va = static_tf.transform(va_static).astype(np.float32)

    Xt_tr = temporal_tf.fit_transform(df_tr[temporal_features]).astype(np.float32)
    Xt_va = temporal_tf.transform(df_va[temporal_features]).astype(np.float32)

    # Sequence tensors: (Batch, TimeSteps, FeaturesPerStep)
    feats_per_step = len(temporal_features) // time_steps
    Xseq_tr = Xt_tr.reshape(-1, time_steps, feats_per_step)
    Xseq_va = Xt_va.reshape(-1, time_steps, feats_per_step)

    # Flat representation for traditional ML
    Xflat_tr = np.hstack([Xs_tr, Xt_tr])
    Xflat_va = np.hstack([Xs_va, Xt_va])
    return Xs_tr, Xs_va, Xseq_tr, Xseq_va, Xflat_tr, Xflat_va


# =============================================================================
# 2. MODEL ARCHITECTURES
# =============================================================================


class StaticMLP(nn.Module):
    """Dense(128) -> BN -> ReLU -> Dropout -> Dense(64)  =>  z_static (64-d)."""

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    """Dense(64) -> ReLU -> Dropout -> Linear(n_classes). Softmax is applied in the
    loss (CrossEntropyLoss) and at inference time."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z):
        return self.net(z)


# ---- Category B: single-branch deep baselines ------------------------------


class StaticMLPOnly(nn.Module):
    def __init__(self, static_dim, seq_feats, n_classes):
        super().__init__()
        self.static = StaticMLP(static_dim)
        self.head = FusionHead(64, n_classes)

    def forward(self, xs, xt):
        return self.head(F.relu(self.static(xs)))


class StandaloneLSTM(nn.Module):
    def __init__(self, static_dim, seq_feats, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(seq_feats, 64, batch_first=True)
        self.head = FusionHead(64, n_classes)

    def forward(self, xs, xt):
        _, (h, _) = self.lstm(xt)
        return self.head(h[-1])


# ---- Category C: hybrid baselines -------------------------------------------


class HybridCNNLSTM(nn.Module):
    """Hybrid Baseline 1: Static MLP + standard (non-causal) Conv1D(32) -> Uni-LSTM(64)."""

    def __init__(self, static_dim, seq_feats, n_classes):
        super().__init__()
        self.static = StaticMLP(static_dim)
        self.conv = nn.Conv1d(seq_feats, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.head = FusionHead(128, n_classes)

    def forward(self, xs, xt):
        zs = self.static(xs)
        c = F.relu(self.conv(xt.transpose(1, 2))).transpose(1, 2)  # (B, T, 32)
        _, (h, _) = self.lstm(c)
        return self.head(torch.cat([zs, h[-1]], dim=1))


class HybridBiLSTM(nn.Module):
    """Hybrid Baseline 2: Static MLP + BiLSTM (32 per direction -> 64-d concat)."""

    def __init__(self, static_dim, seq_feats, n_classes):
        super().__init__()
        self.static = StaticMLP(static_dim)
        self.bilstm = nn.LSTM(seq_feats, 32, batch_first=True, bidirectional=True)
        self.head = FusionHead(128, n_classes)

    def forward(self, xs, xt):
        zs = self.static(xs)
        _, (h, _) = self.bilstm(xt)
        zt = torch.cat([h[-2], h[-1]], dim=1)  # forward || backward
        return self.head(torch.cat([zs, zt], dim=1))


class HybridGRU(nn.Module):
    """Hybrid Baseline 3: Static MLP + GRU(64)."""

    def __init__(self, static_dim, seq_feats, n_classes):
        super().__init__()
        self.static = StaticMLP(static_dim)
        self.gru = nn.GRU(seq_feats, 64, batch_first=True)
        self.head = FusionHead(128, n_classes)

    def forward(self, xs, xt):
        zs = self.static(xs)
        _, h = self.gru(xt)
        return self.head(torch.cat([zs, h[-1]], dim=1))


class HybridTransformer(nn.Module):
    """Hybrid Baseline 4: Static MLP + TransformerEncoderLayer(d_model=feats, nhead=1)
    with learned positional embedding -> mean pool -> Linear(64)."""

    MAX_STEPS = 16

    def __init__(self, static_dim, seq_feats, n_classes):
        super().__init__()
        self.static = StaticMLP(static_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.MAX_STEPS, seq_feats))
        self.encoder = nn.TransformerEncoderLayer(
            d_model=seq_feats, nhead=1, dim_feedforward=64,
            dropout=DROPOUT, batch_first=True,
        )
        self.proj = nn.Linear(seq_feats, 64)
        self.head = FusionHead(128, n_classes)

    def forward(self, xs, xt):
        zs = self.static(xs)
        e = self.encoder(xt + self.pos[:, :xt.size(1)])
        zt = F.relu(self.proj(e.mean(dim=1)))
        return self.head(torch.cat([zs, zt], dim=1))


# ---- Category D: proposed model --------------------------------------------


class CausalResidualBlock(nn.Module):
    """Causal dilated Conv1D (kernel=2, dilation=1) with residual connection.
    Left-padding of (k-1)*d guarantees output at step t depends only on <= t."""

    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1, dropout=DROPOUT):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x: (B, C, T)
        out = self.conv(F.pad(x, (self.left_pad, 0)))
        out = self.dropout(F.relu(out))
        return F.relu(out + self.downsample(x))


class ProposedDualBranch(nn.Module):
    """Static MLP (z_static, 64) || [Causal TCN -> Uni-LSTM(64)] (z_temporal, 64)
    -> Dense(64) -> ReLU -> Dropout -> Softmax output."""

    def __init__(self, static_dim, seq_feats, n_classes, tcn_channels=32):
        super().__init__()
        self.static = StaticMLP(static_dim)
        self.tcn = CausalResidualBlock(seq_feats, tcn_channels, kernel_size=2, dilation=1)
        self.lstm = nn.LSTM(tcn_channels, 64, batch_first=True)
        self.head = FusionHead(128, n_classes)

    def forward(self, xs, xt):
        z_static = self.static(xs)
        c = self.tcn(xt.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        _, (h, _) = self.lstm(c)
        z_temporal = h[-1]
        return self.head(torch.cat([z_static, z_temporal], dim=1))


PROPOSED = "Proposed: MLP + Causal TCN + LSTM"
REFERENCE = "Majority Class (reference)"

# name -> (category, kind, factory). sklearn factories take the class_weight
# argument ("balanced" or None); torch factories are the nn.Module classes.
MODEL_REGISTRY = {
    REFERENCE: ("0: Reference", "sklearn",
                lambda cw: DummyClassifier(strategy="prior")),
    "Logistic Regression": ("A: Traditional ML", "sklearn",
                            lambda cw: LogisticRegression(max_iter=3000, class_weight=cw,
                                                          random_state=SEED)),
    "Random Forest": ("A: Traditional ML", "sklearn",
                      lambda cw: RandomForestClassifier(n_estimators=100, class_weight=cw,
                                                        n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": ("A: Traditional ML", "sklearn",
                             lambda cw: HistGradientBoostingClassifier(class_weight=cw,
                                                                       random_state=SEED)),
    "Static MLP Only": ("B: Single-Branch Deep", "torch", StaticMLPOnly),
    "Standalone LSTM": ("B: Single-Branch Deep", "torch", StandaloneLSTM),
    "Hybrid 1: MLP + CNN + LSTM": ("C: Hybrid Baseline", "torch", HybridCNNLSTM),
    "Hybrid 2: MLP + BiLSTM": ("C: Hybrid Baseline", "torch", HybridBiLSTM),
    "Hybrid 3: MLP + GRU": ("C: Hybrid Baseline", "torch", HybridGRU),
    "Hybrid 4: MLP + Transformer": ("C: Hybrid Baseline", "torch", HybridTransformer),
    PROPOSED: ("D: Proposed", "torch", ProposedDualBranch),
}

# =============================================================================
# 3. TRAINING / INFERENCE UTILITIES
# =============================================================================


def inverse_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """c_y = N / (K * N_y). Classes absent from the training fold get weight 0."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes, dtype=np.float64)
    nz = counts > 0
    w[nz] = len(y) / (n_classes * counts[nz])
    return w.astype(np.float32)


def train_torch_model(model_cls, Xs_tr, Xseq_tr, y_tr, Xs_va, Xseq_va, n_classes, class_w, seed):
    set_seed(seed)
    model = model_cls(Xs_tr.shape[1], Xseq_tr.shape[2], n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ds = TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(Xseq_tr),
                       torch.from_numpy(y_tr.astype(np.int64)))
    gen = torch.Generator().manual_seed(seed)
    # drop_last avoids a size-1 final batch breaking BatchNorm in train mode
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen,
                        drop_last=len(ds) % BATCH_SIZE == 1)

    for _ in range(EPOCHS):
        model.train()
        for xs, xt, yb in loader:
            xs, xt, yb = xs.to(DEVICE), xt.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xs, xt), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xs_va), 1024):
            xs = torch.from_numpy(Xs_va[i:i + 1024]).to(DEVICE)
            xt = torch.from_numpy(Xseq_va[i:i + 1024]).to(DEVICE)
            probs.append(F.softmax(model(xs, xt), dim=1).cpu().numpy())
    return np.vstack(probs)


def train_sklearn_model(factory, X_tr, y_tr, X_va, n_classes, class_weight):
    model = factory(class_weight)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_va)
    # Align columns to the full label space if a class was absent in training
    full = np.zeros((len(X_va), n_classes), dtype=np.float64)
    full[:, model.classes_] = p
    return full


# =============================================================================
# 4. METRICS
# =============================================================================

METRIC_COLS = ["Macro-Track Top-1 Acc (%)", "Macro-Track Macro-F1", "Granular Top-1 Acc (%)",
               "Granular Top-3 Acc (%)", "Granular Macro-F1"]


def compute_metrics(probs, y_true, y_macro_true, label_to_macro_idx, n_macro, granular):
    """`probs` are over the training labels. When the labels are majors (granular=True),
    macro-track probabilities are the sum of each track's member probabilities. When the
    labels are the tracks themselves, the granular metrics are not defined (NaN)."""
    n_labels = probs.shape[1]
    y_pred = probs.argmax(axis=1)

    macro_probs = np.zeros((len(probs), n_macro))
    for label_idx, macro_idx in enumerate(label_to_macro_idx):
        macro_probs[:, macro_idx] += probs[:, label_idx]
    macro_pred = macro_probs.argmax(axis=1)

    m = {
        "Macro-Track Top-1 Acc (%)": 100.0 * accuracy_score(y_macro_true, macro_pred),
        "Macro-Track Macro-F1": f1_score(y_macro_true, macro_pred, average="macro",
                                         labels=np.arange(n_macro), zero_division=0),
        "Granular Top-1 Acc (%)": np.nan,
        "Granular Top-3 Acc (%)": np.nan,
        "Granular Macro-F1": np.nan,
    }
    if granular:
        labels = np.arange(n_labels)
        m["Granular Top-1 Acc (%)"] = 100.0 * accuracy_score(y_true, y_pred)
        m["Granular Top-3 Acc (%)"] = 100.0 * top_k_accuracy_score(y_true, probs, k=3,
                                                                   labels=labels)
        m["Granular Macro-F1"] = f1_score(y_true, y_pred, average="macro", labels=labels,
                                          zero_division=0)
    return m


# =============================================================================
# 5. EXPERIMENT RUNNER
# =============================================================================


def run_experiment(df, label_col=TARGET, label_to_track=None, weighted=True,
                   temporal_features=TEMPORAL_FEATURES, time_steps=TIME_STEPS,
                   summary_csv=SUMMARY_CSV, fold_csv=FOLD_CSV, title="OBJECTIVE 1",
                   oof_npz=None):
    """Stratified 5-fold CV of every model in MODEL_REGISTRY.

    label_col       : column the models are trained on (majors, grouped majors or tracks).
    label_to_track  : dict label -> macro-track; defaults to PROGRAM_TO_TRACK, or the
                      identity when training directly on MACRO_TRACK.
    weighted        : inverse-frequency class weights (torch) / class_weight='balanced'.
    oof_npz         : optional path; saves every model's out-of-fold probabilities
                      (for confusion matrices and per-class analysis).
    """
    t0 = time.time()
    assert len(temporal_features) % time_steps == 0
    granular = label_col != MACRO_TARGET
    if label_to_track is None:
        label_to_track = PROGRAM_TO_TRACK if granular else {m: m for m in df[MACRO_TARGET].unique()}
    primary = "Granular Macro-F1" if granular else "Macro-Track Macro-F1"

    labels = sorted(df[label_col].unique())
    macros = sorted(df[MACRO_TARGET].unique())
    label_idx = {p: i for i, p in enumerate(labels)}
    macro_idx = {m: i for i, m in enumerate(macros)}
    label_to_macro_idx = np.array([macro_idx[label_to_track[p]] for p in labels])
    n_classes, n_macro = len(labels), len(macros)

    y = df[label_col].map(label_idx).values.astype(np.int64)
    y_macro = df[MACRO_TARGET].map(macro_idx).values.astype(np.int64)

    print(f"\n{'#' * 120}\n{title}\n{'#' * 120}")
    print(f"[Setup] labels={label_col} ({n_classes} classes) | weighted={weighted} | "
          f"temporal=(B, {time_steps}, {len(temporal_features) // time_steps}) {temporal_features}")
    min_count = np.bincount(y).min()
    if min_count < N_SPLITS:
        print(f"[CV] WARNING - smallest class has {min_count} samples (< {N_SPLITS} folds); "
              "it will be missing from some validation folds.")

    continuous, nominal = split_static_types(df)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    records = []
    model_names = list(MODEL_REGISTRY)
    oof = np.zeros((len(model_names), len(df), n_classes), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n{'=' * 78}\nFOLD {fold}/{N_SPLITS}  (train={len(tr_idx):,}, val={len(va_idx):,})\n{'=' * 78}")
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        ym_va = y_macro[va_idx]

        Xs_tr, Xs_va, Xseq_tr, Xseq_va, Xflat_tr, Xflat_va = preprocess_fold(
            df_tr, df_va, continuous, nominal, temporal_features, time_steps)
        if weighted:
            class_w = inverse_class_weights(y_tr, n_classes)
        else:
            class_w = (np.bincount(y_tr, minlength=n_classes) > 0).astype(np.float32)

        for name, (category, kind, factory) in MODEL_REGISTRY.items():
            ts = time.time()
            if kind == "sklearn":
                probs = train_sklearn_model(factory, Xflat_tr, y_tr, Xflat_va, n_classes,
                                            "balanced" if weighted else None)
            else:
                probs = train_torch_model(factory, Xs_tr, Xseq_tr, y_tr, Xs_va, Xseq_va,
                                          n_classes, class_w, seed=SEED + fold)
            oof[model_names.index(name), va_idx] = probs
            m = compute_metrics(probs, y_va, ym_va, label_to_macro_idx, n_macro, granular)
            records.append({"Fold": fold, "Model": name, "Category": category, **m})
            line = (f"  {name:<36s} | Track {m['Macro-Track Top-1 Acc (%)']:6.2f}% "
                    f"F1 {m['Macro-Track Macro-F1']:.4f}")
            if granular:
                line += (f" | Top1 {m['Granular Top-1 Acc (%)']:6.2f}% | "
                         f"Top3 {m['Granular Top-3 Acc (%)']:6.2f}% | "
                         f"F1 {m['Granular Macro-F1']:.4f}")
            print(f"{line} | {time.time() - ts:5.1f}s")

    fold_df = pd.DataFrame(records)
    fold_df.to_csv(fold_csv, index=False)
    if oof_npz:
        np.savez_compressed(oof_npz, probs=oof, models=np.array(model_names),
                            y=y, y_macro=y_macro, labels=np.array(labels),
                            macros=np.array(macros), label_to_macro_idx=label_to_macro_idx)

    # ---- Statistical significance: paired t-test on the primary metric ------
    wide = fold_df.pivot(index="Fold", columns="Model", values=primary)
    proposed_scores = wide[PROPOSED].values
    p_col = f"p-value vs Proposed ({primary})"
    metric_cols = METRIC_COLS if granular else METRIC_COLS[:2]

    rows = []
    for name, (category, _, _) in MODEL_REGISTRY.items():
        sub = fold_df[fold_df["Model"] == name]
        row = {"Category": category, "Model": name}
        for col in metric_cols:
            mu, sd = sub[col].mean(), sub[col].std(ddof=1)
            row[col] = f"{mu:.4f} ± {sd:.4f}" if "F1" in col else f"{mu:.2f} ± {sd:.2f}"
            row[col + " [mean]"] = mu
        if name == PROPOSED:
            row[p_col] = "—"
        else:
            other = wide[name].values
            if np.allclose(proposed_scores - other, 0):
                p = 1.0
            else:
                _, p = stats.ttest_rel(proposed_scores, other)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            row[p_col] = f"{p:.4g} ({sig})"
        rows.append(row)

    summary = pd.DataFrame(rows)
    export_cols = ["Category", "Model"] + metric_cols + [p_col]
    summary[export_cols].to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # ---- Display ----------------------------------------------------------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)
    print(f"\n{'=' * 120}\n{title} - SUMMARY ({N_SPLITS}-fold stratified CV, Mean ± Std)\n{'=' * 120}")
    print(summary[export_cols].to_string(index=False))
    print(f"\nSignificance: paired t-test (scipy.stats.ttest_rel) on fold-wise {primary}, "
          f"df={N_SPLITS - 1}. * p<0.05, ** p<0.01, *** p<0.001, n.s. not significant.")

    best = summary[summary["Model"] != REFERENCE].sort_values(
        primary + " [mean]", ascending=False).iloc[0]
    prop = summary[summary["Model"] == PROPOSED].iloc[0]
    print(f"\nBest {primary} model: {best['Model']} ({best[primary]})")
    track = prop["Macro-Track Top-1 Acc (%) [mean]"]
    msg = f"Proposed model targets: Macro-Track Top-1 > 80% -> {track:.2f}% " \
          f"({'MET' if track > 80 else 'NOT MET'})"
    if granular:
        top3 = prop["Granular Top-3 Acc (%) [mean]"]
        msg += f"; Granular Top-3 > 85% -> {top3:.2f}% ({'MET' if top3 > 85 else 'NOT MET'})"
    print(msg)
    print(f"\nSaved: {summary_csv} (summary), {fold_csv} (fold-wise raw metrics)")
    print(f"Runtime: {(time.time() - t0) / 60:.1f} min")
    return summary, fold_df


def main():
    print(f"[Setup] Device: {DEVICE} | torch {torch.__version__}")
    df = load_and_prepare(DATA_PATH)
    return run_experiment(df)


if __name__ == "__main__":
    main()
