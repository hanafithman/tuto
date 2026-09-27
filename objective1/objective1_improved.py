"""
Objective 1 - improved, leakage-safe experiment.

What this adds over objective1_hybrid_model.py (all fitted inside each training fold only):
  * 5-fold stratified cross-validation -> mean +/- std instead of a single split.
  * Richer PRE-ENROLMENT features: remaining household assets, JOB, and SCHOOL_NAME
    (high-cardinality -> cross-fitted multiclass target encoding).
  * Models: gradient boosting (HGB), the dual-branch hybrid network trained longer with
    early stopping + LR schedule, and a probability-averaging ensemble of the two.
  * Macro-F1 decision rule: per-class logit offsets tuned on an inner validation split
    (never on the test fold).
  * A separately-labelled "enrolment-time" setting that adds UNIVERSITY. The university is
    chosen together with the program, so it is NOT a pre-enrolment predictor; it is reported
    only as an upper reference.

Usage:
    python objective1_improved.py          # expects dataset.csv in the current directory
Outputs:
    improved_results_summary.csv, improved_results_per_class.csv,
    confusion_matrix_improved_<setting>.csv
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, TargetEncoder
from torch.utils.data import DataLoader

import objective1_hybrid_model as base

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 128
MAX_CLASS_WEIGHT = 10.0

EXTRA_CATEGORICAL = ["TV", "WASHING_MCH", "MIC_OVEN", "DVD", "FRESH", "PHONE", "MOBILE", "JOB"]

SETTINGS = {
    # name: (categorical one-hot cols, high-cardinality target-encoded cols)
    "A_original_features": (base.STATIC_CATEGORICAL, []),
    "B_all_pre_enrolment": (base.STATIC_CATEGORICAL + EXTRA_CATEGORICAL, ["SCHOOL_NAME"]),
    "C_plus_UNIVERSITY_enrolment_time": (base.STATIC_CATEGORICAL + EXTRA_CATEGORICAL,
                                         ["SCHOOL_NAME", "UNIVERSITY"]),
}


def load():
    df = base.add_macro_track(base.load_and_clean(base.DATA_PATH))
    for col in EXTRA_CATEGORICAL + ["SCHOOL_NAME", "UNIVERSITY"]:
        df[col] = df[col].fillna("Unknown")
    return df


# ----------------------------------------------------------------------------------------------
# Features (fit on training rows only)
# ----------------------------------------------------------------------------------------------
class Features:
    def __init__(self, cat_cols, te_cols):
        self.cat_cols, self.te_cols = cat_cols, te_cols
        self.ct = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), base.STATIC_NUMERIC),
        ])
        self.te = TargetEncoder(target_type="multiclass", cv=KFold(5, shuffle=True, random_state=SEED)) if te_cols else None
        self.te_scaler = StandardScaler() if te_cols else None
        self.temporal_scaler = StandardScaler()

    def _impute(self, df):
        df = df.copy()
        df[base.STATIC_NUMERIC] = df[base.STATIC_NUMERIC].fillna(self.num_med)
        df[base.TEMPORAL_FEATURES] = df[base.TEMPORAL_FEATURES].fillna(self.tmp_med)
        return df

    def fit_transform(self, df, y):
        self.num_med = df[base.STATIC_NUMERIC].median()
        self.tmp_med = df[base.TEMPORAL_FEATURES].median()
        df = self._impute(df)
        static = [self.ct.fit_transform(df)]
        if self.te is not None:
            # fit_transform uses internal cross-fitting so training rows never see their own label
            te = self.te.fit_transform(df[self.te_cols], y)
            static.append(self.te_scaler.fit_transform(te))
        temporal = self.temporal_scaler.fit_transform(df[base.TEMPORAL_FEATURES].to_numpy(float))
        return self._pack(static, temporal)

    def transform(self, df):
        df = self._impute(df)
        static = [self.ct.transform(df)]
        if self.te is not None:
            static.append(self.te_scaler.transform(self.te.transform(df[self.te_cols])))
        temporal = self.temporal_scaler.transform(df[base.TEMPORAL_FEATURES].to_numpy(float))
        return self._pack(static, temporal)

    @staticmethod
    def _pack(static, temporal):
        xs = np.hstack(static).astype(np.float32)
        xt = temporal.astype(np.float32).reshape(-1, base.TIME_STEPS, base.FEATURES_PER_STEP)
        return xs, xt


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def fit_hgb(xs, xt, y, xs_va, xt_va, y_va, num_classes):
    X = np.hstack([xs, xt.reshape(len(xt), -1)])
    X_val = np.hstack([xs_va, xt_va.reshape(len(xt_va), -1)])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, class_weight="balanced", early_stopping=True,
        n_iter_no_change=30, random_state=SEED,
    ).fit(X, y, X_val=X_val, y_val=y_va)  # early stopping on the inner validation split

    def predict(xs_, xt_):
        p = np.zeros((len(xs_), num_classes))
        p[:, clf.classes_] = clf.predict_proba(np.hstack([xs_, xt_.reshape(len(xt_), -1)]))
        return p
    return predict


def fit_hybrid(xs_tr, xt_tr, y_tr, xs_va, xt_va, y_va, num_classes, device):
    base.set_seed(SEED)
    tr_loader = DataLoader(base.StudentPathwayDataset(xs_tr, xt_tr, y_tr), batch_size=BATCH_SIZE,
                           shuffle=True, generator=torch.Generator().manual_seed(SEED))
    va_loader = DataLoader(base.StudentPathwayDataset(xs_va, xt_va, y_va), batch_size=512)
    model = base.DualBranchHybridModel(xs_tr.shape[1], base.FEATURES_PER_STEP, num_classes,
                                       dropout=0.3).to(device)
    # c_y = N / (K * N_y), capped: with 21 programs a 1-student class would otherwise get
    # weight ~400 and the network collapses onto the rarest classes. No-op for the 4 tracks.
    weights = base.inverse_frequency_weights(y_tr, num_classes).clamp(max=MAX_CLASS_WEIGHT).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        base.run_epoch(model, tr_loader, criterion, device, optimizer)
        va_loss, _ = base.run_epoch(model, va_loader, criterion, device)
        scheduler.step(va_loss)
        if va_loss < best_loss - 1e-4:
            best_loss, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()

    @torch.no_grad()
    def predict(xs_, xt_):
        logits = model(torch.as_tensor(xs_).to(device), torch.as_tensor(xt_).to(device))
        return torch.softmax(logits, 1).cpu().numpy()
    return predict


def tune_offsets(probs, y, num_classes, grid=np.linspace(-2, 2, 41), passes=3):
    """Per-class additive log-prob offsets that maximise macro-F1 on a validation set."""
    logp = np.log(probs + 1e-9)
    present = np.unique(y)
    b = np.zeros(num_classes)
    score = lambda bb: f1_score(y, (logp + bb).argmax(1), labels=present, average="macro",
                                zero_division=0)
    best = score(b)
    for _ in range(passes):
        for k in range(num_classes):
            for v in grid:
                cand = b.copy()
                cand[k] = v
                s = score(cand)
                if s > best + 1e-6:
                    best, b = s, cand
    return b


# ----------------------------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------------------------
def evaluate(df, target, setting, device):
    cat_cols, te_cols = SETTINGS[setting]
    enc = LabelEncoder().fit(df[target])
    names = list(enc.classes_)
    K = len(names)
    y_all = enc.transform(df[target])
    strata = base._safe_strata(df["ACADEMIC_PROGRAM"].to_numpy(), min_count=N_FOLDS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rows, cms = [], {}
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, strata), 1):
        inner_strata = base._safe_strata(strata[tr_idx])
        fit_idx, val_idx = train_test_split(tr_idx, test_size=0.15, random_state=SEED,
                                            stratify=inner_strata)
        feats = Features(cat_cols, te_cols)
        xs_fit, xt_fit = feats.fit_transform(df.iloc[fit_idx], y_all[fit_idx])
        xs_val, xt_val = feats.transform(df.iloc[val_idx])
        xs_te, xt_te = feats.transform(df.iloc[te_idx])
        y_fit, y_val, y_te = y_all[fit_idx], y_all[val_idx], y_all[te_idx]

        models = {
            "HGB": fit_hgb(xs_fit, xt_fit, y_fit, xs_val, xt_val, y_val, K),
            "Hybrid": fit_hybrid(xs_fit, xt_fit, y_fit, xs_val, xt_val, y_val, K, device),
        }
        val_p = {m: f(xs_val, xt_val) for m, f in models.items()}
        te_p = {m: f(xs_te, xt_te) for m, f in models.items()}
        val_p["Ensemble"] = (val_p["HGB"] + val_p["Hybrid"]) / 2
        te_p["Ensemble"] = (te_p["HGB"] + te_p["Hybrid"]) / 2

        present = np.unique(y_te)
        for m in te_p:
            for rule in ("argmax", "macroF1-tuned"):
                logp = np.log(te_p[m] + 1e-9)
                if rule == "macroF1-tuned":
                    logp = logp + tune_offsets(val_p[m], y_val, K)
                pred = logp.argmax(1)
                per_class = f1_score(y_te, pred, labels=np.arange(K), average=None, zero_division=0)
                rows.append({
                    "target": target, "setting": setting, "model": m, "rule": rule, "fold": fold,
                    "accuracy": accuracy_score(y_te, pred),
                    "macro_f1": f1_score(y_te, pred, labels=present, average="macro", zero_division=0),
                    "top2_acc": top_k_accuracy_score(y_te, te_p[m], k=2, labels=np.arange(K)),
                    "top3_acc": top_k_accuracy_score(y_te, te_p[m], k=3, labels=np.arange(K)),
                    **{f"F1::{n}": v for n, v in zip(names, per_class)},
                })
                key = (m, rule)
                cms[key] = cms.get(key, 0) + confusion_matrix(y_te, pred, labels=np.arange(K))
        print(f"  [{target} | {setting}] fold {fold}/{N_FOLDS} done", flush=True)
    return pd.DataFrame(rows), cms, names


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load()
    print(f"Loaded {len(df)} records. Device: {device}")

    all_rows, best = [], {}
    for target in ("MACRO_TRACK", "ACADEMIC_PROGRAM"):
        for setting in SETTINGS:
            res, cms, names = evaluate(df, target, setting, device)
            all_rows.append(res)
            agg = res.groupby(["model", "rule"])["macro_f1"].mean()
            m, r = agg.idxmax()
            cm = pd.DataFrame(cms[(m, r)], index=names, columns=names)
            cm.to_csv(f"confusion_matrix_improved_{target}_{setting}.csv")
            best[(target, setting)] = (m, r, agg.max())

    res = pd.concat(all_rows, ignore_index=True)
    metrics = ["accuracy", "macro_f1", "top2_acc", "top3_acc"]
    summary = res.groupby(["target", "setting", "model", "rule"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv("improved_results_summary.csv", index=False)

    f1_cols = [c for c in res.columns if c.startswith("F1::")]
    per_class = (res.groupby(["target", "setting", "model", "rule"])[f1_cols].mean()
                 .dropna(axis=1, how="all").reset_index())
    per_class.to_csv("improved_results_per_class.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for target in ("MACRO_TRACK", "ACADEMIC_PROGRAM"):
        print("\n" + "=" * 100)
        print(f"{target}: {N_FOLDS}-fold cross-validation, mean +/- std on held-out folds")
        print("=" * 100)
        s = summary[summary.target == target]
        for _, r in s.iterrows():
            print(f"{r.setting:34s} {r.model:9s} {r.rule:14s} "
                  f"acc {r.accuracy_mean:.3f}+/-{r.accuracy_std:.3f}  "
                  f"macroF1 {r.macro_f1_mean:.3f}+/-{r.macro_f1_std:.3f}  "
                  f"top2 {r.top2_acc_mean:.3f}  top3 {r.top3_acc_mean:.3f}")
        if target == "MACRO_TRACK":
            print("\nPer-class F1 of the best (model, rule) per setting:")
            for setting in SETTINGS:
                m, r, _ = best[(target, setting)]
                row = per_class[(per_class.target == target) & (per_class.setting == setting)
                                & (per_class.model == m) & (per_class.rule == r)]
                cols = [c for c in f1_cols if c in row.columns and row[c].notna().all()]
                print(f"  {setting} [{m}, {r}]: " +
                      ", ".join(f"{c[4:]}={row[c].iloc[0]:.3f}" for c in cols))


if __name__ == "__main__":
    main()
