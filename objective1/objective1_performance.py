"""
Objective 1 (reframed) - predict end-of-degree Saber Pro performance: above vs below the median
global score (G_SC), from information available before the exam.

Leakage controls
  * G_SC, PERCENTILE, 2ND_DECILE, QUARTILE are derived from the target -> never used as features.
  * Saber Pro sub-scores (QR_PRO, CR_PRO, ...) are part of the same exam -> never used.
  * The median threshold is computed on each training fold only and applied to its test fold.
  * All encoders/scalers are fitted on training rows only; the decision threshold is tuned on an
    inner validation split, never on the test fold.

Feature settings
  A_saber11_socioeconomic : the 14 socioeconomic variables + 5 Saber 11 subject scores
  B_all_pre_enrolment     : A + remaining household assets, JOB, SCHOOL_NAME (target-encoded)
  C_at_enrolment          : B + UNIVERSITY and ACADEMIC_PROGRAM (known once the student enrols,
                            i.e. still before the Saber Pro exam)

Models: gradient boosting (HGB), the dual-branch hybrid network (static MLP + causal TCN + uni-LSTM,
temporal branch over the 5 Saber 11 subjects as a 5-step sequence), and their average.

Usage:  python objective1_performance.py     (expects dataset.csv in the current directory)
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from torch.utils.data import DataLoader

import objective1_hybrid_model as base

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 128
CLASS_NAMES = ["Below median", "Above median"]

SABER11 = ["MAT_S11", "CR_S11", "CC_S11", "BIO_S11", "ENG_S11"]
EXTRA_CATEGORICAL = ["TV", "WASHING_MCH", "MIC_OVEN", "DVD", "FRESH", "PHONE", "MOBILE", "JOB"]
SETTINGS = {
    "A_saber11_socioeconomic": (base.STATIC_CATEGORICAL, []),
    "B_all_pre_enrolment": (base.STATIC_CATEGORICAL + EXTRA_CATEGORICAL, ["SCHOOL_NAME"]),
    "C_at_enrolment": (base.STATIC_CATEGORICAL + EXTRA_CATEGORICAL,
                       ["SCHOOL_NAME", "UNIVERSITY", "ACADEMIC_PROGRAM"]),
}


def load():
    df = base.load_and_clean(base.DATA_PATH)
    for col in EXTRA_CATEGORICAL + ["SCHOOL_NAME", "UNIVERSITY"]:
        df[col] = df[col].fillna("Unknown")
    df["G_SC"] = pd.to_numeric(df["G_SC"], errors="coerce")
    return df.dropna(subset=["G_SC"]).reset_index(drop=True)


class Features:
    """One-hot + scaled static features, target-encoded high-cardinality columns, and the
    Saber 11 scores as a (B, 5, 1) sequence. Fitted on training rows only."""

    def __init__(self, cat_cols, te_cols):
        self.cat_cols, self.te_cols = cat_cols, te_cols
        self.ct = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), base.STATIC_NUMERIC),
        ])
        self.te = (TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=SEED))
                   if te_cols else None)
        self.te_scaler = StandardScaler()
        self.seq_scaler = StandardScaler()

    def _impute(self, df):
        df = df.copy()
        df[base.STATIC_NUMERIC] = df[base.STATIC_NUMERIC].fillna(self.num_med)
        df[SABER11] = df[SABER11].fillna(self.seq_med)
        return df

    def fit_transform(self, df, y):
        self.num_med = df[base.STATIC_NUMERIC].median()
        self.seq_med = df[SABER11].median()
        df = self._impute(df)
        static = [self.ct.fit_transform(df)]
        if self.te is not None:
            static.append(self.te_scaler.fit_transform(self.te.fit_transform(df[self.te_cols], y)))
        seq = self.seq_scaler.fit_transform(df[SABER11].to_numpy(float))
        return self._pack(static, seq)

    def transform(self, df):
        df = self._impute(df)
        static = [self.ct.transform(df)]
        if self.te is not None:
            static.append(self.te_scaler.transform(self.te.transform(df[self.te_cols])))
        seq = self.seq_scaler.transform(df[SABER11].to_numpy(float))
        return self._pack(static, seq)

    @staticmethod
    def _pack(static, seq):
        return (np.hstack(static).astype(np.float32),
                seq.astype(np.float32).reshape(-1, len(SABER11), 1))


def fit_hgb(xs, xt, y, xs_va, xt_va, y_va):
    flat = lambda a, b: np.hstack([a, b.reshape(len(b), -1)])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, early_stopping=True, n_iter_no_change=30, random_state=SEED,
    ).fit(flat(xs, xt), y, X_val=flat(xs_va, xt_va), y_val=y_va)
    return lambda a, b: clf.predict_proba(flat(a, b))[:, 1]


def fit_hybrid(xs, xt, y, xs_va, xt_va, y_va, device):
    base.set_seed(SEED)
    tr = DataLoader(base.StudentPathwayDataset(xs, xt, y), batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))
    va = DataLoader(base.StudentPathwayDataset(xs_va, xt_va, y_va), batch_size=512)
    model = base.DualBranchHybridModel(xs.shape[1], 1, 2, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=base.inverse_frequency_weights(y, 2).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    best, state, bad = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        base.run_epoch(model, tr, criterion, device, optimizer)
        loss, _ = base.run_epoch(model, va, criterion, device)
        scheduler.step(loss)
        if loss < best - 1e-4:
            best, bad = loss, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def predict(a, b):
        logits = model(torch.as_tensor(a).to(device), torch.as_tensor(b).to(device))
        return torch.softmax(logits, 1)[:, 1].cpu().numpy()
    return predict


def tune_threshold(p, y):
    grid = np.linspace(0.2, 0.8, 61)
    scores = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    return grid[int(np.argmax(scores))]


def evaluate(df, setting, device):
    cat_cols, te_cols = SETTINGS[setting]
    strata = base._safe_strata(df["ACADEMIC_PROGRAM"].to_numpy(), min_count=N_FOLDS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, cms = [], {}
    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, strata), 1):
        threshold = df["G_SC"].iloc[tr_idx].median()          # training-fold median only
        y_all = (df["G_SC"] >= threshold).astype(int).to_numpy()
        fit_idx, val_idx = train_test_split(tr_idx, test_size=0.15, random_state=SEED,
                                            stratify=y_all[tr_idx])
        feats = Features(cat_cols, te_cols)
        xs_f, xt_f = feats.fit_transform(df.iloc[fit_idx], y_all[fit_idx])
        xs_v, xt_v = feats.transform(df.iloc[val_idx])
        xs_t, xt_t = feats.transform(df.iloc[te_idx])
        y_f, y_v, y_t = y_all[fit_idx], y_all[val_idx], y_all[te_idx]

        models = {"HGB": fit_hgb(xs_f, xt_f, y_f, xs_v, xt_v, y_v),
                  "Hybrid": fit_hybrid(xs_f, xt_f, y_f, xs_v, xt_v, y_v, device)}
        pv = {k: f(xs_v, xt_v) for k, f in models.items()}
        pt = {k: f(xs_t, xt_t) for k, f in models.items()}
        pv["Ensemble"], pt["Ensemble"] = (pv["HGB"] + pv["Hybrid"]) / 2, (pt["HGB"] + pt["Hybrid"]) / 2

        for m in pt:
            t = tune_threshold(pv[m], y_v)
            pred = (pt[m] >= t).astype(int)
            f1s = f1_score(y_t, pred, average=None)
            rows.append({
                "setting": setting, "model": m, "fold": fold, "median_G_SC": threshold,
                "threshold": t, "accuracy": accuracy_score(y_t, pred),
                "macro_f1": f1s.mean(), "roc_auc": roc_auc_score(y_t, pt[m]),
                "F1_below": f1s[0], "F1_above": f1s[1],
                "precision_below": precision_score(y_t, pred, pos_label=0),
                "precision_above": precision_score(y_t, pred, pos_label=1),
                "recall_below": recall_score(y_t, pred, pos_label=0),
                "recall_above": recall_score(y_t, pred, pos_label=1),
            })
            cms[m] = cms.get(m, 0) + confusion_matrix(y_t, pred, labels=[0, 1])
        print(f"  [{setting}] fold {fold}/{N_FOLDS} done", flush=True)
    return pd.DataFrame(rows), cms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load()
    print(f"Loaded {len(df)} records. Device: {device}")
    print(f"Target: Saber Pro global score (G_SC) >= training-fold median "
          f"(overall median {df['G_SC'].median():.0f})")

    results, all_cms = [], {}
    for setting in SETTINGS:
        res, cms = evaluate(df, setting, device)
        results.append(res)
        for m, cm in cms.items():
            all_cms[(setting, m)] = cm
            pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
                f"confusion_matrix_performance_{setting}_{m}.csv")
    res = pd.concat(results, ignore_index=True)
    res.to_csv("performance_results_folds.csv", index=False)

    metrics = ["accuracy", "macro_f1", "roc_auc", "F1_below", "F1_above",
               "precision_below", "precision_above", "recall_below", "recall_above"]
    summary = res.groupby(["setting", "model"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv("performance_results_summary.csv", index=False)

    print("\n" + "=" * 110)
    print(f"Saber Pro above vs below median - {N_FOLDS}-fold cross-validation, mean +/- std")
    print("=" * 110)
    for _, r in summary.iterrows():
        ok = "YES" if min(r.F1_below_mean, r.F1_above_mean) > 0.8 else "no"
        print(f"{r.setting:25s} {r.model:9s} acc {r.accuracy_mean:.3f}+/-{r.accuracy_std:.3f}  "
              f"macroF1 {r.macro_f1_mean:.3f}+/-{r.macro_f1_std:.3f}  AUC {r.roc_auc_mean:.3f}  "
              f"F1 below {r.F1_below_mean:.3f}+/-{r.F1_below_std:.3f}  "
              f"F1 above {r.F1_above_mean:.3f}+/-{r.F1_above_std:.3f}  both>0.8: {ok}")
    print("\nConfusion matrices (summed over folds; rows = actual, cols = predicted "
          "[below, above]):")
    for (s, m), cm in all_cms.items():
        print(f"  {s:25s} {m:9s} {cm.tolist()}")


if __name__ == "__main__":
    main()
