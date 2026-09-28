"""
Objective 1 - publication-grade evaluation (RO1 predictive validity, RO2 fairness,
RO3 interpretability, plus deep-model seed stability), following the
edm-pathway-fairness-analysis protocol (helpers in edm_kernel.py).

Protocol
  * Leakage-safe features only: socioeconomic/household variables, high-school name
    (target-encoded inside each fold) and the 5 Saber 11 subject scores. Excluded because
    they encode the outcome or post-decision information: G_SC, PERCENTILE, 2ND_DECILE,
    QUARTILE, all *_PRO Saber Pro scores, UNIVERSITY, SEL, SEL_IHE, Cod_SPro, COD_S11.
  * 10-fold stratified CV (shuffle, fixed seed); all imputation/encoding/scaling fitted on the
    training fold only; deep models early-stop on an inner validation split of the training fold.
  * Models: majority class, logistic regression, random forest, gradient boosting (HGB),
    the dual-branch hybrid (static MLP + causal TCN + uni-LSTM; temporal branch = 5 Saber 11
    subjects as a 5-step sequence) averaged over N seeds, and single-branch ablations.
  * Primary metric macro-F1, with balanced accuracy and macro-recall; bootstrap 95% CIs over
    fold means; paired Wilcoxon signed-rank tests across folds (hybrid vs each model).
  * RO2: macro-F1 / balanced-accuracy / macro-recall gaps across GENDER, STRATUM, SISBEN,
    SCHOOL_NAT, SCHOOL_TYPE on out-of-fold predictions; permutation p-values; bootstrap CI on
    the gap difference hybrid - HGB.
  * RO3: permutation importance (drop in held-out macro-F1) for the random forest and the
    hybrid; exact TreeExplainer SHAP for the random forest; rank agreement (Spearman).

Usage
    python objective1_rigorous.py --target MACRO_TRACK
    python objective1_rigorous.py --target ACADEMIC_PROGRAM
Outputs go to rigorous_<TARGET>/.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, TargetEncoder
from torch.utils.data import DataLoader

import edm_kernel as K
import objective1_hybrid_model as base

warnings.filterwarnings("ignore")

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 8
MAX_CLASS_WEIGHT = 10.0

CAT_COLS = base.STATIC_CATEGORICAL + ["TV", "WASHING_MCH", "MIC_OVEN", "DVD", "FRESH",
                                      "PHONE", "MOBILE", "JOB"]
NUM_COLS = ["PEOPLE_HOUSE"]
TE_COLS = ["SCHOOL_NAME"]
SABER11 = ["MAT_S11", "CR_S11", "CC_S11", "BIO_S11", "ENG_S11"]
RAW_FEATURES = CAT_COLS + NUM_COLS + TE_COLS + SABER11
PROTECTED = ["GENDER", "STRATUM", "SISBEN", "SCHOOL_NAT", "SCHOOL_TYPE"]


# ----------------------------------------------------------------------------------------------
# Data & fold-internal preprocessing
# ----------------------------------------------------------------------------------------------
def load():
    df = base.add_macro_track(base.load_and_clean(base.DATA_PATH))
    for c in CAT_COLS + TE_COLS:
        df[c] = df[c].fillna("Unknown")
    return df


class Prep:
    """Static block (one-hot categoricals, scaled numerics, scaled target-encoded school) and
    temporal block (scaled Saber 11 scores as a (B, 5, 1) sequence). Fit on training rows only."""

    def __init__(self):
        self.static = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUM_COLS),
            ("te", Pipeline([("te", TargetEncoder(target_type="multiclass",
                                                  cv=KFold(5, shuffle=True, random_state=SEED))),
                             ("sc", StandardScaler())]), TE_COLS),
        ])
        self.seq = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])

    def fit(self, df, y):
        self.static.fit(df[CAT_COLS + NUM_COLS + TE_COLS], y)
        self.seq.fit(df[SABER11])
        self.feature_names = list(self.static.get_feature_names_out()) + [f"seq__{c}" for c in SABER11]
        return self

    def transform(self, df):
        xs = self.static.transform(df[CAT_COLS + NUM_COLS + TE_COLS]).astype(np.float32)
        xt = self.seq.transform(df[SABER11]).astype(np.float32).reshape(-1, len(SABER11), 1)
        return xs, xt

    def flat(self, df):
        xs, xt = self.transform(df)
        return np.hstack([xs, xt.reshape(len(xt), -1)])


def raw_feature_of(name):
    """Map a transformed column name back to its raw feature."""
    body = name.split("__", 1)[1]
    for raw in sorted(RAW_FEATURES, key=len, reverse=True):
        if body == raw or body.startswith(raw + "_"):
            return raw
    raise KeyError(name)


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
class StaticOnly(nn.Module):
    def __init__(self, static_dim, num_classes, dropout=0.3):
        super().__init__()
        self.branch = base.StaticMLP(static_dim, dropout)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, num_classes))

    def forward(self, xs, xt):
        return self.head(self.branch(xs))


class TemporalOnly(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.branch = base.TemporalEncoder(1, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, num_classes))

    def forward(self, xs, xt):
        return self.head(self.branch(xt))


def make_deep(kind, static_dim, num_classes):
    if kind == "Hybrid":
        return base.DualBranchHybridModel(static_dim, 1, num_classes, dropout=0.3)
    if kind == "StaticMLP_only":
        return StaticOnly(static_dim, num_classes)
    if kind == "TemporalTCN_LSTM_only":
        return TemporalOnly(num_classes)
    raise ValueError(kind)


def fit_deep(kind, seed, xs, xt, y, xs_va, xt_va, y_va, num_classes, device):
    base.set_seed(seed)
    tr = DataLoader(base.StudentPathwayDataset(xs, xt, y), batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    va = DataLoader(base.StudentPathwayDataset(xs_va, xt_va, y_va), batch_size=1024)
    model = make_deep(kind, xs.shape[1], num_classes).to(device)
    w = base.inverse_frequency_weights(y, num_classes).clamp(max=MAX_CLASS_WEIGHT).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, state, bad = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        base.run_epoch(model, tr, crit, device, opt)
        loss, _ = base.run_epoch(model, va, crit, device)
        sched.step(loss)
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
    def proba(a, b):
        out = []
        for i in range(0, len(a), 4096):
            logits = model(torch.as_tensor(a[i:i + 4096]).to(device),
                           torch.as_tensor(b[i:i + 4096]).to(device))
            out.append(torch.softmax(logits, 1).cpu().numpy())
        return np.concatenate(out)
    return proba


def sklearn_models():
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "LogReg": LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                               max_features="sqrt", class_weight="balanced_subsample",
                                               n_jobs=4, random_state=SEED),
        # early_stopping=False: the built-in stratified validation split fails on 1-student programs
        "HGB": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              early_stopping=False, class_weight="balanced",
                                              random_state=SEED),
    }


def full_proba(clf, X, num_classes):
    p = np.zeros((len(X), num_classes))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


# ----------------------------------------------------------------------------------------------
# RO3 helpers
# ----------------------------------------------------------------------------------------------
def permutation_importance_raw(predict_from_df, df_te, y_te, n_repeats, rng):
    """Drop in held-out macro-F1 when one RAW feature is shuffled (re-encoded via the fold's
    fitted preprocessing inside predict_from_df)."""
    labels = np.unique(y_te)
    score = lambda d: f1_score(y_te, predict_from_df(d).argmax(1), labels=labels,
                               average="macro", zero_division=0)
    base_score = score(df_te)
    rows = []
    for f in RAW_FEATURES:
        drops = []
        for _ in range(n_repeats):
            d = df_te.copy()
            d[f] = rng.permutation(d[f].to_numpy())
            drops.append(base_score - score(d))
        rows.append({"feature": f, "importance_mean": float(np.mean(drops)),
                     "importance_std": float(np.std(drops))})
    return pd.DataFrame(rows)


def shap_raw(rf, X_sample, names):
    import shap
    sv = shap.TreeExplainer(rf).shap_values(X_sample, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == X_sample.shape[0]:      # (n, features, classes)
        per_col = np.abs(sv).sum(axis=2).mean(axis=0)
    else:                                                        # (classes, n, features)
        per_col = np.abs(sv).sum(axis=0).mean(axis=0)
    s = pd.Series(per_col, index=names).groupby(lambda n: raw_feature_of(n)).sum()
    return s.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------
def boot_ci(values, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return float("nan"), 1.0
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="MACRO_TRACK", choices=["MACRO_TRACK", "ACADEMIC_PROGRAM"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the hybrid model")
    ap.add_argument("--ablation-seeds", type=int, default=3)
    ap.add_argument("--perm-repeats", type=int, default=10)
    ap.add_argument("--shap-rows-per-fold", type=int, default=150)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    out = f"rigorous_{args.target}"
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load()
    enc = LabelEncoder().fit(df[args.target])
    names = list(enc.classes_)
    KC = len(names)
    y = enc.transform(df[args.target])
    print(f"{args.target}: {len(df)} students, {KC} classes, {args.folds}-fold CV, "
          f"hybrid seeds={args.seeds}", flush=True)

    strata = base._safe_strata(df["ACADEMIC_PROGRAM"].to_numpy(), min_count=args.folds)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    deep_kinds = {"Hybrid": args.seeds, "StaticMLP_only": args.ablation_seeds,
                  "TemporalTCN_LSTM_only": args.ablation_seeds}
    model_names = list(sklearn_models()) + list(deep_kinds)

    oof = {m: np.zeros((len(df), KC)) for m in model_names}
    hybrid_seed_oof = np.zeros((args.seeds, len(df), KC))
    fold_of = np.zeros(len(df), int)
    fold_rows, perm_rows, shap_rows = [], [], []
    rng = np.random.default_rng(SEED)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, strata), 1):
        fold_of[te_idx] = fold
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        prep = Prep().fit(df_tr, y_tr)
        X_tr, X_te = prep.flat(df_tr), prep.flat(df_te)

        # --- classical models (fit on the full training fold)
        fitted = {}
        for m, clf in sklearn_models().items():
            clf.fit(X_tr, y_tr)
            fitted[m] = clf
            oof[m][te_idx] = full_proba(clf, X_te, KC)

        # --- deep models: inner validation split for early stopping; preprocessing refit on the
        #     inner-training part so the validation rows stay unseen
        fit_idx, val_idx = train_test_split(np.arange(len(tr_idx)), test_size=0.15,
                                            random_state=SEED, stratify=base._safe_strata(strata[tr_idx]))
        dprep = Prep().fit(df_tr.iloc[fit_idx], y_tr[fit_idx])
        xs_f, xt_f = dprep.transform(df_tr.iloc[fit_idx])
        xs_v, xt_v = dprep.transform(df_tr.iloc[val_idx])
        xs_t, xt_t = dprep.transform(df_te)
        hybrid_fns = []
        for kind, n_seeds in deep_kinds.items():
            probs = []
            for s in range(n_seeds):
                fn = fit_deep(kind, SEED + s, xs_f, xt_f, y_tr[fit_idx], xs_v, xt_v, y_tr[val_idx],
                              KC, device)
                p = fn(xs_t, xt_t)
                probs.append(p)
                if kind == "Hybrid":
                    hybrid_seed_oof[s, te_idx] = p
                    hybrid_fns.append(fn)
            oof[kind][te_idx] = np.mean(probs, axis=0)

        # --- per-fold metrics
        for m in model_names:
            pred = oof[m][te_idx].argmax(1)
            fold_rows.append({"fold": fold, "model": m, **K.fair_metrics(y_te, pred)})

        # --- RO3: permutation importance (RF and Hybrid) and SHAP (RF)
        rf = fitted["RandomForest"]
        pi_rf = permutation_importance_raw(lambda d: full_proba(rf, prep.flat(d), KC),
                                           df_te, y_te, args.perm_repeats, rng)
        pi_rf["model"], pi_rf["fold"] = "RandomForest", fold

        def hybrid_from_df(d):
            a, b = dprep.transform(d)
            return np.mean([fn(a, b) for fn in hybrid_fns], axis=0)
        pi_hy = permutation_importance_raw(hybrid_from_df, df_te, y_te, args.perm_repeats, rng)
        pi_hy["model"], pi_hy["fold"] = "Hybrid", fold
        perm_rows += [pi_rf, pi_hy]

        take = rng.choice(len(te_idx), size=min(args.shap_rows_per_fold, len(te_idx)), replace=False)
        sh = shap_raw(rf, X_te[take], prep.feature_names)
        sh["fold"] = fold
        shap_rows.append(sh)

        fm = pd.DataFrame(fold_rows)
        cur = fm[fm.fold == fold].set_index("model")["macro_f1"].round(3).to_dict()
        print(f"  fold {fold}/{args.folds} done ({time.time() - t0:.0f}s) macro-F1: {cur}", flush=True)

    # ------------------------------------------------------------------ RO1
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(f"{out}/ro1_per_fold_metrics.csv", index=False)
    metrics = ["macro_f1", "balanced_accuracy", "macro_recall", "accuracy", "macro_precision"]
    summ = []
    for m in model_names:
        f = folds[folds.model == m]
        row = {"model": m}
        for k in metrics:
            lo, hi = boot_ci(f[k])
            row[f"{k}_mean"], row[f"{k}_std"] = f[k].mean(), f[k].std(ddof=1)
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ).sort_values("macro_f1_mean", ascending=False)
    summ.to_csv(f"{out}/ro1_summary.csv", index=False)

    tests = []
    hy = folds[folds.model == "Hybrid"].sort_values("fold")
    for m in model_names:
        if m == "Hybrid":
            continue
        o = folds[folds.model == m].sort_values("fold")
        for k in ["macro_f1", "balanced_accuracy", "macro_recall"]:
            stat, p = paired_wilcoxon(hy[k].to_numpy(), o[k].to_numpy())
            tests.append({"comparison": f"Hybrid vs {m}", "metric": k,
                          "mean_diff": float((hy[k].to_numpy() - o[k].to_numpy()).mean()),
                          "wilcoxon_stat": stat, "p_value": p,
                          "significant_0.05": bool(p < 0.05)})
    tests = pd.DataFrame(tests)
    tests.to_csv(f"{out}/ro1_paired_tests.csv", index=False)

    oof_df = pd.DataFrame({"true_label": y, "fold": fold_of})
    for m in model_names:
        oof_df[f"pred_{m}"] = oof[m].argmax(1)
    for a in PROTECTED:
        oof_df[a] = df[a].astype(str).to_numpy()
    oof_df.to_csv(f"{out}/oof_predictions.csv", index=False)
    for m in model_names:
        np.save(f"{out}/oof_proba_{m}.npy", oof[m])

    # ------------------------------------------------------------------ seed stability
    seed_rows = []
    for s in range(args.seeds):
        seed_rows.append({"seed": SEED + s, **K.fair_metrics(y, hybrid_seed_oof[s].argmax(1))})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(f"{out}/seed_per_seed_metrics.csv", index=False)
    seed_sum = K.seed_stability_summary(seed_df, ["macro_f1", "balanced_accuracy", "macro_recall"])
    seed_sum.to_csv(f"{out}/seed_stability_summary.csv", index=False)
    curve = K.cumulative_ensemble_curve(hybrid_seed_oof, y, groups=df["STRATUM"].astype(str))
    curve.to_csv(f"{out}/seed_ensemble_curve.csv", index=False)

    # ------------------------------------------------------------------ RO2
    fair_models = ["Hybrid", "HGB", "RandomForest", "LogReg"]
    group_tables, gap_rows = [], []
    for a in PROTECTED:
        for m in fair_models:
            per, gap = K.group_fairness_gaps(y, oof_df[f"pred_{m}"], df[a])
            per["attribute"], per["model"] = a, m
            group_tables.append(per)
            row = {"attribute": a, "model": m, **gap}
            if m in ("Hybrid", "HGB"):
                merged, _ = K.merge_small_groups(df[a])
                row["perm_p_value"] = K.permutation_gap_pvalue(
                    y, oof_df[f"pred_{m}"], merged, gap["macro_f1_gap"], n_perm=args.n_perm)
            gap_rows.append(row)
        merged, _ = K.merge_small_groups(df[a])
        d = K.bootstrap_gap_delta(y, oof_df["pred_Hybrid"], oof_df["pred_HGB"], merged,
                                  n_bootstrap=args.n_boot)
        gap_rows.append({"attribute": a, "model": "Delta(Hybrid-HGB)", "macro_f1_gap": d["delta"],
                         "delta_ci_low": d["ci_low"], "delta_ci_high": d["ci_high"]})
        print(f"  RO2 {a} done ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(group_tables).to_csv(f"{out}/ro2_group_metrics.csv", index=False)
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(f"{out}/ro2_gaps.csv", index=False)

    # ------------------------------------------------------------------ RO3
    perm = pd.concat(perm_rows)
    perm_g = (perm.groupby(["model", "feature"])[["importance_mean"]].mean()
              .join(perm.groupby(["model", "feature"])["importance_mean"].std().rename("fold_std"))
              .reset_index().sort_values(["model", "importance_mean"], ascending=[True, False]))
    perm_g.to_csv(f"{out}/ro3_permutation_importance.csv", index=False)
    shap_g = (pd.concat(shap_rows).groupby("feature")["mean_abs_shap"].mean().reset_index()
              .sort_values("mean_abs_shap", ascending=False))
    shap_g.to_csv(f"{out}/ro3_shap_randomforest.csv", index=False)
    rank_df, rho = K.importance_rank_agreement(perm_g[perm_g.model == "RandomForest"],
                                               "importance_mean", shap_g, "mean_abs_shap")
    rank_df.to_csv(f"{out}/ro3_rank_agreement_rf.csv", index=False)
    rank_hy, rho_hy = K.importance_rank_agreement(perm_g[perm_g.model == "Hybrid"],
                                                  "importance_mean", shap_g, "mean_abs_shap")
    rank_hy.to_csv(f"{out}/ro3_rank_agreement_hybrid_perm_vs_rf_shap.csv", index=False)

    summary = {
        "target": args.target, "n_students": int(len(df)), "n_classes": KC, "classes": names,
        "folds": args.folds, "hybrid_seeds": args.seeds, "ablation_seeds": args.ablation_seeds,
        "features": RAW_FEATURES,
        "ro1": summ.round(4).to_dict(orient="records"),
        "ro1_tests": tests.round(4).to_dict(orient="records"),
        "seed_stability": seed_sum.round(4).to_dict(orient="records"),
        "ro3_spearman_rf_perm_vs_rf_shap": rho,
        "ro3_spearman_hybrid_perm_vs_rf_shap": rho_hy,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(f"{out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # ------------------------------------------------------------------ console report
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nRO1 summary (mean [95% bootstrap CI] over folds):")
    for _, r in summ.iterrows():
        print(f"  {r.model:24s} macroF1 {r.macro_f1_mean:.3f} [{r.macro_f1_ci_low:.3f},{r.macro_f1_ci_high:.3f}]"
              f"  balAcc {r.balanced_accuracy_mean:.3f}  macroRecall {r.macro_recall_mean:.3f}"
              f"  acc {r.accuracy_mean:.3f}")
    print("\nPaired Wilcoxon (Hybrid vs others):")
    print(tests.round(4).to_string(index=False))
    print("\nSeed stability (Hybrid):")
    print(seed_sum.round(4).to_string(index=False))
    print(curve.round(4).to_string(index=False))
    print("\nRO2 gaps:")
    print(gaps.round(4).to_string(index=False))
    print(f"\nRO3 Spearman rho RF permutation vs RF SHAP: {rho:.3f}; "
          f"Hybrid permutation vs RF SHAP: {rho_hy:.3f}")
    print(perm_g.groupby("model").head(8).round(4).to_string(index=False))
    print(shap_g.head(10).round(4).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
