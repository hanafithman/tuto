# Copied from the edm-pathway-fairness-analysis skill (kernel.py) so the analysis is
# reproducible from this repository alone. Unmodified below this header.

# kernel.py — reusable analysis helpers for fairness-aware, explainable
# multiclass pathway prediction (RO1 predictive validity / RO2 fairness /
# RO3 interpretability). Model-agnostic: these operate on out-of-fold (OOF)
# prediction tables and per-seed probability arrays, so they port across
# data cuts, seed sets, and model families. numpy/pandas are in the starter
# set; sklearn is imported inside function bodies (not always preinstalled).

import numpy as np
import pandas as pd

MIN_GROUP_SIZE = 50


def merge_small_groups(series, min_group_size=None):
    """Collapse group levels with < min_group_size members into
    'OTHER_SMALL_GROUP'. Matches the RO2 fairness protocol exactly:
    cast to str, fill NA with 'Unknown', then merge. Returns
    (merged_series, list_of_small_levels)."""
    if min_group_size is None:
        min_group_size = MIN_GROUP_SIZE
    series = pd.Series(series).astype(str).fillna("Unknown")
    counts = series.value_counts()
    small = counts[counts < min_group_size].index.tolist()
    merged = series.where(~series.isin(small), "OTHER_SMALL_GROUP")
    return merged, small


def fair_metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def group_fairness_gaps(y_true, y_pred, groups, min_group_size=None):
    """Per-group metrics + max-min gaps for one model on one attribute.
    Small levels are merged first (RO2 semantics), then EVERY resulting
    level contributes to the gap. Returns (per_group_df, gap_dict)."""
    if min_group_size is None:
        min_group_size = MIN_GROUP_SIZE
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    merged, _ = merge_small_groups(groups, min_group_size)
    merged = merged.reset_index(drop=True)
    rows = []
    for g in merged.value_counts().index.tolist():
        m = (merged == g).to_numpy()
        rows.append({"group": g, "n_students": int(m.sum()),
                     **{k: float(v) for k, v in fair_metrics(y_true[m], y_pred[m]).items()}})
    mdf = pd.DataFrame(rows)
    gap = {
        "group_count": int(mdf.shape[0]),
        "macro_f1_gap": float(mdf["macro_f1"].max() - mdf["macro_f1"].min()),
        "balanced_accuracy_gap": float(mdf["balanced_accuracy"].max() - mdf["balanced_accuracy"].min()),
        "macro_recall_gap": float(mdf["macro_recall"].max() - mdf["macro_recall"].min()),
        "lowest_macro_f1_group": str(mdf.sort_values("macro_f1").iloc[0]["group"]),
        "highest_macro_f1_group": str(mdf.sort_values("macro_f1", ascending=False).iloc[0]["group"]),
    }
    return mdf, gap


def permutation_gap_pvalue(y_true, y_pred, groups, observed_gap,
                           n_perm=1000, seed=42):
    """Permutation test: how often does a random re-labelling of group
    membership produce a macro-F1 gap >= the observed gap? Small p means
    the disparity is unlikely under the null of no group effect. Uses the
    (extreme+1)/(n+1) estimator."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(groups).astype(str).to_numpy()
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(groups)
        vals = [fair_metrics(y_true[perm == g], y_pred[perm == g]) for g in np.unique(perm)]
        mdf = pd.DataFrame(vals)
        if float(mdf["macro_f1"].max() - mdf["macro_f1"].min()) >= observed_gap:
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def bootstrap_gap_delta(y_true, pred_a, pred_b, groups,
                        n_bootstrap=1000, seed=42):
    """Bootstrap CI for the DIFFERENCE in macro-F1 gap between two models
    (model_a gap minus model_b gap) on the same attribute. A CI excluding
    zero means one model is reliably fairer on this attribute. Returns
    {delta, ci_low, ci_high}."""
    rng = np.random.default_rng(seed)
    g_all = pd.Series(groups).astype(str).to_numpy()
    y = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    n = len(y); deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        g, t, aa, bb = g_all[idx], y[idx], a[idx], b[idx]
        ma = pd.DataFrame([fair_metrics(t[g == gn], aa[g == gn]) for gn in np.unique(g)])
        mb = pd.DataFrame([fair_metrics(t[g == gn], bb[g == gn]) for gn in np.unique(g)])
        deltas.append(float((ma["macro_f1"].max() - ma["macro_f1"].min()) -
                            (mb["macro_f1"].max() - mb["macro_f1"].min())))
    arr = np.asarray(deltas)
    return {"delta": float(arr.mean()),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975))}


def seed_stability_summary(per_seed_df, cols):
    """Given a per-seed metrics table (one row per seed), report
    mean/std/min/max per column plus a coefficient-of-variation flag.
    A metric with cv > 0.5 is flagged seed-unstable — report its
    distribution, not a point estimate."""
    out = []
    for c in cols:
        v = per_seed_df[c].to_numpy(dtype=float)
        mean = float(v.mean()); std = float(v.std(ddof=1))
        out.append({"metric": c, "mean": mean, "std": std,
                    "min": float(v.min()), "max": float(v.max()),
                    "cv": float(std / mean) if mean else np.nan,
                    "seed_unstable": bool(mean and std / mean > 0.5)})
    return pd.DataFrame(out)


def cumulative_ensemble_curve(seed_probs, y_true, groups=None, min_group_size=None):
    """Stabilization curve for a probability ensemble. seed_probs is
    (n_seeds, n_samples, n_classes). Returns a DataFrame with, for k=1..n_seeds,
    the balanced accuracy / macro-F1 of the mean-probability prediction over
    the first k seeds (and the macro-F1 group gap if groups given). A curve
    that keeps swinging with k signals a seed-unstable estimate."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    if min_group_size is None:
        min_group_size = MIN_GROUP_SIZE
    seed_probs = np.asarray(seed_probs); y_true = np.asarray(y_true)
    n_seeds = seed_probs.shape[0]
    running = np.zeros(seed_probs.shape[1:]); rows = []
    for k in range(1, n_seeds + 1):
        running += seed_probs[k - 1]
        pred = (running / k).argmax(1)
        row = {"k_seeds": k,
               "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
               "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0))}
        if groups is not None:
            _, gap = group_fairness_gaps(y_true, pred, groups, min_group_size)
            row["macro_f1_gap"] = gap["macro_f1_gap"]
        rows.append(row)
    return pd.DataFrame(rows)


def importance_rank_agreement(df_a, val_a, df_b, val_b, key="feature"):
    """Compare two global-importance rankings (e.g. permutation importance
    vs SHAP). Returns (merged_rank_df, spearman_rho). Large rank gaps flag
    features that one method credits and the other misses — often correlated
    predictors that permutation importance under-credits but SHAP does not."""
    a = df_a[[key, val_a]].copy(); b = df_b[[key, val_b]].copy()
    m = a.merge(b, on=key)
    m["rank_a"] = m[val_a].rank(ascending=False)
    m["rank_b"] = m[val_b].rank(ascending=False)
    m["rank_gap"] = (m["rank_a"] - m["rank_b"]).abs()
    rho = float(m[["rank_a", "rank_b"]].corr(method="spearman").iloc[0, 1])
    return m.sort_values("rank_gap", ascending=False), rho
