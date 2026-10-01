# -*- coding: utf-8 -*-
"""
Objective 1 - follow-up experiments
===================================
Each experiment changes ONE factor relative to the baseline protocol in
`objective1_experiments.py` (same folds, seeds, models and hyperparameters):

  E0  Baseline          : 21 majors, balanced class weights, all 12 temporal features.
  E1  Direct macro-track: models trained on the 4 MACRO_TRACK classes directly,
                          instead of summing the 21 major probabilities.
  E2  No class weights  : plain (unweighted) loss / class_weight=None.
  E3  Rare majors grouped: majors with < RARE_MIN students become
                          "OTHER - <track>"; a group still smaller than the number
                          of folds is merged into its track's largest major.
  E4  No Saber Pro      : drops G_SC, PERCENTILE, 2ND_DECILE, QUARTILE (exit-exam
                          scores measured after the major was chosen). The remaining
                          8 temporal features form 2 steps x 4 features.
  E5  Final granular    : E3 grouping + no class weights + no Saber Pro.
  E6  Final macro-track : 4-class training + no class weights + no Saber Pro.

Outputs per experiment: objective1_<tag>_results.csv, objective1_<tag>_folds.csv and
objective1_<tag>_oof.npz (out-of-fold probabilities),
plus objective1_ablation_overview.csv comparing the key metrics across experiments.

Usage: put dataset.csv and objective1_experiments.py in the working directory, then
`python objective1_ablations.py [E0 E1 ...]` (no arguments runs all of them).
"""

import sys

import numpy as np
import pandas as pd
import torch

import objective1_experiments as base

RARE_MIN = 30
SABER_PRO_FEATURES = ["G_SC", "PERCENTILE", "2ND_DECILE", "QUARTILE"]
GROUPED_COL = "PROGRAM_GROUPED"


def group_rare_majors(df: pd.DataFrame, min_count: int = RARE_MIN):
    """Return (df with GROUPED_COL, label -> track map, printable mapping table)."""
    counts = df[base.TARGET].value_counts()
    mapping = {}
    for prog, n in counts.items():
        track = base.PROGRAM_TO_TRACK[prog]
        mapping[prog] = prog if n >= min_count else f"OTHER - {track}"

    grouped = df[base.TARGET].map(mapping)
    group_sizes = grouped.value_counts()
    for prog, label in list(mapping.items()):
        if label.startswith("OTHER - ") and group_sizes[label] < base.N_SPLITS:
            track = base.PROGRAM_TO_TRACK[prog]
            largest = max((p for p in counts.index if base.PROGRAM_TO_TRACK[p] == track),
                          key=lambda p: counts[p])
            mapping[prog] = largest

    out = df.copy()
    out[GROUPED_COL] = out[base.TARGET].map(mapping)
    label_to_track = {mapping[p]: base.PROGRAM_TO_TRACK[p] for p in mapping}
    table = pd.DataFrame({"n": counts, "grouped_label": pd.Series(mapping)})
    return out, label_to_track, table[table.index != table["grouped_label"]]


def main(selected):
    print(f"[Setup] Device: {base.DEVICE} | torch {torch.__version__}")
    df = base.load_and_prepare(base.DATA_PATH)
    df_grouped, grouped_to_track, grouping_table = group_rare_majors(df)
    print(f"\n[E3] Majors regrouped (< {RARE_MIN} students):")
    print(grouping_table.to_string())
    print(f"[E3] {df_grouped[GROUPED_COL].nunique()} classes after grouping")

    no_pro = [f for f in base.TEMPORAL_FEATURES if f not in SABER_PRO_FEATURES]

    experiments = {
        "E0": dict(title="E0 BASELINE (21 majors, weighted, all features)"),
        "E1": dict(title="E1 DIRECT MACRO-TRACK (4 classes)", label_col=base.MACRO_TARGET),
        "E2": dict(title="E2 NO CLASS WEIGHTS", weighted=False),
        "E3": dict(title=f"E3 RARE MAJORS GROUPED (< {RARE_MIN} -> OTHER per track)",
                   label_col=GROUPED_COL, label_to_track=grouped_to_track),
        "E4": dict(title="E4 NO SABER PRO FEATURES (8 temporal features, 2 steps)",
                   temporal_features=no_pro, time_steps=2),
        # Combined, leakage-free final protocols
        "E5": dict(title="E5 FINAL GRANULAR (grouped majors, unweighted, no Saber Pro)",
                   label_col=GROUPED_COL, label_to_track=grouped_to_track, weighted=False,
                   temporal_features=no_pro, time_steps=2),
        "E6": dict(title="E6 FINAL MACRO-TRACK (4 classes, unweighted, no Saber Pro)",
                   label_col=base.MACRO_TARGET, weighted=False,
                   temporal_features=no_pro, time_steps=2),
    }

    overview = []
    for tag, cfg in experiments.items():
        if selected and tag not in selected:
            continue
        summary, _ = base.run_experiment(
            df_grouped, summary_csv=f"objective1_{tag}_results.csv",
            fold_csv=f"objective1_{tag}_folds.csv",
            oof_npz=f"objective1_{tag}_oof.npz", **cfg)
        for _, r in summary.iterrows():
            overview.append({
                "Experiment": cfg["title"], "Model": r["Model"],
                "Macro-Track Top-1 Acc (%)": r["Macro-Track Top-1 Acc (%) [mean]"],
                "Macro-Track Macro-F1": r["Macro-Track Macro-F1 [mean]"],
                "Granular Top-1 Acc (%)": r.get("Granular Top-1 Acc (%) [mean]", np.nan),
                "Granular Top-3 Acc (%)": r.get("Granular Top-3 Acc (%) [mean]", np.nan),
                "Granular Macro-F1": r.get("Granular Macro-F1 [mean]", np.nan),
            })

    ov = pd.DataFrame(overview).round(4)
    ov.to_csv("objective1_ablation_overview.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'#' * 120}\nABLATION OVERVIEW (fold means)\n{'#' * 120}")
    for exp, sub in ov.groupby("Experiment", sort=False):
        print(f"\n{exp}")
        print(sub.drop(columns="Experiment").to_string(index=False))
    print("\nSaved: objective1_ablation_overview.csv")


if __name__ == "__main__":
    main(set(a.upper() for a in sys.argv[1:]))
