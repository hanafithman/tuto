# Objective 1 — Results: predicting engineering study pathways

All numbers below come from `objective1_improved.py` (log: `improved_results.log`,
tables: `improved_results_summary.csv`, `improved_results_per_class.csv`) and
`objective1_hybrid_model.py` (single 80/10/10 split). Nothing here is estimated by hand.

## Data

- 12,411 students, 21 engineering programs mapped to 4 macro-tracks.
- Macro-track distribution: Industrial & Management 5,407 (43.6%), Civil & Infrastructure
  3,481 (28.0%), Mechanical, Electrical & Tech 2,522 (20.3%), Chemical & Process 1,001 (8.1%).
- Two programs have a single student (Textile; Industrial Control & Automation).

## Protocol

- 5-fold stratified cross-validation (stratified on program, so also on macro-track);
  results are mean ± std over the held-out folds.
- Inside each training fold: 15% inner validation split for early stopping and for tuning
  the decision rule. All encoders, scalers and target encoders are fitted on training rows only.
- Models: gradient boosting (HGB), the dual-branch hybrid network (static MLP + causal TCN +
  uni-LSTM) with early stopping, and their probability-averaged ensemble.
- Decision rules: plain argmax, and per-class log-probability offsets tuned on the inner
  validation split to maximise macro-F1.
- Class imbalance: inverse-frequency weights c_y = N / (K · N_y), capped at 10 (the cap only
  affects the 21-program target).

Feature settings:

| Setting | Features | Available before program choice? |
|---|---|---|
| A — original | 14 socioeconomic variables + 9 score variables (reshaped to 3×3) | Yes* |
| B — all pre-enrolment | A + remaining household assets, JOB, high-school name (target-encoded) | Yes* |
| C — + university | B + UNIVERSITY (target-encoded) | **No** — chosen together with the program; upper reference only |

\* `G_SC`, `PERCENTILE`, `2ND_DECILE` and `QUARTILE` appear to be Saber Pro (end-of-degree)
measures. They were kept because they are part of the specified feature set; if confirmed
to be Saber Pro, they are post-enrolment information and should be removed or flagged.

## Main result — 4 macro-tracks

Reference points: always predicting the majority track gives macro-F1 0.15; guessing at
random in proportion to class frequencies gives an expected macro-F1 of 0.25.

| Setting | Best model | Accuracy | Macro-F1 | Top-2 acc. | Top-3 acc. |
|---|---|---|---|---|---|
| A — original | Ensemble, tuned | 0.411 ± 0.025 | 0.367 ± 0.019 | 0.670 | 0.895 |
| B — all pre-enrolment | Ensemble, tuned | 0.415 ± 0.018 | **0.374 ± 0.021** | 0.689 | 0.899 |
| C — + university (reference) | Ensemble, tuned | 0.628 ± 0.014 | 0.612 ± 0.017 | 0.879 | 0.980 |

Per-class F1 (same models):

| Setting | Chemical & Process | Civil & Infrastructure | Industrial & Management | Mechanical, Electrical & Tech |
|---|---|---|---|---|
| A — original | 0.236 | 0.311 | 0.518 | 0.401 |
| B — all pre-enrolment | 0.249 | 0.352 | 0.521 | 0.373 |
| C — + university (reference) | 0.592 | 0.613 | 0.686 | 0.556 |

No class reaches 0.8 in any setting. The highest per-class F1 obtained anywhere is 0.69
(Industrial & Management, setting C).

Confusion matrix, setting B (summed over the 5 held-out folds; rows = actual):

| Actual \ Predicted | Chemical | Civil | Industrial | Mech/Elec | Recall |
|---|---|---|---|---|---|
| Chemical & Process | **286** | 190 | 274 | 251 | 0.29 |
| Civil & Infrastructure | 301 | **1,222** | 1,095 | 863 | 0.35 |
| Industrial & Management | 525 | 1,315 | **2,578** | 989 | 0.48 |
| Mechanical, Electrical & Tech | 204 | 712 | 538 | **1,068** | 0.42 |

Errors are spread across all tracks rather than concentrated between two similar ones,
which indicates the features carry little track-specific information rather than the
model confusing a single pair.

## Model comparison

| Setting | HGB macro-F1 | Hybrid macro-F1 | Ensemble macro-F1 |
|---|---|---|---|
| A — original | 0.349 ± 0.013 | 0.363 ± 0.012 | 0.367 ± 0.019 |
| B — all pre-enrolment | 0.364 ± 0.012 | 0.371 ± 0.008 | 0.374 ± 0.021 |
| C — + university | 0.606 ± 0.007 | 0.608 ± 0.014 | 0.612 ± 0.017 |

(Best decision rule per model.) Differences between models are within about one standard
deviation; the hybrid architecture performs on par with gradient boosting, not better.

## Secondary result — 21 programs

| Setting | Best macro-F1 | Best accuracy | Top-3 acc. (HGB) |
|---|---|---|---|
| A — original | 0.079 ± 0.006 | 0.319 ± 0.038 | 0.744 |
| B — all pre-enrolment | 0.097 ± 0.012 | 0.349 ± 0.028 | 0.770 |
| C — + university (reference) | 0.483 ± 0.014 | 0.571 ± 0.007 | 0.920 |

At the program level the hybrid network is clearly weaker than gradient boosting
(e.g. setting B accuracy 0.20 vs 0.35), largely because many programs have fewer than 100
students.

## Other checks

- Adding `SEL` and `SEL_IHE` (socioeconomic level of the student and of the institution) to
  the original features moved gradient-boosting macro-F1 from 0.349 to 0.371 — no change in
  the conclusion.
- Adding the Saber Pro sub-scores (`QR_PRO`, `CR_PRO`, …) gave macro-F1 ≈ 0.32 on a single
  split; they are post-enrolment and were not used further.
- `Cod_SPro` has the same prefix (`EK2018`) for every student and carries no information.

## Conclusion for Objective 1

Using only information available before program choice (Saber 11 scores and socioeconomic
background), the engineering macro-track can be predicted above chance but only modestly:
macro-F1 0.37 ± 0.02 under 5-fold cross-validation, with per-class F1 between 0.25 and 0.52.
Adding the university — which is decided together with the program — raises macro-F1 to
0.61, still with no class above 0.69. The ceiling is set by the information in the data,
not by the model: gradient boosting, the dual-branch hybrid network and their ensemble all
land within about one standard deviation of each other.

This suggests that the choice between engineering tracks depends mainly on factors not
recorded in this dataset (vocational interests, preferences, local program availability).
Collecting such variables is the most promising route to higher per-class performance.

## Reproducing

```bash
cd objective1
pip install -r requirements.txt
# place dataset.csv in this folder (it is not committed)
python objective1_hybrid_model.py   # single-split baseline, ~20 s on CPU
python objective1_improved.py       # 5-fold CV experiment, ~15 min on CPU
```

---

# Reframed Objective 1 — predicting Saber Pro performance (above vs below median)

Script: `objective1_performance.py` (log: `performance_results.log`, tables:
`performance_results_summary.csv`, `performance_results_folds.csv`).

**Target:** Saber Pro global score `G_SC` ≥ median, where the median is computed on each
training fold only (overall median 163). The two classes are therefore balanced (≈50/50).

**Leakage controls:** `G_SC`, `PERCENTILE`, `2ND_DECILE`, `QUARTILE` (all derived from the
target) and the Saber Pro sub-scores are excluded from the features. The temporal branch of the
hybrid model uses the 5 Saber 11 subject scores as a 5-step sequence. The decision threshold is
tuned on an inner validation split only.

**Settings:** A = socioeconomic + Saber 11; B = A + all other pre-enrolment variables and
high-school name; C = B + university and program (known at enrolment, still before the exam).

5-fold cross-validation, mean ± std:

| Setting | Model | Accuracy | Macro-F1 | ROC-AUC | F1 below median | F1 above median |
|---|---|---|---|---|---|---|
| A — Saber 11 + socioeconomic | HGB | 0.805 ± 0.009 | 0.805 | 0.888 | 0.803 ± 0.011 | 0.807 ± 0.008 |
| | Hybrid | 0.803 ± 0.009 | 0.803 | 0.891 | 0.798 ± 0.011 | 0.808 ± 0.012 |
| | Ensemble | 0.805 ± 0.013 | 0.805 | 0.891 | 0.805 ± 0.015 | 0.805 ± 0.011 |
| B — all pre-enrolment | HGB | 0.801 ± 0.010 | 0.801 | 0.887 | 0.796 ± 0.012 | 0.807 ± 0.010 |
| | Hybrid | 0.806 ± 0.010 | 0.806 | 0.891 | 0.803 ± 0.017 | 0.809 ± 0.007 |
| | Ensemble | 0.804 ± 0.011 | 0.803 | 0.891 | 0.798 ± 0.016 | 0.809 ± 0.011 |
| C — at enrolment | HGB | 0.809 ± 0.009 | 0.809 | 0.894 | 0.804 ± 0.013 | 0.814 ± 0.012 |
| | Hybrid | 0.810 ± 0.010 | 0.810 | 0.896 | 0.806 ± 0.011 | 0.815 ± 0.013 |
| | Ensemble | **0.810 ± 0.010** | **0.810** | **0.896** | **0.806 ± 0.014** | **0.814 ± 0.012** |

Confusion matrix, setting C, hybrid model (summed over the 5 held-out folds):

| Actual \ Predicted | Below median | Above median |
|---|---|---|
| Below median (6,094) | **4,881** | 1,213 |
| Above median (6,317) | 1,140 | **5,177** |

**How robust is "above 0.8"?** On average, both classes exceed 0.8 for 7 of the 9
model/setting combinations. The margin is small (0.80–0.815) compared with the fold-to-fold
standard deviation (≈0.01): looking at individual folds, both classes exceed 0.8 in only 1–3
of the 5 folds, and the lowest single-fold class F1 is 0.784. The honest statement is
"per-class F1 ≈ 0.80–0.81 (5-fold mean)", not "reliably above 0.8".

**Conclusion (reframed objective):** Saber 11 scores and socioeconomic background predict
whether a student will finish in the top or bottom half of the end-of-degree Saber Pro exam
with ≈80% accuracy and per-class F1 ≈ 0.80 (ROC-AUC ≈ 0.89). Adding university and program
raises this slightly, to 0.81 (AUC 0.90). The dual-branch hybrid network matches gradient
boosting but does not outperform it.
