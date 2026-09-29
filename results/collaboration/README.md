# Cross-modality collaboration package (updated 2026-09-29)

Answers three PI requests: intraoperative RGB timepoints, day recovery for
unnumbered photos, and a single shared fold map for modality comparison/fusion.

## 1. SHARED_fold_map_v2.csv  — USE THIS FOR ALL MODALITIES

`pid, fold, y_true, n_smartphone, n_thermal, source`

**289 patients, 26 SSI.** `y_true` is the final 26-patient SSI list confirmed by
Dr. Kewalramani (`final_ssi_list.txt`); no other label source is used. Folds:

| Fold | Patients | SSI |
|---|---|---|
| 0 | 58 | 7 |
| 1 | 59 | 5 |
| 2 | 58 | 4 |
| 3 | 57 | 5 |
| 4 | 57 | 5 |

Assembled in layers, each added without moving anyone already placed (asserted in
code): 210 patients inherit their fold from the published smartphone master map
(`MASTER_map_1948_seed42.csv`, seed 42); 11 thermal-only patients from v1; 3 SSI
patients with no images in the locked package; and 65 further thermal patients
from the 2026-09-29 thermal list, assigned by stratified split (seed 42). Two of
that list's 67 IDs (RU-A1108, RU-A1195) were already present, hence 65 new.
Every patient in v1 and in the v2 interim file keeps the same fold, so all
published smartphone results stand unchanged.

`SHARED_fold_map_v1.csv` (221 patients) and `SHARED_fold_map_v2_interim.csv`
(224 patients) are retained for provenance only and are **superseded**.

**SSI events are not evenly distributed across folds (7/5/4/5/5), and neither
are patients (58/59/58/57/57). This is expected and does not need correcting:
every metric in this project is computed on pooled out-of-fold predictions
across all five folds with a patient-clustered bootstrap, never as an average of
per-fold metrics, so an uneven split costs a little precision but does not bias
any estimate. Rebalancing would move patients between folds and invalidate every
already-trained model.**

**Why one shared map matters.** The locked per-arm fold maps are NOT
interchangeable: of the 210 patients present in both the smartphone and the
both-modality arms, only 33 (16%) landed in the same fold. Scoring thermal on its
own arm map and RGB on the smartphone map would compare models trained on
different splits, and any later fusion would leak across folds. Both modalities
must report per-patient scores on this one file.

## 2. RGB at the intraoperative timepoints

`rgb_intraoperative_scores.csv` — `pid, rgb_pre_closure_score,
rgb_post_closure_score, n_pre_closure, n_post_closure, y_true, fold`

The 1,948-image analytic set **does** include intraoperative phone photographs:
598 images labelled pre/post, of which 578 are the true intraoperative
wound-closure timepoints (filenames `(Pre-Standard n)` / `(Post-Standard n)`,
the RGB counterparts of the `(Pre-Thermal n)` / `(Post-Thermal n)` files).
Patient-level out-of-fold RGB scores at those timepoints (`rgb_intraoperative_summary.csv`):

| Timepoint | Patients | Events | Patient AUROC (95% CI) |
|---|---|---|---|
| Pre-closure | 139 | 14 | 0.706 (0.594–0.810) |
| Post-closure | 153 | 16 | 0.786 (0.684–0.874) |
| Pre + post mean | 131 | 12 | 0.775 (0.660–0.873) |

These are directly comparable with thermal scores at the same timepoints once
thermal is scored on the shared fold map.

## 3. Day recovery for photographs without a day number

`day_recovery_intraoperative.csv` — `pid, Image_Filename, phase_norm,
pod_as_analysed, recovered_pod, evidence, y_true`

All 598 resolved, using a precedence of evidence:

| Evidence | Images |
|---|---|
| Explicit day number in filename | 49 |
| Own EXIF capture date vs. patient surgery date | 66 |
| Paired thermal (FLIR) capture date at the same timepoint | 21 |
| Intraoperative naming convention (validated below) | 462 |

Surgery dates were derived per patient from EXIF capture dates of day-labelled
photographs (median of capture date − labelled day); 36/40 patients were
internally consistent within 1 day, and these agreed exactly with chart-review
surgery dates (diagnosis date − POD) in 9 of 11 checkable patients.

**Validation of the day-0 assignment:** among 233 intraoperative-named images
with both a capture time and a derived surgery date, 100% fall within ±1 day of
surgery and 85% exactly on the day of surgery. The day-0 assignment is therefore
evidence-based, not a convention.

**Corrections found:** 20 images across 8 patients were *not* day 0 (recovered
days 1, 3, 4, 5, 7, 8, 9; two `RU-A1276 D7 Post-Op` files and several `Day_N`
exports had been forced to day 0). Re-running the primary with corrected days
leaves it unchanged: AUROC 0.7607 either way (`pod7_day_corrected_sensitivity.csv`).

Two data-quality items for the chart review: `RU-A1104` has diagnosis year 2035
(typo for 2025), and `RU-A1180`'s chart-derived surgery date differs from the
EXIF-derived date by 2 days.

## Privacy note

EXIF capture datetimes and derived surgery dates are calendar dates tied to
patients (PHI) and are **not** in this repository. They remain offline in the
working directory. Only relative postoperative days appear here.
