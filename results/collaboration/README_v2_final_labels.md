# Final SSI label set + fold map v2 (2026-09-28)

The 26-patient list confirmed by Dr. Kewalramani (`final_ssi_list.txt`) is now the
only SSI definition used here. Labels are no longer taken from chart review, the
survey, or any other source.

## Effect of the label change on the analytic frame

- Two flips inside the smartphone frame: **RU-A1106 negative -> SSI**, **RU-A1107
  SSI -> negative**. They cancel, so the frame still holds 20 SSI patients.
- Six of the 26 are not in the smartphone frame: RU-A1104, RU-A1108, RU-A1163,
  RU-A1195, RU-A1303, RU-A1365.
- Three had no images at all in the locked package and were absent from v1:
  RU-A1108, RU-A1195, RU-A1365. They are added in v2.

**Important caveat.** The image models were *trained* with the old labels, so the
re-runs below re-score fixed predictions under the final labels. This is a valid
evaluation-only comparison, but the networks have not been retrained on the
corrected labels; RU-A1106 was trained as a negative and RU-A1107 as a positive.
Retraining is required before these numbers are final (cost below). Logistic
clinical/combined models were fully refit under each label set.

## Re-runs, old vs final labels (`relabel_comparison.csv`)

| Analysis | Patients | Events | AUROC old (95% CI) | AUROC final (95% CI) | AUPRC old -> final |
|---|---|---|---|---|---|
| POD-7 primary (mean) | 193 | 15 | 0.761 (0.645–0.860) | **0.744 (0.629–0.846)** | 0.172 -> 0.162 |
| POD-7 latest photo | 193 | 15 | 0.746 (0.635–0.844) | 0.742 (0.632–0.841) | 0.162 -> 0.158 |
| POD-7 closest photo | 193 | 15 | 0.744 (0.646–0.833) | 0.752 (0.657–0.842) | 0.157 -> 0.161 |
| RGB pre-closure | 139 | 14 | 0.706 (0.593–0.810) | 0.705 (0.593–0.809) | 0.173 -> 0.173 |
| RGB post-closure | 153 | 16 | 0.786 (0.685–0.873) | **0.763 (0.659–0.857)** | 0.245 -> 0.231 |
| RGB pre+post mean | 131 | 12 | 0.775 (0.661–0.872) | 0.757 (0.636–0.865) | 0.197 -> 0.190 |
| All-images comparator | 210 | 20 | 0.779 (0.680–0.863) | 0.765 (0.666–0.847) | 0.226 -> 0.217 |
| Clinical (intraop, refit) | 210 | 20 | 0.747 (0.605–0.874) | **0.664 (0.494–0.813)** | 0.387 -> 0.340 |
| Image only | 210 | 20 | 0.779 (0.685–0.861) | 0.765 (0.666–0.850) | 0.226 -> 0.217 |
| Combined (refit) | 210 | 20 | 0.794 (0.681–0.898) | 0.800 (0.691–0.890) | 0.357 -> 0.316 |

Architectures, 3 seeds each (`architectures_relabelled.csv`), patient AUROC mean±SD:
EfficientNet-B0 0.755±0.037 -> 0.751±0.037; ResNet18 0.731±0.052 -> 0.723±0.034;
ViT-B/16 0.725±0.049 -> 0.707±0.047.

Masking, 206 patients / 19 events (`masking_relabelled.csv`): full 0.725 -> 0.714;
wound-only 0.670 -> 0.616; background-only 0.821 -> 0.834; sham wound 0.817 ->
0.797; sham background 0.654 -> 0.647.

**What changes:** everything shifts slightly, mostly downward, by 0.01–0.02 AUROC,
which is far inside the confidence intervals. The exception is the clinical model
(-0.083), which is the most label-sensitive because it is refit on 210 patients
with few events. **No conclusion changes**: the primary remains a moderate signal,
architectures remain indistinguishable, the sham controls still match the true
masks, and wound-only still fails to beat full-image.

The healing-stage positive control is unaffected because its labels are
postoperative stage, not SSI.

## Fold map v2 (`SHARED_fold_map_v2_interim.csv`)

224 patients, 26 SSI. Every patient carried over from v1 keeps the same fold
(asserted in code), `y_true` comes from the final list, and the three missing
SSI patients were added to the fold with the fewest SSI events at the time of
assignment (deterministic, lowest fold index breaks ties).

| Fold | Patients | SSI |
|---|---|---|
| 0 | 45 | 7 |
| 1 | 46 | 5 |
| 2 | 45 | 4 |
| 3 | 44 | 5 |
| 4 | 44 | 5 |

This is marked **interim** because the 67-patient thermal list
(`thermal_patients_not_in_SHARED_fold_map_v1.csv`) has not been received. Those
patients will be appended with the same seeded stratified procedure without
moving any existing fold, and the file promoted to `SHARED_fold_map_v2.csv`.

### On rebalancing

Under the final labels, v1's own 221 patients give 7/2/4/5/5. Adding the three
missing SSI patients already lifts fold 1 from 2 to 5, giving 7/5/4/5/5, so the
worst of the imbalance resolves without touching anyone's fold.

Recommendation: **do not rebalance further.** All reported metrics are computed on
pooled out-of-fold predictions with patient-clustered bootstrap, not averaged per
fold, so an uneven event split costs a little precision but does not bias the
estimates. Rebalancing would move existing patients between folds and invalidate
every trained model. Measured rework: ResNet18 3 seeds ~1.4 h, EfficientNet-B0 3
seeds ~3.2 h, ViT-B/16 3 seeds ~12 h, five masking conditions ~6 h, learning
curve ~2.7 h, totalling roughly 25 GPU-hours plus CPU analyses and repackaging,
about 2-3 days wall-clock on the current hardware. If rebalancing is wanted
anyway, it should happen once, now, before thermal training begins.

## Day-recovery clarifications

**Photos without a day outside the 598.** Nine images from five patients
(RU-A1264, RU-A1304, RU-A1307, RU-A1310, RU-A1311; none SSI) have no parseable
day and no phase label; they carry generic camera filenames (`IMG_4666.HEIC`).
None appear in the EXIF archive and none of those patients has a derivable
surgery date, so **0 of 9 are recoverable** from available sources. They are
listed in `undated_non_intraoperative_9.csv`. They were already excluded from
POD-dependent analyses, so nothing changes.

**233 versus 66.** The two counts describe different sets:

- **233** is the *validation* set: every intraoperative-named image anywhere in
  the EXIF archive that had both a capture time and a derived surgery date. It
  includes 92 thermal images, which are not RGB analysis images, and 141 RGB
  images, of which only 66 matched into the 1,948-image analytic frame (the
  other 75 were quality-gated out or are separate copies). It was used only to
  test whether intraoperative images fall on the day of surgery.
- **66** is the number of *analysis-frame* images whose day was actually assigned
  from their own capture time, after precedence. A filename day number outranks
  EXIF, so the 49 frame images carrying an explicit day were resolved by that
  rule even where EXIF also existed.
