# Final SSI label set + fold map v2 (updated 2026-10-01)

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

**Retraining complete (2026-10-01).** The networks have now been retrained on the
final labels — 3 architectures x 3 seeds plus the five masking conditions, all on
the identical `MASTER_map_1948_seed42` folds and the same sham random stream
(seed 20260807). Retrained results are in the **Retrained results** section below
and supersede the evaluation-only numbers; both are reported side by side.

Why both are shown: the earlier re-runs re-scored predictions from networks
*trained* with the old labels (RU-A1106 as a negative, RU-A1107 as a positive),
which is a valid evaluation-only comparison but penalises a model for two labels
it was taught wrong. Logistic clinical/combined models were fully refit under each
label set throughout, so the clinical, combined and POD-7 rows are unaffected by
retraining and stand as reported below.

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

Architectures and masking: the evaluation-only values are superseded by the
retrained results below.

## Retrained results (final labels)

Per-patient out-of-fold predictions for all 14 runs are in
`results/master_models_final/` (images carry the hashed `image_id`, never a path).

### Architectures, 3 seeds each (`architectures_retrained_final.csv`)

Patient AUROC, mean ± SD across seeds:

| Architecture | Retrained (final) | Published (old) | Evaluation-only | Retrained − published | Per-seed (retrained) |
|---|---|---|---|---|---|
| ResNet18 | **0.7556 ± 0.0380** | 0.731 ± 0.052 | 0.723 ± 0.034 | +0.025 | 0.789 / 0.714 / 0.764 |
| EfficientNet-B0 | **0.7327 ± 0.0490** | 0.755 ± 0.037 | 0.751 ± 0.037 | −0.022 | 0.746 / 0.678 / 0.774 |
| ViT-B/16 | **0.6839 ± 0.0453** | 0.725 ± 0.049 | 0.707 ± 0.047 | −0.041 | 0.706 / 0.714 / 0.632 |

Pooled across all nine runs: 0.7241 ± 0.0498.

Every architecture's retrained mean lies within one SD of its published value and
the shifts go in both directions, which is the signature of seed variance rather
than a label effect. Two qualifications worth stating rather than smoothing over:
the evaluation-only comparison *understated* ResNet18, which gains +0.033 when
retrained, so the direction of the label correction is not uniform; and ViT falls
furthest (−0.041), widening ResNet18 − ViT from +0.006 published to +0.072
retrained. With three seeds the spreads still overlap heavily, so
"indistinguishable" survives, but the retrained ranking is less flat than the
published one and should not be presented as identical in character.

### Masking, 206 patients / 19 events (`masking_retrained_final.csv`)

| Condition | Retrained (final) | Published (old) | Evaluation-only | Retrained − published |
|---|---|---|---|---|
| Full image | **0.6361** | 0.725 | 0.714 | −0.089 |
| Wound-only | **0.6189** | 0.670 | 0.616 | −0.051 |
| Background-only | **0.7489** | 0.821 | 0.834 | −0.072 |
| Sham wound-only | **0.7762** | 0.817 | 0.797 | −0.041 |
| Sham background-only | **0.6065** | 0.654 | 0.647 | −0.048 |

All five conditions shift down by 0.04–0.09, but every structural relationship the
masking experiment was built to test is preserved:

- **Background-only still beats wound-only by a wide margin** (+0.130 retrained,
  +0.151 published). The infection-related signal is not concentrated in the
  annotated wound.
- **The sham control still matches the true mask.** Sham wound-only 0.776 against
  true background-only 0.749, a gap of +0.027 (published: 0.817 vs 0.821, −0.004).
  The gap widens slightly, but the conclusion — that a randomly relocated mask of
  equal area performs like the true one — is unchanged.
- **Wound-only still fails to beat full-image** (0.619 vs 0.636).

The uniform downward shift across all five conditions, including both sham
controls, is consistent with refitting from scratch on the smaller masking subset
(1,886 images, 206 patients, 19 events) rather than with anything specific to the
corrected labels.

**What changes:** the POD-7 and RGB-timepoint rows shift by 0.01–0.02 AUROC, far
inside their confidence intervals. The clinical model moves most (−0.083); its
decomposition is below. The retrained image models shift by 0.02–0.09, in both
directions. **No conclusion changes**: the primary remains a moderate signal, the
architectures remain statistically indistinguishable, the sham controls still match
the true masks, and wound-only still fails to beat full-image.

The healing-stage positive control is unaffected because its labels are
postoperative stage, not SSI.

## Fold map v2 (`SHARED_fold_map_v2.csv`) — final

289 patients, 26 SSI, folds 58/59/58/57/57 and 7/5/4/5/5. Built in layers, none
of which moves a patient already placed (asserted in code): 210 from the
published smartphone master map (seed 42), 11 thermal-only from v1, 3 SSI
patients with no images in the locked package, and 65 from the 2026-09-29
thermal list by stratified assignment (seed 42). Two of that list's 67 IDs
(RU-A1108, RU-A1195) were already in the map, so 65 were new; none of the 65 is
SSI-positive. v1 and the interim file are superseded and kept for provenance.

### On rebalancing

Adding the three missing SSI patients already lifted fold 1 from 2 events to 5,
giving 7/5/4/5/5, so the worst imbalance resolved without touching anyone's fold.
**Do not rebalance further.** All metrics are computed on pooled out-of-fold
predictions with patient-clustered bootstrap, not averaged per fold, so an uneven
event split costs a little precision but does not bias the estimates. Rebalancing
would move existing patients between folds and invalidate every trained model.
Measured rework if it were done anyway: ResNet18 3 seeds ~1.4 h, EfficientNet-B0
3 seeds ~3.2 h, ViT-B/16 3 seeds ~12 h, five masking conditions ~6 h, learning
curve ~2.7 h — roughly 25 GPU-hours plus CPU analyses and repackaging.

## Paired comparisons under the final labels

`paired_comparisons_final_labels.csv` — 210 patients, 20 events, 20,000-resample
patient-clustered bootstrap, differences taken within each replicate.

| Contrast | ΔAUROC (95% CI) | P | ΔAUPRC | P |
|---|---|---|---|---|
| Combined vs clinical | +0.135 (+0.037 to +0.253) | 0.005 | −0.024 | 0.83 |
| Image vs clinical | +0.101 (−0.066 to +0.281) | 0.25 | −0.122 | 0.24 |
| Combined vs image | +0.034 (−0.075 to +0.132) | 0.49 | +0.098 | 0.15 |

Under the old labels the same contrasts were +0.048 (P=0.13) and +0.033 (P=0.69).

**Do not report the P=0.005 as a finding.** It is partition-fragile, in exactly
the way that exposed the earlier P=0.013 artifact. Repeating the whole nested-CV
procedure under five cross-validation partitions
(`paired_partition_stability.csv`):

| CV seed | Clinical | Combined | Δ combined−clinical | P |
|---|---|---|---|---|
| 20260807 | 0.664 | 0.799 | +0.135 | 0.004 |
| 42 | 0.648 | 0.770 | +0.122 | 0.018 |
| 7 | 0.723 | 0.747 | +0.025 | 0.42 |
| 2024 | 0.763 | 0.800 | +0.037 | 0.22 |
| 1 | 0.735 | 0.783 | +0.048 | 0.12 |

The contrast is significant in 2 of 5 partitions and ranges +0.025 to +0.135. The
AUPRC difference points the other way in every specification. The defensible
statement is unchanged from the old labels: **adding the image score to the
clinical model does not produce a significant improvement at this event count.**

## What drives the −0.083 clinical drop

`clinical_drop_decomposition.csv`. Splitting the drop into an evaluation
component (score the model refit on old labels against the final labels) and a
refit component (refit on the final labels):

| Step | AUROC | Δ |
|---|---|---|
| Refit old labels, scored old labels | 0.7466 | — |
| Refit old labels, scored **final** labels | 0.7153 | −0.031 |
| Refit **final** labels, scored final labels | 0.6641 | −0.051 |

So roughly 38% of the drop is the two label flips and 62% is the refit.

The flips are fully explained and split almost evenly: flipping RU-A1106 alone
costs −0.016, flipping RU-A1107 alone −0.015, and the two together give −0.031,
the entire evaluation component. The reason is that both flips move a patient to
the wrong end of the clinical score distribution: RU-A1106 became SSI-positive
and sits at the 19th percentile of clinical risk, while RU-A1107 became negative
and sits at the 95th. With 20 events, two maximally adverse flips are worth about
0.03 AUROC.

**The refit component is not a stable quantity and should not be interpreted.**
Repeating the whole nested-CV procedure under five CV partitions for *both* label
sets (`clinical_partition_stability_matched.csv`):

| Label set | Per-partition clinical AUROC | Range | SD |
|---|---|---|---|
| Old | 0.7466 / 0.7509 / 0.7342 / 0.7718 / 0.7711 | 0.038 | 0.016 |
| Final | 0.6641 / 0.6482 / 0.7225 / 0.7626 / 0.7353 | **0.114** | 0.049 |

The final-label range is 3x the old-label range and wider than the 0.083 drop
itself. The corrected labels make the refit markedly less stable, which is what you
would expect when refitting eight covariates on 210 patients after moving two of
twenty events. Both label sets are evaluated on the same five partitions here; an
earlier draft compared a five-partition range against a three-partition one, which
understated the old-label spread.

**Answer to the question as posed:** it is mostly the refit, but the honest
reading is that only the ~0.031 flip component is a real, attributable change.
The remaining ~0.051 is inside partition noise and should be reported as such
rather than as a degradation of the clinical model.

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
