# Paper 1 — Round 2 revisions (2026-08-18)

Every point from the round-2 review and the six-item reconciliation is addressed; all numbers below are fold-paired on the frozen master map (`MASTER_map_1948_seed42.csv`: 1,948 images / 210 patients / 20 SSI+, locked seed-42 patient partition).

## 1. Architectures, fold-paired with seed variability

Identical folds and 12-epoch protocol. Patient AUROC: efficientnet_b0 (3 seeds) mean 0.755 sd 0.037 range 0.727-0.797; resnet18 (3 seeds) mean 0.731 sd 0.052 range 0.674-0.776; vit_b_16 (1 seed) 0.774. Every between-architecture paired delta crosses zero (p 0.46-0.93); the LARGEST contrast in the table is ResNet18 against itself at a different seed (-0.102, p=0.09). Conclusion wording: **similar performance across architectures**; seed variability dominates architecture differences. 'Architecture saturation' and learning-curve plateau language are retired.

## 2. Primary estimand: POD7 risk set

Among patients not diagnosed by POD7 (undated positives excluded), RGB through POD7 (aggregation: MEAN of all photographs through POD7) predicts subsequent SSI: **AUROC 0.7607 (0.6454-0.8599), 193 patients, 15 events**. Aggregation-robust (latest 0.746, closest 0.744). POD14 secondary (0.709, 9 events); POD30 descriptive (0.538, 2 events). The pre-diagnosis truncation analysis is now a leak-robustness sensitivity, cited only as the paired same-204-patient delta (+0.0066, p=0.435).

## 3. Clinical comparison: AUPRC co-reported, prospective alignment

Full cohort paired deltas — combined vs image: dAUROC -0.0455 (p=0.467), dAUPRC +0.1194 (-0.0267-0.2725, p=0.122). POD7 risk set (identical 193 patients, information through POD7): clinical AUROC 0.728 / AUPRC 0.391; image 0.761 / 0.172; combined 0.751 / 0.306; paired combined-vs-image dAUPRC +0.134 (p=0.095). Calibration table + decision curves included (all models slope 0.80-0.88, Brier 0.079-0.083 after cross-fold Platt). Conclusion wording: **no demonstrated AUROC improvement; the AUPRC signal for adding clinical data is suggestive but not significant at 20 events** - not 'no incremental clinical value'. Imputation/scaling/C-selection are fold-internal (asserted in code).

## 4. Human raters: one adjudicated outcome

All Task-10 outputs regenerated from the locked roster label (RU-A1042 positive; the REDCap export's stale is_ssi is unused). Cohort flow: 468 rated -> 435 outcome-known (27 pos img / 9 pos pts; 33 img from 7 non-cohort pids excluded) -> **277 matched (14 pos img / 5 pos PATIENTS)**. CNN vs consensus on the matched cohort: 0.8685 vs 0.6103, paired clustered delta +0.2583 (0.0067-0.4641). **Marked supplement-grade** (5 positive patients) pending PI sign-off; not placed in main Results.

## 5. Masked retrains on the master map

Annotated subset 1886 img / 206 pts / 19 SSI+ (RU-A1308 unmaskable — documented attrition): full 0.725; wound-only 0.670 (delta -0.055, p=0.61); background-only 0.821 (delta +0.097, p=0.083). Consistent with round 1: the wound region contains no privileged signal; background performs at least as well. Alongside Aim-3 inference-masking (~0.55), the contrast is: the trained model USES the wound region, but the signal is spatially diffuse.

## 6. Reconciliation

All six cross-check items resolved - see RECONCILIATION.md (image-count provenance, RU-A1308 attrition, 0.55-vs-0.712 explanation, intraop=182 standardized, paired-delta wording, undated-positives set identity).
