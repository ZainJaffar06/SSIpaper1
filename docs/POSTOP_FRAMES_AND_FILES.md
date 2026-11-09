# Analysis frames of the postoperative-only manuscript, and the file behind each number

Written 2026-11-06 for Mahdi Babaei's question "which result files does each set of numbers
come from". Every result in the current manuscript (the 2026-10-26 version with the round-2
edits) is computed on the postoperative frame or on one of its subsets. There is no longer a
210-patient result anywhere in the paper: the exploratory clinical comparison, Table 1 and
every supplementary table were re-run on the 171-patient postoperative frame. The 210-patient
locked cohort appears only in the cohort description (Results paragraph 1, Methods cohort
paragraph, Figure 1) as the state before intraoperative photographs were removed.

All paths below are under `results/collaboration/` unless stated. Per-photograph out-of-fold
predictions of every postoperative network are in `results/postop_models/` (columns `pid,
image_id, fold, y_true, proba`; `y_true` is the final 26-patient label set; `image_id` is the
hashed identifier of `MASTER_map_1948_seed42.csv`).

## Frames

| Frame | Photographs / patients / SSI | Defined by |
|---|---|---|
| Locked cohort (before intraoperative exclusion) | 1,948 / 210 / 20 | `per_photo_frame_1948.csv` (all rows); folds from `MASTER_map_1948_seed42.csv` (folds only, never its label column) |
| **Postoperative frame** (the analytic frame) | 1,341 / 171 / 13 | `per_photo_frame_1948.csv`, rows with `photo_type == postoperative` |
| POD-7 risk set (primary estimand) | 544 photographs through POD 7 / 132 / 8 | `postop_pod7_patient_scores.csv`; rule: patients of the postoperative frame with >= 1 photograph through POD 7, excluding anyone diagnosed on or before POD 7 and any SSI patient with no recorded diagnosis day |
| POD-14 / POD-30 landmarks | 157 / 5 and 159 / 1 | `postop_primary_and_landmarks.csv` |
| Dated frame (diagnosis-timing analyses) | 170 / 12, pre-diagnosis comparison on 168 / 10 | `postop_prediagnosis_truncation_fullcohort.csv` |
| Mask-eligible frame (incision geometry complete) | 1,296 / 168 / 12 | subset of the postoperative frame with a usable incision annotation (`postop_masks.npz` offline; the image list is the `pid, image_id` set of any `results/postop_models/mask_*_seed*.csv`) |
| Healing-stage positive control | 549 / 140 (84 early, 465 late) | `postop_stage_control.csv`, `results/postop_models/stage_control.csv` |
| Uncurated postoperative photographs (curation sensitivity) | 3,110 / 225 / 18 | `postop_curation_sensitivity_postop.csv`; scored with the saved out-of-fold predictions of the earlier expanded-frame model, not retrained |
| Clinician panel (supplementary) | shared image set rated by 15 clinicians | `results/round1/task10_*.csv`, `results/round2/task10v2_per_rater_performance.csv`; this set predates the postoperative restriction (flagged for Dr. Kewalramani) |

Patient folds are the locked seed-42 assignment throughout; the all-modality version with
final labels is `SHARED_fold_map_v2.csv`.

## Where each set of numbers comes from

| Manuscript item | Frame | File(s) |
|---|---|---|
| Cohort flow, Figure 1 counts | locked cohort -> postoperative frame | `postop_cohort_flow_patients.csv`, `postop_cohort_flow_photographs.csv`, `per_photo_frame_1948.csv` |
| Table 1 (baseline) | postoperative frame, 171 / 13 | `baseline_table1_postop.csv` |
| Primary POD-7 AUROC 0.705 (0.466-0.921), AUPRC 0.377, latest / closest variants, POD 14 and 30 | POD-7 risk set 132 / 8 | `postop_primary_and_landmarks.csv`, `postop_primary_auprc_ci.csv`, `postop_pod7_patient_scores.csv`; network `results/postop_models/resnet18_seed20260807.csv` |
| Operating points, calibration (Table 2 / S6) | POD-7 risk set | `postop_operating_points.csv`, `postop_calibration.csv`, `postop_calibration_ci.csv` |
| Architectures x seeds (Table S1 / Table 4) | postoperative frame | `postop_architectures.csv`; networks `results/postop_models/{resnet18,efficientnet_b0,vit_b_16}_seed{20260807,42,7}.csv` |
| Architectures on the mask-eligible subset | mask-eligible frame | `postop_architectures_restricted.csv` |
| Learning curve (Table S2) | postoperative frame | `postop_learning_curve.csv`; networks `results/postop_models/lc_{0.333,0.667}_seed*.csv` |
| Clinical 8-variable 0.622, image 0.744, stacked 0.881, clinical v2 0.662 (Table 3) and contrasts | postoperative frame 171 / 13 | `postop_clinical_models.csv`, `postop_clinical_contrasts.csv`, `postop_clinical_partition_stability.csv` |
| POD-7 clinical / image / stacked (Table 4) | POD-7 risk set 132 / 8 | `postop_clinical_pod7.csv`, `postop_clinical_pod7_contrast.csv` |
| Masking, five conditions x three seeds and the six contrasts (Table S4) | mask-eligible frame | `postop_masking_absolute.csv`, `postop_masking_contrasts.csv`; networks `results/postop_models/mask_{full_image,wound_only,background_only,sham_wound_only,sham_background_only}_seed*.csv` |
| Sham-mask overlap audit | mask-eligible frame | `postop_single_shift_overlap_audit.csv` |
| Permutation test of the sham contrast | mask-eligible frame | `postop_permutation_summary.csv`, `postop_permutation_null.csv` |
| Binary mask as image | mask-eligible frame | `postop_maskimage_finetuned.csv`; networks `results/postop_models/maskimage_seed*.csv` |
| Frozen geometry descriptors; geometry encodes region | mask-eligible frame | `postop_frozen_geometry.csv`, `postop_geometry_encodes_region.csv` |
| Context probes (POD, acquisition, prior behaviour, combined) and CNN minus probe | mask-eligible frame | `postop_context_probes.csv`, `postop_cnn_vs_probe_paired.csv`, `postop_seedaware_probe_deltas.csv`, `postop_seedaware_probe_delta_summary.csv` |
| Full-frame context probe vs CNN | postoperative frame (1,341 images / 171 patients) | `postop_cnn_vs_context_fullframe.csv` |
| Foundation-model probes (DINOv2, BiomedCLIP) and paired contrasts (Table S12) | mask-eligible frame | `postop_foundation_probes.csv`, `postop_foundation_paired_contrasts.csv` (DINOv2), `postop_foundation_paired_contrasts_biomedclip.csv` (BiomedCLIP, added 2026-11-06), `postop_foundation_paired_contrasts_both_encoders.csv`, `postop_foundation_rank_agreement.csv`; patient scores `item9_patientscores_{dinov2,biomedclip}_*_postop.csv` |
| Healing-stage positive control (Table S7) | stage frame | `postop_stage_control.csv` |
| Pre-diagnosis truncation | dated frame / POD-7 risk set | `postop_prediagnosis_truncation.csv`, `postop_prediagnosis_truncation_fullcohort.csv` |
| Region strata, CDC class | POD-7 risk set | `postop_region_strata.csv` |
| Incisional-only sensitivity | POD-7 risk set | `postop_incisional_sensitivity.csv` |
| Curation sensitivity | uncurated postoperative photographs | `postop_curation_sensitivity_postop.csv` |
| Recommended-controls checklist | - | `postop_recommended_controls.csv`, `postop_recommended_controls_framework.csv` |
| New references (verified against publishers) | - | `postop_new_references_verified.csv` |
| Old-to-new number map used while editing | - | `postop_renumber_map.csv` |
| Pre-landmark training (requested 2026-11-06, not in the manuscript): ResNet18 x 3 seeds trained on photographs through POD 7 only (`trainpod7`) or on pre-diagnosis photographs only (`trainpredx`) | postoperative frame, evaluated on the POD-7 risk set and on all photographs | `postop_prelandmark_training_sets.csv` (training-set sizes per fold), `postop_prelandmark_training.csv` and `postop_prelandmark_training_contrasts.csv` (each run vs the primary), `postop_prelandmark_seed_matched*.csv` (each run vs the full-training ResNet18 of the same seed, with three-seed summaries); networks `results/postop_models/prelandmark_{trainpod7,trainpredx}_resnet18_seed{20260807,42,7}.csv`; scripts `retrain_prelandmark_postop.py`, `prelandmark_eval.py`, `prelandmark_seed_matched.py` |
| Tracker patients without an outcome label (74) | outside every frame | `postop_unlabelled_tracker_patients.csv` |
| Patients with postoperative smartphone photographs on AWS that are in neither the locked cohort nor the 74 (28; Mahdi Babaei's cohort-lock audit, 2026-11-08): why each is missing, local copies, what is needed to add them | outside every frame | `postop_missing_phone_photo_patients_2026-11-08.csv` |
| Photo-level reconciliation of every distinct phone photograph on AWS (S3 listing of 2026-11-09, re-listed from CloudShell, copies collapsed by content hash) against the annotation tracker, with the reason each photograph is or is not in the locked set (2026-11-09). Per-patient counts only; the photo-level workbooks with file names stay inside the study team (AWS folder names carry visit dates) | all patients with images on AWS | `postop_photo_exclusion_reasons_by_patient_2026-11-09.csv`; 28-patient quality-check summary `postop_quality_check_28_patients_summary_2026-11-09.csv`; 34 locked/74-list patients with only intraoperative photographs in the data `postop_intraop_only_34_patients_reasons_2026-11-09.csv` |

Correction (2026-11-09) to the 2026-11-08 table: the AWS "postoperative phone photo" counts of 2026-11-08
double-counted copies and counted thermal images stored under app file names (`file-uploads/Day_N`) as
phone photos, and the app file names `Day_N` are not postoperative days. The corrected per-patient
numbers and the corrected per-patient explanation are in the four `*_2026-11-09` columns of
`postop_missing_phone_photo_patients_2026-11-08.csv`; the earlier columns are kept as sent.

Files that must not be used for any number: `rgb_intraoperative_scores.csv`,
`rgb_intraoperative_summary.csv`, `results/canonical/`, and the `y_true` column of
`MASTER_map_1948_seed42.csv` (they carry the superseded label set, and the intraoperative
scores are outside the postoperative frame).
