# Reproducibility addendum — 2026-09-13
Responds to the five missing-file requests. All bootstrap values in this package are
canonical: 20,000 patient-level resamples, dedicated RNG seed 913. Earlier lower-limit
variants for the primary CI (0.645 / 0.647 / 0.650) were Monte-Carlo noise from 2,000-
resample runs drawing from shared RNG streams in different orders; at 20,000 resamples
the canonical CI is 0.645–0.860, matching the manuscript.

FILES
1. clinical_final_patient_oof.csv — full locked cohort (210 pts): pid, y_true, fold,
   clinical_score (intraoperative-only, temperatures range-checked to 30–43 °C, 11
   entries set missing), image_score (locked all-images mean), combined_score.
2. clinical_final_rangechecked_summary.csv / _deltas.csv — AUROC+AUPRC with 95% CIs and
   paired contrasts. Canonical: clinical 0.747, image 0.779, combined 0.794;
   combined-vs-clinical +0.048 (P=0.14); combined-vs-image +0.015 (P=0.76);
   image-vs-clinical +0.033 (P=0.68).
3. clinical_pod7_patient_oof.csv / clinical_pod7_summary.csv — same three models on the
   POD-7 risk set (193 pts / 15 events), image information through POD 7 only, same
   range-checked clinical spec. Canonical: clinical 0.740 (AUPRC 0.351), image AUPRC
   0.172, combined 0.764 (AUPRC 0.347); combined-vs-image dAUROC +0.003 (P=0.94),
   dAUPRC +0.175 (P=0.085).
4. primary_pod7_patient_scores.csv — canonical patient-level primary scores: pid,
   y_true, fold, n_images_thru_pod7, score_mean (primary), score_latest, score_closest.
   primary_pod7_canonical_metrics.csv — AUROC 0.761 (0.645–0.860), AUPRC 0.172
   (0.095–0.320), aggregation sensitivities 0.746 / 0.744.
5. healing_stage_oof_predictions.csv / healing_stage_summary.csv — locked healing-stage
   out-of-fold predictions: image AUROC 0.976, patient 0.973 (supports "~0.975").
6. sham_mask_overlap_audit.csv / _summary.csv — per-image IoU between each true incision
   mask and its relocated sham (exact replication of the sham RNG stream, seed 20260807,
   np.roll offsets 56–168 px). Median IoU 0.000; mean 0.003; 98.1% of images IoU<0.05;
   99.8% IoU<0.20; max 0.217. Relocated masks did not materially overlap the true wound.

NOTE — two manuscript rows predate the final range-check and should be updated to the
canonical values in (2) and (3): the full-cohort combined-vs-image and image-vs-clinical
deltas, and the POD-7 combined-vs-image row. Conclusions are unchanged (all NS).
