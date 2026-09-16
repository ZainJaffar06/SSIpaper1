# Paper 1 (smartphone arm) — executed review tasks, results (2026-08-15)

Scope: RGB/smartphone arm only. Locked curated cohort **1,948 unique images / 210 patients / 20 SSI-positive patients**; all CV patient-grouped; all CIs patient-bootstrap.

## The primary estimand is now leak-free (task 2)

Detect-before-diagnosis (positives scored on strictly pre-diagnosis images only): patient AUROC **0.7811** (0.6762–0.8731), vs locked all-images 0.7792. Paired difference 0.0066 — the locked headline never depended on post-diagnosis images (39/182 dx-linked positive images are post-dx, concentrated in 6 patients). 2 positives (RU-A1345, RU-A1347) lack a diagnosis date and are excluded from this estimand (reported, not hidden). This is RECOGNITION of an evolving infection before the formal diagnosis, not forecasting from a pre-morbid state.

Landmark analyses (all images up to POD t, label = eventual SSI): POD7 = 0.7643, POD14 = 0.7808, POD30 = 0.7747. Among patients not yet diagnosed at t, discrimination attenuates as diagnosed positives drop out (n_pos 17 → 11 → 4). The leak-free serial model does NOT beat the single-image mean at any landmark (task 9).

## Architecture ceiling is now end-to-end, not frozen-probe (task 5)

EfficientNet-B0 fine-tuned end-to-end on the published folds: image 0.7404 / patient 0.7937. ViT-B/16: image 0.7415 / patient 0.7626. Locked trained ResNet18: 0.7554 / 0.7792. All three architectures land on the same plateau; the ~0.76-0.79 patient ceiling is a property of the data, not the backbone. Learning curve (EffB0): patient AUROC 0.7276@33% → 0.6197@66% → 0.7429@100% of training patients — non-monotonic and FLAT WITHIN SEED NOISE (single subsample per fraction; the 100% rerun differs from the main run by ~0.05 through RNG state alone at 20 events). No data-volume trend is detectable; read as consistent with a substrate/label limit, not as a smooth saturation curve.

## Confound probes, leak-fixed (task 6)

With FUTURE-INFORMATION features removed (no total-visit counts), the trivial-probe ladder at patient level: POD-only 0.6091, acquisition 0.6513, past-only behavior 0.6516, combined 0.6995. The CNN (0.7792) significantly beats acquisition alone (Δ+0.1279 p=0.018) and every probe at image level (all p≤0.004), but its patient-level edge over the combined probe is Δ+0.0797 (p=0.12, NS at 20 events): roughly 0.70 of the 0.78 is reachable from timing + submission pattern + trivial image statistics. Lead with this number when framing the imaging claim. (Sharpness at 224px is resolution-degraded — ranking signal only.)

## Clinical vs image on the FULL locked cohort (task 4) — ascertainment-corrected

The adversarial audit caught a blocker in the first version: chart-review coverage is outcome-correlated (18/20 positives covered vs 77/190 negatives) — the bare coverage indicator alone scores AUROC 0.7474, so missingness indicators leaked ascertainment, not clinical signal (the leaky model, 0.7686, is retained in the workbook as a do-not-cite diagnostic). The clean full-cohort comparison uses intraop covariates only (coverage AUROC 0.4908 = neutral): clinical 0.652 (0.4823–0.8029) vs image 0.7792 vs combined 0.7337; all paired deltas NS at 20 events (image minus clinical +0.1272, p=0.185). Chart-based (demographics/comorbidity) comparisons remain interpretable only WITHIN the chart-covered subgroup — that is the existing 5b result (all 0.56–0.66, NS).

## Calibration + operating points (task 7)

Raw locked scores are miscalibrated (slope 0.5504, intercept -1.0266). Cross-fold Platt improves calibration (slope 0.7258, Brier 0.087); the isotonic slope/intercept are suppressed as degenerate at 20 events (57/210 patients at exact 0/1 — audit finding). Operating points on the RAW locked score (primary; identical under any monotone deployed calibrator — the pooled cross-fold Platt table understated sens/PPV and is kept only as a labeled secondary): spec 0.80 → sens 0.60 (0.30–0.83), PPV 0.24; spec 0.90 → sens 0.25 (0.05–0.50), PPV 0.21; spec 0.95 → sens 0.15 (0.00–0.31), PPV 0.23. Modest sensitivity at deployable specificity — state this plainly.

## Wound-region retrains (task 8)

Retrained (not inference-masked) EffB0 across all 5 folds on the annotated smartphone subset (1886 images, 206 patients, 19 SSI+): full_image 0.7571 (0.6363–0.8609); wound_only 0.7124 (0.5525–0.8478); background_only 0.7188 (0.5996–0.8303). Deltas vs full image: wound_only_minus_full(patient) -0.0448 (p=0.402); background_only_minus_full(patient) -0.0383 (p=0.51). This tests whether the wound region CONTAINS the signal (training-time), upgrading the earlier inference-masking ablation (which only tested whether the model USED it).

## Cohort bookkeeping (tasks 1 + 3)

Adjudication: all five PI-flagged patients (RU-A1357/1303/1051/1327/1355) contribute **zero images to both smartphone arms** — every Paper-1 number is identical under any adjudication outcome. RU-A1303 has 4 images in the (non-Paper-1) both-arm only. The double-enrollment still needs PI resolution for the enrollment log itself.

Fold maps: the locked seed-42 StratifiedGroupKFold assignment was reconstructed and verified EXACT against the saved fold-0 test indices for all four arms (task3_fold_reconstruction_report.csv); full per-image fold maps are published for the locked models and the new fine-tunes (single fold-map file each, patient-disjointness asserted).

Three-subset table: PRIMARY: locked curated smartphone: 0.7792 (0.6761–0.8643), n=210, 20 SSI+, prev 9.5%; sensitivity: expanded/uncurated smartphone: 0.6906 (0.5701–0.8035), n=228, 20 SSI+, prev 8.8%; context only (NOT Paper 1): smartphone+thermal patients: 0.6803 (0.5665–0.7877), n=221, 23 SSI+, prev 10.4%.

## Clinicians vs CNN on the same photographs (task 10 — now COMPLETE)

Recovered the full per-rating export (7,062 ratings; 'Clean Real Study Rows' sheet of the SharePoint workbook); extraction verified by exact reproduction of the workbook's own summary (Fleiss kappa 0.1054, Pbar 0.6865, Pe 0.6496). On the 277 rated images that map to the locked cohort (adjudicated roster labels; 3 RU-A1042 images relabeled from a stale REDCap snapshot): 15-clinician consensus AUROC **0.6103** (0.4746–0.7502) vs CNN **0.8685** (0.7278–0.9609); paired patient-clustered Δ **+0.2583** (p=0.044) — the CNN significantly outperforms pooled clinician judgment on the same photographs. Individual raters: sens ~0.37–0.40, spec ~0.78; surgeons vs non-surgeons similar (Surgeon 0.5925; Non-surgeon 0.6257). Caveats: 191/468 rated images fall outside the locked (pid, day) set (139 curation-dropped days, 52 images from 11 non-cohort patients); labels are patient-level for both humans and CNN.
