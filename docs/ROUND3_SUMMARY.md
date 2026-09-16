# Paper 1 — Round 3 (2026-08-19)

All blocking, should-run, and provenance items delivered. Everything is fold-paired on `MASTER_map_1948_seed42.csv`; every trained number now carries a 3-seed distribution.

## Multi-seed distributions (blocking 1)

Patient AUROC, mean±SD over seeds {20260807, 42, 7}: efficientnet_b0 0.755±0.037 (0.727-0.797); resnet18 0.731±0.052 (0.674-0.776); vit_b_16 0.725±0.049 (0.677-0.774). Distributions fully overlap; within-architecture seed spread (up to 0.10) exceeds every between-architecture difference. Learning curve (EffB0, mean±SD): 33% -> 0.718±0.016; 66% -> 0.746±0.057; 100% -> 0.755±0.037 — no significant data-volume trend. Seed-aware confound deltas: CNN minus combined probe averages -0.009±0.052 across ResNet seeds (zero-centred).

## Masking negative control (blocking 2)

Sham masks (each image's own mask randomly relocated; area/shape preserved): sham_wound_only 0.8168; sham_background_only 0.6541. Key paired results: sham-wound = real background (delta -0.005, p=0.94); real wound-only is marginally WORSE than a random equal-area patch (+0.147 in favour of sham, p=0.077). Reading: the background>wound geometry is real (not a masking artifact), the wound crop carries no privileged signal, and differences among masking conditions sit within training-seed noise — so we claim signal DIFFUSENESS, not background superiority.

## Background-only confound probe (blocking 3)

The probe battery (POD / acquisition / past-only behaviour / combined) does NOT explain the background model: background-minus-probe deltas +0.09 to +0.16 (p 0.07-0.28), Spearman <= 0.36. Combined with the sham control: context carries signal beyond the measured confounds; the paper frames this as contextual/diffuse information with unmeasured-context caveats.

## POD7 risk-set calibration + operating points (should-run 1)

patient_auroc(raw) 0.7607 (0.6501-0.8596); brier(platt) 0.0748 (0.047-0.1055); cal_slope(platt) 0.6319 (0.3003-1.0525); cal_intercept(platt) -0.855 (-1.9914-0.3103). Rule-out row: at sensitivity 14/15 (0.93), threshold 0.028 clears ~49% of patients (spec 0.53, flag rate 0.51); full sensitivity clears 23%.

## Utility metrics with CIs (should-run 2)

Brier / calibration slope+intercept now carry patient-bootstrap 95% CIs for clinical/image/combined (slopes 0.80-0.88, CIs ~0.25-1.55 — wide, stated); decision curves have bootstrap bands (r3_decision_curves_with_cis.csv).

## Task 10 pre-dx restriction (should-run 3)

Pre-dx-only images: 271 images, 8 positive images, 3 positive PATIENTS. CNN 0.894 vs consensus 0.6328; delta +0.2612 loses significance (p=0.333). Confirms supplement-grade placement.

## Provenance (required deliverables)

Masking cohort audit: all 62 excluded images are 'no embedded annotation source' (zero parse failures); 4 patients fully excluded (incl. SSI+ RU-A1308), 15 retained with fewer images (r3_masking_cohort_audit.csv; example full/wound/background/sham images in R3_F1). Raw OOF export: r3_raw_oof_image_level.csv (26,962 rows; every architecture x seed x condition) + patient-level aggregation. METHODS_CONFIRMATION.md states the master-map / patient-grouped / out-of-fold / clustered-resampling discipline.
