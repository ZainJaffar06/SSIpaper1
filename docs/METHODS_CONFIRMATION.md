# Methods confirmation (round 3)

All analyses in PAPER1_ROUND2 and round-3 files used:
- the locked 1,948-image master file `MASTER_map_1948_seed42.csv` (210 patients, 20 SSI+),
  whose patient fold assignment is the verified locked seed-42 StratifiedGroupKFold partition;
- patient-grouped 5-fold cross-validation with folds asserted patient-disjoint;
- strictly out-of-fold predictions for every model, probe, and calibrator
  (imputation means, scaler statistics, inner-CV hyperparameter selection, and Platt
  calibrators are all fit inside the training fold only);
- patient-clustered resampling for every interval and paired comparison
  (patients are the bootstrap unit at image level; patient-level frames use plain
  patient bootstrap, 2,000 resamples, seed 20260807);
- multi-seed training (seeds 20260807, 42, 7) for ResNet18, EfficientNet-B0, ViT-B/16
  and the learning-curve fractions, reported as mean±SD distributions, not point estimates.
