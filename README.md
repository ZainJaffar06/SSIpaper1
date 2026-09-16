# How much infection-specific information does a smartphone wound photograph contain?

**Paper 1 (RGB / smartphone arm) — SSI surveillance information audit**
Project AiCCESS · Rutgers · Zain Jaffar

This repository contains the analysis pipeline, canonical results, and figures for a prospective study of smartphone-based surgical site infection (SSI) surveillance. The study does not try to build the best possible classifier. It asks how much infection-related information a single postoperative RGB photograph actually contains, and audits where a model's apparent performance comes from.

> **Status:** manuscript under PI review. Some analytic decisions are still open (see [Open decisions](#open-decisions)). Numbers in `results/canonical/` are the current source of truth.

---

## Headline results

Cohort: 1,948 quality-gated photographs, 210 patients, 20 SSI events (9.5%).

| Analysis | Result |
|---|---|
| **Primary:** POD-7 landmark, patients undiagnosed by day 7 (193 pts / 15 events) | AUROC **0.761** (95% CI 0.645–0.860), AUPRC 0.172 (0.095–0.320) |
| Leak sensitivity (pre-diagnosis images only, 204 pts / 18 events) | Paired Δ +0.007 (−0.011 to +0.025), P=0.44 |
| Architectures, 3 seeds each on identical folds | EfficientNet-B0 0.755±0.037 · ResNet18 0.731±0.052 · ViT-B/16 0.725±0.049 (no significant differences) |
| Context-only model (timing + acquisition + behavior, no wound pixels) | Δ vs image model −0.009±0.052 |
| Masking with sham controls (206 pts / 19 events) | Sham wound 0.817 ≈ true background 0.821 (P=0.94); sham-vs-true mask IoU median 0.000 |
| Healing-stage positive control, same photographs | AUROC ≈0.975 (vs ≈0.755 for SSI) |
| Clinical vs image vs combined (range-checked, full cohort) | 0.747 / 0.779 / 0.794; combined vs clinical +0.048, P=0.14 |
| Clinician panel (15 raters) | Fleiss κ = 0.105 (exploratory) |

Taken together, the results are consistent with an information limit in single RGB photographs rather than a model-capacity limit.

---

## Repository layout

```
src/
  lib.py                  shared metrics, patient aggregation, clustered bootstrap, grouped OOF
  00_cohort/              analysis-frame assembly (build_base.py gates on exact reproduction
                          of the locked result) + de-identified chart-review parsing
  01_folds/               locked seed-42 fold reconstruction (verified exact) + frozen master map
  02_training/            end-to-end fine-tuning: 3 architectures x 3 seeds, learning curve,
                          sham-mask retraining
  03_estimand/            leak audit, POD-7 risk set (primary), landmark + serial analyses
  04_probes/              context-only probes (POD, acquisition, past-only behavior)
  05_masking/             wound-only / background-only retraining on incision annotations
  06_clinical/            clinical vs image vs combined (nested CV, fold-internal imputation)
  07_calibration/         cross-fold Platt/isotonic recalibration, operating points
  08_raters/              clinician panel: kappa reproduction, adjudicated-label comparison
  09_packaging/           exports, figures, canonical reproducibility addendum,
                          path de-identification
results/
  canonical/              SOURCE OF TRUTH: patient-level OOF scores + canonical metrics
                          (20,000-resample bootstrap, dedicated seed)
  master_models/          per-architecture x seed OOF predictions on the frozen master map
  MASTER_map_1948_seed42.csv   the single image-to-fold map used by every analysis
  round1/ round2/ round3/ historical outputs from each PI review round (see note below)
figures/                  all manuscript and supplementary figures
docs/                     round summaries, data reconciliation, methods confirmation
manuscript/               current manuscript draft (.docx) + draft generator scripts
```

### Canonical vs historical results

The analysis went through three PI review rounds and several audits. Each round's outputs are kept for provenance, but **some earlier values were superseded** — for example, the full-cohort clinical comparison was re-run after implausible temperature entries were range-checked, which removed a spurious significant result. Always cite `results/canonical/`. `results/canonical/CANONICAL_README.md` documents every file and the superseded values.

---

## Running the pipeline

```bash
pip install -r requirements.txt
```

Scripts expect the locked data package (images, manifests, locked OOF predictions) and the chart-review and intraoperative exports, which are **not** in this repository (see below). Paths are set at the top of each script.

Suggested order:

1. `00_cohort/parse_baseline.py`, then `00_cohort/build_base.py` (aborts unless the locked headline reproduces exactly: 1,948 images, image AUROC 0.7554, patient AUROC 0.7792)
2. `01_folds/task3_folds.py`, then `01_folds/freeze_map.py`
3. `02_training/finetune.py`, `retrain_master.py`, `round3_compute.py` (GPU; about 30 hours total on Apple M-series)
4. `03_estimand/` → `08_raters/` (CPU)
5. `09_packaging/repro_addendum.py` regenerates `results/canonical/`

Conventions throughout: patient-grouped 5-fold CV, strictly out-of-fold predictions, patient-level mean aggregation, patient-clustered bootstrap CIs, paired comparisons within the same bootstrap replicate, and model differences reported as distributions across seeds.

Manuscript generator scripts (`manuscript/draft_generators/*.js`) need Node and `npm install docx`.

---

## Data and privacy

This repository contains **no patient photographs, no raw clinical extracts, and no direct identifiers.**

- Patients appear only as study-assigned codes (`RU-A####`).
- Original image file paths were replaced with a hashed `image_id` (`src/09_packaging/deidentify_paths.py`), because some source folder names embedded dates of service. The path-to-ID mapping is stored offline and is never committed.
- Relative postoperative days (POD) are retained; calendar dates are not.
- Figures with patient wound photographs are kept on institutional storage only.
- `.gitignore` blocks images, model weights, and raw extracts.

The repository still contains de-identified row-level study data (outcome labels and model predictions by study code).
---

## Open decisions

1. Approval of the 30–43 °C physiologic range check for intraoperative temperature covariates (11 entries set to missing).
2. Handling of 318 photographs labeled "post" with no day number. They are currently assigned to day 0 by convention; excluding them gives a POD-7 AUROC of 0.730 (187 patients, 14 events). EXIF capture dates could resolve this.
3. Protocol imaging-day wording (the data show a day-of-surgery cluster plus days 4/7/10/14/30).
4. Citations and named comparators for the Discussion.


