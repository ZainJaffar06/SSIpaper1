# Data reconciliation — Paper 1 (RGB/smartphone arm), round 2

Responses to the six cross-checking items, each verified against source data (2026-08-17).

## 1. Image count: 1,979 vs 1,948 — RESOLVED, structural fix underway

Both numbers are real and now both documented. The locked 2026-08-07 training frame had
**1,979 rows**, of which 30 materialized paths appear twice (31 extra rows). Verified:
every duplicate pair belongs to the **same patient and the same fold** (patient-grouped
splitting), so the locked result contains no cross-fold leakage from duplication — the
duplicates only double-weighted 31 images in training. The analytic image count is
**1,948 unique images**, which is what every prediction/fold file uses.

**Structural fix (PI point 1):** `MASTER_map_1948_seed42.csv` is now the single auditable
record — 1,948 deduplicated images, 210 patients, 20 SSI+, patient fold assignment = the
verified locked seed-42 partition (fold sizes 312/340/466/425/405). ResNet18,
EfficientNet-B0, ViT-B/16 and all masked conditions are being retrained on exactly this
map under one identical protocol (ResNet18 and EffB0 at 3 seeds for seed-variability),
so every architecture comparison becomes fold-paired. The old reconstruction report is
retained with both counts explained (see note row added to
`task3_fold_reconstruction_report.csv`).

Until the fold-paired rerun completes, the defensible wording is
**"similar performance across architectures,"** not "architecture saturation."

## 2. Task 8 frame 206/19 — EXPLAINED, benign attrition

The masked conditions run on images with parseable incision-mask geometry
(incision-line/wound-bed annotations). Four patients have **zero maskable images** and
drop: RU-A1081, RU-A1131, RU-A1304, and **RU-A1308 — the missing positive** (no
annotation source with valid incision shapes for any of its images). 1,948 → 1,886
images; 210/20 → 206/19. All three conditions share this frame, so the internal
wound-vs-background contrast is valid. Every table/figure showing Task-8 numbers now
carries the frame label "annotated subset: 1,886 img / 206 pts / 19 SSI+", and the
full-image 0.757 is labeled as the subset baseline, never presented against 0.779.

## 3. Wound-only 0.55 vs 0.712 — TWO EXPERIMENTS, stated side by side

| Condition | Method | Frame | Patient AUROC | What it tests |
|---|---|---|---|---|
| Wound-only ≈0.55 | **Inference-masking** the trained model (Aim 3, fold-0 explainability subset) | fold-0 annotated subset | ~0.55 | Does the trained model **use** the wound region? → yes, masking its input collapses it |
| Wound-only 0.712 | **Retraining from scratch** on wound-only inputs (Task 8, all folds) | 1,886/206/19 | 0.712 | Does the wound region **contain** privileged signal? → no more than background (0.719) |

The contrast IS the finding: the deployed model relies on the wound crop, but the
discriminative information is spatially diffuse — a fresh model recovers most performance
from either region. This paragraph goes in Results verbatim so the two numbers are never
read as one quantity.

## 4. Intraop coverage 182 vs 183 — STANDARDIZED to 182

On the Paper-1 frame (210 patients), intraop covariates cover **182 patients (17 SSI+)**;
this is the number every Paper-1 methods/coverage table now uses. The earlier 183 came
from a different frame/coverage rule in the Aim-5a run (the both-arm frame gives 193) and
is superseded for Paper 1.

## 5. Estimand denominators — WORDING FIXED

The leak-robustness claim is now cited only as the **paired 204-patient delta:
+0.0066 (p=0.41)** — same patients, both scoring rules. The two AUROCs are reported each
with its own n: locked all-images **0.7792 (210 pts, 20 SSI+)**; detect-before-diagnosis
**0.7811 (204 pts, 18 SSI+; excludes RU-A1345/RU-A1347, no diagnosis date)**. The phrase
"0.781 vs 0.779" is retired.

## 6. 'No dx date' vs 'unknown-dx' — CONFIRMED SAME POPULATION

Both are exactly {RU-A1345, RU-A1347}, driven by a single constant in the code
(`UNKNOWN_DX_POS`). Task 2 excludes them from the pre-diagnosis estimand (their
truncation time is unknown); the Task-9/landmark rows report them both ways (excluded =
primary, retained = labeled sensitivity). One undated pair, two deliberate handlings,
now stated in both files.
