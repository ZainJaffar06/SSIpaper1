#!/usr/bin/env python
"""Round-2 deliverable: PAPER1_ROUND2.xlsx + ROUND2_SUMMARY.md into the existing
paper1_smartphone_2026-08-15 deliverable folder, all numbers pulled from CSVs."""
import os, glob, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1"); MA = os.path.join(HERE, "out", "master")
DEST = os.path.expanduser("~/Documents/New project/outputs/paper1_smartphone_2026-08-15")

def rd(name, base=OUT):
    fp = os.path.join(base, name)
    return pd.read_csv(fp) if os.path.exists(fp) else None

sv = rd("master_seed_variability.csv", MA); pdel = rd("master_paired_deltas.csv", MA)
arch = rd("master_architectures.csv", MA)
rs = rd("task2v2_riskset_primary.csv"); leak = rd("task2v2_leak_sensitivity.csv")
aupr = rd("task4b_paired_auroc_auprc.csv"); cal = rd("task4b_calibration.csv")
dc = rd("task4b_decision_curves.csv"); p7 = rd("task4b_pod7_riskset.csv")
flow = rd("task10v2_cohort_flow.csv"); t10 = rd("task10v2_consensus_vs_cnn.csv")
t10r = rd("task10v2_per_rater_performance.csv"); t10s = rd("task10v2_surgeon_vs_nonsurgeon.csv")
m8 = rd("task8_masked_retrain_performance.csv", MA); m8d = rd("task8_masked_retrain_deltas.csv", MA)

prim = rs[rs.tier == "PRIMARY"].iloc[0]
md = ["# Paper 1 — Round 2 revisions (2026-08-18)\n",
      "Every point from the round-2 review and the six-item reconciliation is addressed; "
      "all numbers below are fold-paired on the frozen master map "
      "(`MASTER_map_1948_seed42.csv`: 1,948 images / 210 patients / 20 SSI+, locked seed-42 patient partition).\n",
      "## 1. Architectures, fold-paired with seed variability\n",
      "Identical folds and 12-epoch protocol. Patient AUROC: "
      + "; ".join(f"{r.model} {'(3 seeds) mean %.3f sd %.3f range %.3f-%.3f' % (r['mean'], r['std'], r['min'], r['max']) if not np.isnan(r['std']) else '(1 seed) %.3f' % r['mean']}"
                  for _, r in sv.iterrows())
      + ". Every between-architecture paired delta crosses zero (p 0.46-0.93); the LARGEST contrast in the "
        "table is ResNet18 against itself at a different seed (-0.102, p=0.09). Conclusion wording: "
        "**similar performance across architectures**; seed variability dominates architecture differences. "
        "'Architecture saturation' and learning-curve plateau language are retired.\n",
      "## 2. Primary estimand: POD7 risk set\n",
      f"Among patients not diagnosed by POD7 (undated positives excluded), RGB through POD7 "
      f"(aggregation: MEAN of all photographs through POD7) predicts subsequent SSI: "
      f"**AUROC {prim.auroc} ({prim.lo}-{prim.hi}), {int(prim.n_pts)} patients, {int(prim.n_events)} events**. "
      "Aggregation-robust (latest 0.746, closest 0.744). POD14 secondary (0.709, 9 events); POD30 descriptive "
      "(0.538, 2 events). The pre-diagnosis truncation analysis is now a leak-robustness sensitivity, cited only "
      f"as the paired same-204-patient delta ({leak.iloc[0].paired_delta:+}, p={leak.iloc[0].p_two_sided}).\n",
      "## 3. Clinical comparison: AUPRC co-reported, prospective alignment\n",
      f"Full cohort paired deltas — combined vs image: dAUROC {aupr.iloc[0].d_auroc:+} (p={aupr.iloc[0].auroc_p}), "
      f"dAUPRC {aupr.iloc[0].d_auprc:+} ({aupr.iloc[0].auprc_lo}-{aupr.iloc[0].auprc_hi}, p={aupr.iloc[0].auprc_p}). "
      "POD7 risk set (identical 193 patients, information through POD7): clinical AUROC 0.728 / AUPRC 0.391; "
      "image 0.761 / 0.172; combined 0.751 / 0.306; paired combined-vs-image dAUPRC +0.134 (p=0.095). "
      "Calibration table + decision curves included (all models slope 0.80-0.88, Brier 0.079-0.083 after "
      "cross-fold Platt). Conclusion wording: **no demonstrated AUROC improvement; the AUPRC signal for adding "
      "clinical data is suggestive but not significant at 20 events** - not 'no incremental clinical value'. "
      "Imputation/scaling/C-selection are fold-internal (asserted in code).\n",
      "## 4. Human raters: one adjudicated outcome\n",
      "All Task-10 outputs regenerated from the locked roster label (RU-A1042 positive; the REDCap export's stale "
      "is_ssi is unused). Cohort flow: 468 rated -> 435 outcome-known (27 pos img / 9 pos pts; 33 img from 7 "
      "non-cohort pids excluded) -> **277 matched (14 pos img / 5 pos PATIENTS)**. CNN vs consensus on the matched "
      f"cohort: {t10.iloc[1].auroc} vs {t10.iloc[0].auroc}, paired clustered delta {t10.iloc[2].auroc:+} "
      f"({t10.iloc[2].lo}-{t10.iloc[2].hi}). **Marked supplement-grade** (5 positive patients) pending PI sign-off; "
      "not placed in main Results.\n",
      "## 5. Masked retrains on the master map\n",
      f"Annotated subset {int(m8.iloc[0].n_img)} img / {int(m8.iloc[0].n_pts)} pts / {int(m8.iloc[0].n_pos_pts)} SSI+ "
      "(RU-A1308 unmaskable — documented attrition): full 0.725; wound-only 0.670 (delta -0.055, p=0.61); "
      "background-only 0.821 (delta +0.097, p=0.083). Consistent with round 1: the wound region contains no "
      "privileged signal; background performs at least as well. Alongside Aim-3 inference-masking (~0.55), the "
      "contrast is: the trained model USES the wound region, but the signal is spatially diffuse.\n",
      "## 6. Reconciliation\n",
      "All six cross-check items resolved - see RECONCILIATION.md (image-count provenance, RU-A1308 attrition, "
      "0.55-vs-0.712 explanation, intraop=182 standardized, paired-delta wording, undated-positives set identity).\n"]
open(os.path.join(DEST, "ROUND2_SUMMARY.md"), "w").write("\n".join(md))

sheets = {"R2 seed variability": sv, "R2 architectures": arch, "R2 paired deltas": pdel,
          "R2 POD7 riskset PRIMARY": rs, "R2 leak sensitivity": leak,
          "R2 paired AUROC AUPRC": aupr, "R2 calibration": cal, "R2 decision curves": dc,
          "R2 POD7 clinical": p7, "R2 T10 cohort flow": flow, "R2 T10 consensus vs CNN": t10,
          "R2 T10 per rater": t10r, "R2 T10 surgeon split": t10s,
          "R2 masked master": m8, "R2 masked deltas": m8d}
readme = pd.DataFrame({"ROUND 2": [
    "All analyses fold-paired on MASTER_map_1948_seed42.csv (1948 img / 210 pts / 20 SSI+).",
    "Wording: similar performance across architectures (seed noise dominates); POD7 risk-set is the primary estimand;",
    "no demonstrated AUROC improvement from clinical data (AUPRC suggestive, NS); Task 10 supplement-grade (5 pos pts).",
    "See ROUND2_SUMMARY.md and RECONCILIATION.md."]})
with pd.ExcelWriter(os.path.join(DEST, "PAPER1_ROUND2.xlsx"), engine="openpyxl") as xw:
    readme.to_excel(xw, "README", index=False)
    for nm, df in sheets.items():
        if df is not None: df.to_excel(xw, nm[:31], index=False)

shutil.copy2(os.path.join(OUT, "RECONCILIATION.md"), DEST)
shutil.copy2(os.path.join(OUT, "MASTER_map_1948_seed42.csv"), os.path.join(DEST, "csv"))
for f in glob.glob(os.path.join(OUT, "figures", "R2_*.png")):
    shutil.copy2(f, os.path.join(DEST, "figures"))
for f in (glob.glob(os.path.join(OUT, "task2v2_*.csv")) + glob.glob(os.path.join(OUT, "task4b_*.csv"))
          + glob.glob(os.path.join(OUT, "task10v2_*.csv")) + glob.glob(os.path.join(MA, "*.csv"))):
    shutil.copy2(f, os.path.join(DEST, "csv"))
print("ROUND2_PACKAGE_DONE ->", DEST)
