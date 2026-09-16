#!/usr/bin/env python
"""P2: one row-level raw OOF export covering every architecture x seed x masking
condition on the master map: pid, image path, outcome, fold, model, seed, condition,
image-level prediction, patient-level (mean) aggregation."""
import os, glob, warnings
warnings.filterwarnings("ignore")
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MA = os.path.join(HERE, "out", "master"); OUT = os.path.join(HERE, "out", "paper1")
frames = []
for fp in sorted(glob.glob(os.path.join(MA, "*_oof.csv"))):
    nm = os.path.basename(fp).replace("_oof.csv", "")
    arch, seed = nm.rsplit("_seed", 1)
    f = pd.read_csv(fp)
    f["model"] = arch; f["seed"] = int(seed); f["condition"] = "full_image_unmasked"
    frames.append(f.rename(columns={"proba": "prediction"}))
for fp in sorted(glob.glob(os.path.join(MA, "task8_oof_*.csv"))):
    cond = os.path.basename(fp).replace("task8_oof_", "").replace(".csv", "")
    f = pd.read_csv(fp)
    f["model"] = "efficientnet_b0"; f["seed"] = 20260807; f["condition"] = cond
    frames.append(f.rename(columns={"proba_masked": "prediction"}))
allf = pd.concat(frames, ignore_index=True)[
    ["pid", "materialized_path", "y_true", "fold", "model", "seed", "condition", "prediction"]]
pat = (allf.groupby(["model", "seed", "condition", "pid"])
       .agg(y_true=("y_true", "max"), patient_mean_prediction=("prediction", "mean")).reset_index())
allf.to_csv(os.path.join(OUT, "r3_raw_oof_image_level.csv"), index=False)
pat.to_csv(os.path.join(OUT, "r3_raw_oof_patient_level.csv"), index=False)
print(f"image-level rows: {len(allf)} | patient-level rows: {len(pat)} | "
      f"model-seed-condition combos: {allf.groupby(['model','seed','condition']).ngroups}")
print("EXPORT_DONE")
