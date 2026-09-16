#!/usr/bin/env python
"""Task 3a: reconstruct + verify + publish the LOCKED ResNet18 fold assignments.
cnn_locked_arms.py used StratifiedGroupKFold(5, shuffle, random_state=42) on the
arm frame (row order = OOF CSV order). Verification: fold-0 test indices must
match the saved ssi_smartphone_fold0_test_idx.npy EXACTLY, else we refuse to
publish a reconstruction and say so."""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1"); os.makedirs(OUT, exist_ok=True)
MOD = "/Volumes/Backup Plus/SSI_CNN_PUBLISHABLE_2026_08_07/modeling_2026-08-07"

rows = []
for arm in ["ssi_smartphone", "ssi_both", "ssi_smartphone_expanded", "stage_smartphone"]:
    fp = os.path.join(MOD, f"{arm}_oof_predictions.csv")
    if not os.path.exists(fp):
        rows.append(dict(arm=arm, status="no OOF csv")); continue
    d = pd.read_csv(fp, encoding="utf-8-sig")
    y = d.y_true.astype(int).values; g = d.pid.values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.full(len(d), -1)
    for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
        folds[te] = k
    # verify against saved fold-0 test indices where available
    npy = os.path.join(MOD, f"{arm}_fold0_test_idx.npy")
    status = "no fold0 npy to verify"
    if os.path.exists(npy):
        saved = np.load(npy)
        recon = np.where(folds == 0)[0]
        status = ("VERIFIED exact" if len(saved) == len(recon) and np.array_equal(np.sort(saved), np.sort(recon))
                  else f"MISMATCH (saved {len(saved)} vs recon {len(recon)})")
    rows.append(dict(arm=arm, n_rows=len(d), n_pts=d.pid.nunique(), status=status,
                     fold_sizes=";".join(str(int((folds == k).sum())) for k in range(5))))
    if "VERIFIED" in status or "no fold0" in status:
        out = d[["pid", "materialized_path", "y_true"]].copy(); out["fold"] = folds
        # patient-disjointness
        for k in range(5):
            assert len(set(g[folds == k]) & set(g[folds != k])) == 0, f"{arm} fold overlap!"
        out.to_csv(os.path.join(OUT, f"task3_lockedfoldmap_{arm}.csv"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "task3_fold_reconstruction_report.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("TASK3_DONE")
