#!/usr/bin/env python
"""PI round-2, point 1: establish ONE deduplicated 1,948-image master file with ONE
frozen patient fold map = the VERIFIED locked seed-42 StratifiedGroupKFold assignment.
Every architecture (ResNet18, EffB0, ViT) and the masked conditions retrain on exactly
this map, making all comparisons fold-paired. Duplicates note: the 30 duplicated paths
in the locked 1,979-row frame are within-patient repeats (never cross folds or
patients — verified), so deduplication changes sample weighting only, not partition."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1")

lk = pd.read_csv(os.path.join(OUT, "task3_lockedfoldmap_ssi_smartphone.csv"))
assert len(lk) == 1979
dup = lk[lk.duplicated("materialized_path", keep=False)]
assert (dup.groupby("materialized_path").fold.nunique() > 1).sum() == 0
assert (dup.groupby("materialized_path").pid.nunique() > 1).sum() == 0
master = lk.drop_duplicates("materialized_path").reset_index(drop=True)
assert len(master) == 1948 and master.pid.nunique() == 210
for k in sorted(master.fold.unique()):
    assert len(set(master.pid[master.fold == k]) & set(master.pid[master.fold != k])) == 0
# patient fold constant within patient
assert (master.groupby("pid").fold.nunique() == 1).all()
master.to_csv(os.path.join(OUT, "MASTER_map_1948_seed42.csv"), index=False)
pos = master[master.y_true == 1].pid.nunique()
print(f"MASTER frozen: {len(master)} images | {master.pid.nunique()} pts | {pos} SSI+ | "
      f"fold sizes {master.fold.value_counts().sort_index().tolist()}")
print("MASTER_OK")
