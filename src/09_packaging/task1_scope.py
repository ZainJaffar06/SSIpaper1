#!/usr/bin/env python
"""Task 1 memo + three-subset table.
(1) Adjudication impact: for each PI-flagged patient (double-enrollment /
withdrawal-note flags from the chart-review audit), how many images they
contribute to each modeling arm. Result documents that the Paper-1 smartphone
cohort is ADJUDICATION-STABLE (zero images from any flagged patient).
(2) Three-subset patient AUROC table with n / prevalence / CI, so the paper
reports one primary number with its cohort definition explicit."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lib import dedup, safe_auroc

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "paper1")
RNG = np.random.default_rng(20260807)

FLAGGED = {
    "RU-A1357": "double-enrolled, same person as RU-A1303 (opposite labels); withdrawal note",
    "RU-A1303": "the RETAINED enrollment of the double-enrolled patient (SSI-positive)",
    "RU-A1051": "withdrawal/refusal note in chart review",
    "RU-A1327": "withdrawal/refusal note in chart review",
    "RU-A1355": "withdrawal/refusal note in chart review",
}
ARMS = ["ssi_smartphone", "ssi_smartphone_expanded", "ssi_both"]
frames = {a: dedup(pd.read_csv(os.path.join(BASE, f"{a}.csv"))) for a in ARMS}

rows = []
for pid, why in FLAGGED.items():
    r = dict(pid=pid, flag=why)
    for a in ARMS:
        f = frames[a]
        r[f"n_img_{a}"] = int((f.pid == pid).sum())
    rows.append(r)
memo = pd.DataFrame(rows)
memo.to_csv(os.path.join(OUT, "task1_adjudication_impact.csv"), index=False)
print(memo.to_string(index=False), flush=True)
sp_cols = [c for c in memo.columns if c.startswith("n_img_ssi_smartphone")]
stable = (memo[sp_cols].sum().sum() == 0)
print(f"smartphone arms adjudication-stable: {stable}", flush=True)

def boot(y, s, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = safe_auroc(y, s)
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True)
        a = safe_auroc(y[b], s[b])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

srows = []
LABELS = {
    "ssi_smartphone": "PRIMARY: locked curated smartphone",
    "ssi_smartphone_expanded": "sensitivity: expanded/uncurated smartphone",
    "ssi_both": "context only (NOT Paper 1): smartphone+thermal patients",
}
for a in ARMS:
    f = frames[a]
    pat = f.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
    au, lo, hi = boot(pat.y.values, pat.s.values)
    srows.append(dict(subset=LABELS[a], n_images=len(f), n_patients=len(pat),
                      n_pos=int(pat.y.sum()), prevalence=round(float(pat.y.mean()), 4),
                      patient_auroc=au, lo=lo, hi=hi))
tbl = pd.DataFrame(srows)
tbl.to_csv(os.path.join(OUT, "task1_three_subset_table.csv"), index=False)
print(tbl.to_string(index=False), flush=True)
print("TASK1_DONE")
