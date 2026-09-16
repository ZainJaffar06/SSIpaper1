#!/usr/bin/env python
"""Task 2 v2 (PI round-2, point 2): the PRIMARY clinical estimand is the POD7 RISK-SET
analysis — among patients NOT diagnosed with SSI by POD7 (positives with unknown dx date
excluded), predict subsequent SSI from RGB information available through POD7.
Aggregation is stated and varied: mean of all photos through POD7 (primary),
latest photo through POD7, photo closest to POD7 (sensitivities).
POD14/POD30 are secondary/descriptive (9 and 2 remaining events).
The pre-diagnosis truncation analysis is retained as a leak-robustness SENSITIVITY,
cited only as the paired same-204-patient delta."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lib import dedup, safe_auroc, safe_auprc

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "paper1")
RNG = np.random.default_rng(20260807)
UNKNOWN_DX_POS = ["RU-A1345", "RU-A1347"]

d = dedup(pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv")))
d = d.dropna(subset=["pod"]).reset_index(drop=True)

def boot(y, s, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = safe_auroc(y, s)
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True)
        a = safe_auroc(y[b], s[b])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

dxmap = d.dropna(subset=["dx_pod"]).groupby("pid").dx_pod.first()
rows = []
for t, tier in [(7, "PRIMARY"), (14, "secondary (9 events)"), (30, "descriptive only (2 events)")]:
    upto = d[d.pod <= t].sort_values(["pid", "pod"])
    diagnosed_by_t = set(dxmap[dxmap <= t].index)
    risk = upto[~upto.pid.isin(diagnosed_by_t) & ~upto.pid.isin(UNKNOWN_DX_POS)]
    aggs = {
        "mean of all photos through POD%d" % t: risk.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")),
        "latest photo through POD%d" % t: risk.groupby("pid").agg(y=("y_true", "max"), s=("proba", "last")),
        "photo closest to POD%d" % t: (risk.assign(gap=(t - risk.pod))
                                       .sort_values(["pid", "gap"]).groupby("pid")
                                       .agg(y=("y_true", "max"), s=("proba", "first"))),
    }
    for name, pt in aggs.items():
        a, lo, hi = boot(pt.y.values, pt.s.values)
        rows.append(dict(tier=tier if "mean" in name else tier + " / aggregation sensitivity",
                         landmark=t, aggregation=name, n_pts=len(pt), n_events=int(pt.y.sum()),
                         auroc=a, lo=lo, hi=hi,
                         auprc=round(float(safe_auprc(pt.y.values, pt.s.values)), 4)))
        if "mean" not in name and t > 7:
            rows.pop()  # aggregation sensitivities only for the primary POD7 landmark
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "task2v2_riskset_primary.csv"), index=False)
print(res[["tier", "landmark", "aggregation", "n_pts", "n_events", "auroc", "lo", "hi"]].to_string(index=False), flush=True)

# leak-robustness sensitivity: paired same-patient delta only (wording per reconciliation #5)
pos_dx = d[(d.y_true == 1)].dropna(subset=["dx_pod"])
predx = pd.concat([pos_dx[pos_dx.pod < pos_dx.dx_pod], d[d.y_true == 0]], ignore_index=True)
p1 = predx.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
lock_same = d[d.pid.isin(set(p1.pid))].groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
mm = p1.merge(lock_same, on="pid", suffixes=("_predx", "_lock"))
idx = np.arange(len(mm)); v = []
obs = safe_auroc(mm.y_predx, mm.s_predx) - safe_auroc(mm.y_predx, mm.s_lock)
for _ in range(2000):
    b = RNG.choice(idx, len(idx), True)
    da, db = safe_auroc(mm.y_predx.values[b], mm.s_predx.values[b]), safe_auroc(mm.y_predx.values[b], mm.s_lock.values[b])
    if not (np.isnan(da) or np.isnan(db)): v.append(da - db)
p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
sens = pd.DataFrame([dict(
    analysis="leak-robustness sensitivity (pre-dx truncation)", n_pts=len(mm), n_events=int(mm.y_predx.sum()),
    paired_delta=round(float(obs), 4), lo=round(float(np.percentile(v, 2.5)), 4),
    hi=round(float(np.percentile(v, 97.5)), 4), p_two_sided=round(float(min(p, 1)), 4),
    note="cite ONLY this paired same-patient delta; controls have no analogous censoring time, "
         "so this is a sensitivity, not the primary estimand")])
sens.to_csv(os.path.join(OUT, "task2v2_leak_sensitivity.csv"), index=False)
print(sens.to_string(index=False), flush=True)
print("TASK2V2_DONE", flush=True)
