#!/usr/bin/env python
"""Task 6: confound probes with the future-information leak REMOVED, plus paired
CNN-vs-probe deltas (ΔAUROC/ΔAUPRC) at image and patient level.
Fixes vs the Aim 2 version: the submission-behavior probe previously used
n_total_visits / n_missing_sched (functions of the WHOLE follow-up = future info).
Here every behavior feature is PAST-ONLY at capture time: POD, n_prior_images,
days_since_previous, is_first_image. Acquisition features come from the decoded
224px tensor cache (brightness/contrast/sharpness/color/edges; sharpness at 224px
is resolution-degraded — relative ranking only, see CAVEATS)."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lib import dedup, safe_auroc, safe_auprc, grouped_oof

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "paper1")
FT = os.path.join(HERE, "out", "finetune")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260807)

d = dedup(pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv"))).reset_index(drop=True)
meta = pd.read_csv(os.path.join(FT, "sp224_meta.csv"))
X8 = np.load(os.path.join(FT, "sp224_uint8.npy"))
assert list(meta.materialized_path) == list(d.materialized_path), "cache order mismatch"

# ---------- acquisition features from tensor cache ----------
Xf = X8.astype(np.float32)
gray = Xf.mean(axis=3)
lap = (np.abs(4 * gray[:, 1:-1, 1:-1] - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1]
              - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:])).reshape(len(Xf), -1)
mx = Xf.max(axis=3); mn = Xf.min(axis=3)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
acq = np.column_stack([
    gray.reshape(len(Xf), -1).mean(1),                    # brightness
    gray.reshape(len(Xf), -1).std(1),                     # contrast
    lap.var(1),                                           # sharpness (Laplacian var)
    lap.mean(1),                                          # edge density
    sat.reshape(len(Xf), -1).mean(1),                     # saturation
    Xf[..., 0].reshape(len(Xf), -1).mean(1) - Xf[..., 2].reshape(len(Xf), -1).mean(1),  # R-B tint
])

# ---------- past-only submission behavior ----------
db = d.copy().sort_values(["pid", "pod"])
db["n_prior"] = db.groupby("pid").cumcount()
db["days_since_prev"] = db.groupby("pid").pod.diff().fillna(0)
db["is_first"] = (db.n_prior == 0).astype(float)
db = db.sort_index()
beh = db[["pod", "n_prior", "days_since_prev", "is_first"]].fillna(0).values

y = d.y_true.values.astype(int); pids = d.pid.values; cnn = d.proba.values
pod_only = d[["pod"]].fillna(d.pod.median()).values

probes = {
    "pod_only": grouped_oof(pod_only, y, pids, C=1.0),
    "acquisition": grouped_oof(acq, y, pids, C=1.0),
    "behavior_past_only": grouped_oof(beh, y, pids, C=1.0),
    "acquisition+behavior+pod": grouped_oof(np.column_stack([acq, beh]), y, pids, C=1.0),
}

def clus_boot(y_, s_, p_, stat, nb=2000):
    df = pd.DataFrame({"p": p_}); groups = df.groupby("p").indices
    ps = list(groups); obs = stat(y_, s_); v = []
    for _ in range(nb):
        pick = RNG.choice(len(ps), len(ps), True)
        idx = np.concatenate([groups[ps[i]] for i in pick])
        a = stat(y_[idx], s_[idx])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

def paired_pat(y_, a_, b_, nb=2000):
    idx = np.arange(len(y_)); v = []
    obs = safe_auroc(y_, a_) - safe_auroc(y_, b_)
    for _ in range(nb):
        s = RNG.choice(idx, len(idx), True)
        da, db_ = safe_auroc(y_[s], a_[s]), safe_auroc(y_[s], b_[s])
        if not (np.isnan(da) or np.isnan(db_)): v.append(da - db_)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1)), 4)

pat = pd.DataFrame({"pid": pids, "y": y, "cnn": cnn})
rows, drows = [], []
for name, s in probes.items():
    # orientation guard: a probe whose pooled OOF is below chance is reported as-is
    ia, ilo, ihi = clus_boot(y, s, pids, safe_auroc)
    ip, plo_, phi_ = clus_boot(y, s, pids, safe_auprc)
    pat[name] = s
    pg = pat.groupby("pid").agg(y=("y", "max"), a=("cnn", "mean"), b=(name, "mean")).reset_index()
    pa = safe_auroc(pg.y, pg.b)
    rows.append(dict(probe=name, image_auroc=ia, img_lo=ilo, img_hi=ihi,
                     image_auprc=ip, patient_auroc=round(float(pa), 4)))
    # paired CNN minus probe
    io, iol, ioh, ipv = paired_pat(y, cnn, s)         # image level (unclustered delta CI at image level is anti-conservative; keep patient level primary)
    po, pol, poh, ppv = paired_pat(pg.y.values, pg.a.values, pg.b.values)
    drows.append(dict(contrast=f"CNN_minus_{name}", image_delta=io, image_p=ipv,
                      patient_delta=po, patient_lo=pol, patient_hi=poh, patient_p=ppv))
res = pd.DataFrame(rows); dres = pd.DataFrame(drows)
res.to_csv(os.path.join(OUT, "task6_probe_performance.csv"), index=False)
dres.to_csv(os.path.join(OUT, "task6_cnn_vs_probe_paired.csv"), index=False)
print(res.to_string(index=False), flush=True)
print(dres.to_string(index=False), flush=True)
print("TASK6_DONE")
