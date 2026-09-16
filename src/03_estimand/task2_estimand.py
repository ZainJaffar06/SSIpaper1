#!/usr/bin/env python
"""Task 2 + 9: leak-free primary estimand and landmark serial analysis (smartphone).

The locked patient headline (0.779) averages ALL of a patient's OOF image scores,
including images taken AFTER clinical diagnosis -> future information leaks into
the primary number. Rebuild:
  (A) Leak audit: how many positive images are post-diagnosis; per-patient shift.
  (B) DETECT-BEFORE-DIAGNOSIS estimand: positives scored on strictly pre-dx images
      (POD < dx_POD; 18/20 positives have dx info, 2 excluded and reported);
      negatives use all images (they are never post-dx). Paired delta vs locked.
  (C) Truncation sensitivity: both groups truncated at POD<=14 / POD<=30.
  (D) LANDMARK analyses at t in {7,14,30}: score = mean of images with POD<=t.
      Labels: (a) eventual SSI; (b) among patients NOT yet diagnosed at t.
  (E) Landmark SERIAL model: features [last,mean,max,slope] from images POD<=t,
      grouped-OOF ridge -> the leak-free version of the Aim 6 serial result."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lib import dedup, safe_auroc, safe_auprc, grouped_oof

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "paper1")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260807)

d_full = dedup(pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv"))).reset_index(drop=True)
d = d_full.dropna(subset=["pod"]).reset_index(drop=True)   # pod-dependent analyses only
pod_missing = d_full[d_full.pod.isna()]
POD_DROP_NOTE = (f"{len(pod_missing)} images without a parseable POD "
                 f"({pod_missing.pid.nunique()} patients, all SSI-negative) excluded from POD-dependent analyses")
pos_pids = set(d.y_true.eq(1).groupby(d.pid).max().pipe(lambda s: s[s].index))
UNKNOWN_DX_POS = ["RU-A1345", "RU-A1347"]   # SSI-positive, no diagnosis date/POD anywhere

def pat_boot(y, s, nb=2000):
    y = np.asarray(y); s = np.asarray(s); idx = np.arange(len(y)); v = []
    obs = safe_auroc(y, s)
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True); a = safe_auroc(y[b], s[b])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

def paired_delta(y, a, b, nb=2000):
    y = np.asarray(y); a = np.asarray(a); b = np.asarray(b); idx = np.arange(len(y)); v = []
    obs = safe_auroc(y, a) - safe_auroc(y, b)
    for _ in range(nb):
        s = RNG.choice(idx, len(idx), True)
        da, db = safe_auroc(y[s], a[s]), safe_auroc(y[s], b[s])
        if not (np.isnan(da) or np.isnan(db)): v.append(da - db)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1)), 4)

# ---------- (A) leak audit ----------
pos = d[d.y_true == 1].copy()
pos_dx = pos.dropna(subset=["dx_pod"])
pos_dx = pos_dx.assign(post_dx=pos_dx.pod >= pos_dx.dx_pod)
audit = pos_dx.groupby("pid").agg(n_img=("pod", "size"), n_post_dx=("post_dx", "sum"),
                                  dx_pod=("dx_pod", "first")).reset_index()
audit["frac_post_dx"] = (audit.n_post_dx / audit.n_img).round(3)
audit.to_csv(os.path.join(OUT, "task2_leak_audit_per_patient.csv"), index=False)
n_post = int(pos_dx.post_dx.sum())
print(f"(A) leak audit: {n_post}/{len(pos_dx)} dx-linked positive images are ON/AFTER diagnosis "
      f"({len(audit[audit.n_post_dx>0])} of {len(audit)} patients have >=1 post-dx image)", flush=True)

# ---------- (B) detect-before-diagnosis estimand ----------
predx_pos = pos_dx[~pos_dx.post_dx]
neg = d[d.y_true == 0]
evaluable = set(predx_pos.pid)
excluded_pos = sorted(pos_pids - set(pos_dx.pid))
no_predx_img = sorted(set(pos_dx.pid) - evaluable)
ev = pd.concat([predx_pos, neg], ignore_index=True)
pat = ev.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
a, lo, hi = pat_boot(pat.y.values, pat.s.values)
lockpat = d[d.pid.isin(set(pat.pid))].groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
mm = pat.merge(lockpat, on="pid", suffixes=("_predx", "_locked"))
o, dlo, dhi, p = paired_delta(mm.y_predx.values, mm.s_predx.values, mm.s_locked.values)
res_b = pd.DataFrame([
    dict(estimand="locked_all_images (LEAKY: sees post-dx)", n_pts=d_full.pid.nunique(),
         n_pos=int(d_full.groupby('pid').y_true.max().sum()), patient_auroc=round(float(safe_auroc(
             d_full.groupby('pid').y_true.max(), d_full.groupby('pid').proba.mean())), 4), lo=np.nan, hi=np.nan),
    dict(estimand="detect_before_diagnosis (pre-dx images only)", n_pts=len(pat),
         n_pos=int(pat.y.sum()), patient_auroc=a, lo=lo, hi=hi),
    dict(estimand=f"paired delta (pre-dx minus locked, same {len(mm)} pts)", n_pts=len(mm),
         n_pos=int(mm.y_predx.sum()), patient_auroc=o, lo=dlo, hi=dhi),
])
res_b["note"] = ["primary as locked 2026-08-07 (full 210-pt frame)",
                 f"excluded: {excluded_pos} (no dx date) + {no_predx_img} (no pre-dx image); {POD_DROP_NOTE}",
                 f"p={p} (two-sided, paired patient bootstrap)"]
res_b.to_csv(os.path.join(OUT, "task2_estimand_primary.csv"), index=False)
print("(B) detect-before-diagnosis:", res_b.patient_auroc.tolist(), flush=True)

# ---------- (C) truncation sensitivity ----------
rows_c = []
for tmax in (14, 30):
    sub = ev[ev.pod <= tmax]
    pt = sub.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
    a2, lo2, hi2 = pat_boot(pt.y.values, pt.s.values)
    rows_c.append(dict(truncation=f"images POD<={tmax}", n_pts=len(pt), n_pos=int(pt.y.sum()),
                       patient_auroc=a2, lo=lo2, hi=hi2))
pd.DataFrame(rows_c).to_csv(os.path.join(OUT, "task2_truncation_sensitivity.csv"), index=False)

# ---------- (D) landmark single-image ----------
rows_d = []
for t in (7, 14, 30):
    upto = d[d.pod <= t]
    pt = upto.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
    a3, lo3, hi3 = pat_boot(pt.y.values, pt.s.values)
    rows_d.append(dict(landmark=t, label="eventual_SSI", n_pts=len(pt), n_pos=int(pt.y.sum()),
                       patient_auroc=a3, lo=lo3, hi=hi3))
    dxmap = d.dropna(subset=["dx_pod"]).groupby("pid").dx_pod.first()
    diagnosed_by_t = set(dxmap[dxmap <= t].index)
    # positives with UNKNOWN dx timing cannot be classified as (not-)yet-diagnosed: exclude (primary)
    pt2 = pt[~pt.pid.isin(diagnosed_by_t) & ~pt.pid.isin(UNKNOWN_DX_POS)]
    a4, lo4, hi4 = pat_boot(pt2.y.values, pt2.s.values)
    rows_d.append(dict(landmark=t, label="not_yet_diagnosed_at_t (unknown-dx pos excluded)",
                       n_pts=len(pt2), n_pos=int(pt2.y.sum()), patient_auroc=a4, lo=lo4, hi=hi4))
    # sensitivity: retain the 2 unknown-dx positives as not-yet-diagnosed (previous behavior, disclosed)
    pt3 = pt[~pt.pid.isin(diagnosed_by_t)]
    a5s, lo5s, hi5s = pat_boot(pt3.y.values, pt3.s.values)
    rows_d.append(dict(landmark=t, label="not_yet_diagnosed_at_t (SENSITIVITY: unknown-dx pos retained)",
                       n_pts=len(pt3), n_pos=int(pt3.y.sum()), patient_auroc=a5s, lo=lo5s, hi=hi5s))
pd.DataFrame(rows_d).to_csv(os.path.join(OUT, "task2_landmark_single.csv"), index=False)
print("(D) landmark:", [(r['landmark'], r['label'], r['patient_auroc'], r['n_pos']) for r in rows_d], flush=True)

# ---------- (E) landmark serial (leak-free Aim 6) ----------
rows_e = []
for t in (7, 14, 30):
    upto = d[d.pod <= t].sort_values(["pid", "pod"])
    feats = []
    for pid, g in upto.groupby("pid"):
        p2 = g.proba.values; pods = g.pod.values
        slope = np.polyfit(pods, p2, 1)[0] if len(p2) >= 2 and np.ptp(pods) > 0 else 0.0
        feats.append(dict(pid=pid, y=int(g.y_true.max()), last=p2[-1], mean=p2.mean(),
                          mx=p2.max(), slope=slope))
    tf = pd.DataFrame(feats)
    X = tf[["last", "mean", "mx", "slope"]].values
    oof = grouped_oof(X, tf.y.values, tf.pid.values, C=1.0)
    a5, lo5, hi5 = pat_boot(tf.y.values, oof)
    am, lom, him = pat_boot(tf.y.values, tf["mean"].values)
    o2, dlo2, dhi2, p2v = paired_delta(tf.y.values, oof, tf["mean"].values)
    rows_e.append(dict(landmark=t, n_pts=len(tf), n_pos=int(tf.y.sum()),
                       serial_model_auroc=a5, serial_lo=lo5, serial_hi=hi5,
                       single_mean_auroc=am, mean_lo=lom, mean_hi=him,
                       serial_minus_mean=o2, delta_lo=dlo2, delta_hi=dhi2, delta_p=p2v))
pd.DataFrame(rows_e).to_csv(os.path.join(OUT, "task9_landmark_serial.csv"), index=False)
print("(E) landmark serial:", [(r['landmark'], r['serial_model_auroc'], r['single_mean_auroc'], r['delta_p']) for r in rows_e], flush=True)
print("TASK2_9_DONE")
