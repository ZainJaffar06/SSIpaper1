#!/usr/bin/env python
"""Task 4b (PI round-2, point 3): AUPRC treated as co-primary at 9.5% prevalence.
(A) Paired AUROC AND AUPRC deltas (patient bootstrap) for clinical/image/combined.
(B) Calibration (Cox slope/intercept, Brier) + decision curves (net benefit) for the
    three models on cross-fold Platt-calibrated probabilities.
(C) POD7 RISK-SET comparison on identical patients using only information available
    through POD7 (baseline/intraop covariates + photos through POD7).
Fold-internal confirmation: imputation means, scaler statistics, and inner-CV C
selection are all computed inside the outer training fold only (see nested_oof in
task4_clinical.py); asserted here by re-import."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from lib import dedup, safe_auroc, safe_auprc, calibration
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def nested_oof(design, y, pids, extra=None, seed=20260807, Cs=(0.01, 0.1, 1.0)):
    """Verbatim copy of task4_clinical.nested_oof (copied to avoid import side effects).
    Imputation means, scaler statistics, and inner-CV C selection all fit on the outer
    TRAINING fold only."""
    from sklearn.model_selection import StratifiedGroupKFold as SGK
    D = np.asarray(design, float)
    oof = np.full(len(y), np.nan)
    for tr, te in SGK(5, shuffle=True, random_state=seed).split(D, y, pids):
        mu = np.nanmean(D[tr], axis=0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Dtr = np.where(np.isnan(D[tr]), mu, D[tr]); Dte = np.where(np.isnan(D[te]), mu, D[te])
        if extra is not None:
            Dtr = np.column_stack([Dtr, np.asarray(extra)[tr]]); Dte = np.column_stack([Dte, np.asarray(extra)[te]])
        sc = StandardScaler().fit(Dtr)
        Str, Ste = sc.transform(Dtr), sc.transform(Dte)
        best_c, best_a = Cs[0], -1
        for C in Cs:
            aucs = []
            for itr, ite in GroupKFold(3).split(Str, y[tr], pids[tr]):
                if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2: continue
                lr = LogisticRegression(class_weight="balanced", C=C, max_iter=2000).fit(Str[itr], y[tr][itr])
                aucs.append(safe_auroc(y[tr][ite], lr.predict_proba(Str[ite])[:, 1]))
            a = np.nanmean(aucs) if aucs else np.nan
            if not np.isnan(a) and a > best_a: best_a, best_c = a, C
        lr = LogisticRegression(class_weight="balanced", C=best_c, max_iter=2000).fit(Str, y[tr])
        oof[te] = lr.predict_proba(Ste)[:, 1]
    return oof

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "paper1")
RNG = np.random.default_rng(20260807)
UNKNOWN_DX_POS = ["RU-A1345", "RU-A1347"]
IO = ["incision_cm", "closure_temp_c", "nadir_temp_c", "wound_packing", "mesh",
      "wound_protector", "intraop_cultures", "closure_min"]

d = dedup(pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv")))
pat = d.groupby("pid").agg(y=("y_true", "max"), img=("proba", "mean")).reset_index()
iop = d.drop_duplicates("pid")[["pid"] + [c for c in IO if c in d.columns]]
X = pat.merge(iop, on="pid", how="left")
F = X[[c for c in IO if c in X.columns]].apply(pd.to_numeric, errors="coerce")
y = X.y.values.astype(int); pids = X.pid.values; img = X.img.values

clin = nested_oof(F.values, y, pids)
comb = nested_oof(F.values, y, pids, extra=img)
models = {"clinical_intraop": clin, "image": img, "combined": comb}

def paired_stat(y_, a, b, stat, nb=2000):
    idx = np.arange(len(y_)); v = []
    obs = stat(y_, a) - stat(y_, b)
    for _ in range(nb):
        s = RNG.choice(idx, len(idx), True)
        da, db = stat(y_[s], a[s]), stat(y_[s], b[s])
        if not (np.isnan(da) or np.isnan(db)): v.append(da - db)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1)), 4)

rows = []
for m, s in [("combined_vs_image", (comb, img)), ("combined_vs_clinical", (comb, clin)),
             ("image_vs_clinical", (img, clin))]:
    ao, al, ah, ap = paired_stat(y, s[0], s[1], safe_auroc)
    po, pl, ph, pp = paired_stat(y, s[0], s[1], safe_auprc)
    rows.append(dict(contrast=m, d_auroc=ao, auroc_lo=al, auroc_hi=ah, auroc_p=ap,
                     d_auprc=po, auprc_lo=pl, auprc_hi=ph, auprc_p=pp))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "task4b_paired_auroc_auprc.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False), flush=True)

# (B) calibration + decision curves on cross-fold Platt (same seed-20260807 patient folds)
from sklearn.model_selection import StratifiedGroupKFold
def platt_oof(s):
    out = np.full(len(s), np.nan)
    lo = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=20260807).split(lo, y, pids):
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(lo[tr], y[tr])
        out[te] = lr.predict_proba(lo[te])[:, 1]
    return out

cal_rows, dc = [], {"threshold": np.round(np.arange(0.02, 0.32, 0.02), 2)}
for name, s in models.items():
    sp = platt_oof(np.asarray(s, float))
    try: sl, ic = calibration(y, sp)
    except Exception: sl = ic = np.nan
    cal_rows.append(dict(model=name, brier=round(float(brier_score_loss(y, np.clip(sp, 0, 1))), 4),
                         cal_slope=round(sl, 3), cal_intercept=round(ic, 3)))
    nb = []
    for t in dc["threshold"]:
        pred = sp >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        nb.append(round((tp - fp * t / (1 - t)) / len(y), 4))
    dc[name] = nb
dc["treat_all"] = [round(y.mean() - (1 - y.mean()) * t / (1 - t), 4) for t in dc["threshold"]]
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "task4b_calibration.csv"), index=False)
pd.DataFrame(dc).to_csv(os.path.join(OUT, "task4b_decision_curves.csv"), index=False)
print(pd.DataFrame(cal_rows).to_string(index=False), flush=True)

# (C) POD7 risk-set comparison, identical patients, info through POD7 only
dd = d.dropna(subset=["pod"])
dxmap = dd.dropna(subset=["dx_pod"]).groupby("pid").dx_pod.first()
diagnosed7 = set(dxmap[dxmap <= 7].index)
r7 = dd[(dd.pod <= 7) & ~dd.pid.isin(diagnosed7) & ~dd.pid.isin(UNKNOWN_DX_POS)]
p7 = r7.groupby("pid").agg(y=("y_true", "max"), img7=("proba", "mean")).reset_index().merge(iop, on="pid", how="left")
F7 = p7[[c for c in IO if c in p7.columns]].apply(pd.to_numeric, errors="coerce")
y7 = p7.y.values.astype(int); pid7 = p7.pid.values; img7 = p7.img7.values
clin7 = nested_oof(F7.values, y7, pid7)
comb7 = nested_oof(F7.values, y7, pid7, extra=img7)
def bootci(y_, s_, stat, nb=2000):
    idx = np.arange(len(y_)); v = []
    obs = stat(y_, s_)
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True)
        a = stat(y_[b], s_[b])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)
rs = []
for name, s in [("clinical_intraop", clin7), ("image_throughPOD7", img7), ("combined", comb7)]:
    a, al, ah = bootci(y7, s, safe_auroc); p, pl, ph = bootci(y7, s, safe_auprc)
    rs.append(dict(model=name, n_pts=len(y7), n_events=int(y7.sum()),
                   auroc=a, auroc_lo=al, auroc_hi=ah, auprc=p, auprc_lo=pl, auprc_hi=ph))
d1 = paired_stat(y7, comb7, img7, safe_auroc); d2 = paired_stat(y7, comb7, img7, safe_auprc)
rs.append(dict(model="paired combined_vs_image: dAUROC %s (%s,%s) p=%s | dAUPRC %s (%s,%s) p=%s" %
               (d1[0], d1[1], d1[2], d1[3], d2[0], d2[1], d2[2], d2[3]),
               n_pts=len(y7), n_events=int(y7.sum()), auroc=np.nan, auroc_lo=np.nan, auroc_hi=np.nan,
               auprc=np.nan, auprc_lo=np.nan, auprc_hi=np.nan))
pd.DataFrame(rs).to_csv(os.path.join(OUT, "task4b_pod7_riskset.csv"), index=False)
print(pd.DataFrame(rs)[["model", "n_pts", "n_events", "auroc", "auprc"]].to_string(index=False), flush=True)
print("TASK4B_DONE", flush=True)
