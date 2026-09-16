#!/usr/bin/env python
"""Task 7: cross-fold recalibration of the locked smartphone OOF + full metric CIs.
Uses the VERIFIED locked fold map (task3, seed-42 reconstruction): for each fold k,
fit Platt (logistic on logit p) and isotonic calibrators on the OTHER folds' OOF
predictions, apply to fold k -> honestly out-of-fold calibrated probabilities.
Patient level is primary (score = mean of image probs; folds are patient-disjoint).
Reports AUROC / AUPRC / Brier / calibration slope+intercept / sens@spec operating
points, all with patient(-clustered) bootstrap 95% CIs, before + after calibration."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from lib import dedup, safe_auroc, safe_auprc, calibration, operating_points

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "paper1")
RNG = np.random.default_rng(20260807)

d = dedup(pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv")))
fm = pd.read_csv(os.path.join(OUT, "task3_lockedfoldmap_ssi_smartphone.csv")).drop_duplicates("materialized_path")
d = d.merge(fm[["materialized_path", "fold"]], on="materialized_path", how="left")
assert d.fold.notna().all()

# patient-level frame (fold constant within patient — folds are patient-disjoint)
pat = d.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean"), fold=("fold", "first")).reset_index()
assert len(pat) == 210 and pat.y.sum() == 20
EPS = 1e-7

def platt_iso_oof(df, score_col="s"):
    out_p = np.full(len(df), np.nan); out_i = np.full(len(df), np.nan)
    s = np.clip(df[score_col].values.astype(float), EPS, 1 - EPS)
    lo = np.log(s / (1 - s)); y = df.y.values.astype(int)
    for k in sorted(df.fold.unique()):
        tr = df.fold.values != k; te = ~tr
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(lo[tr].reshape(-1, 1), y[tr])
        out_p[te] = lr.predict_proba(lo[te].reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(s[tr], y[tr])
        out_i[te] = iso.predict(s[te])
    return out_p, out_i

pat["s_platt"], pat["s_iso"] = platt_iso_oof(pat)

def pat_boot_stat(y, s, stat, nb=2000):
    idx = np.arange(len(y)); obs = stat(y, s); v = []
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True)
        a = stat(y[b], s[b])
        if a is not None and not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

def slope_only(y, s):
    try: return calibration(y, s)[0]
    except Exception: return np.nan

def intercept_only(y, s):
    try: return calibration(y, s)[1]
    except Exception: return np.nan

rows = []
for name, col in [("raw", "s"), ("platt", "s_platt"), ("isotonic", "s_iso")]:
    y = pat.y.values.astype(int); s = pat[col].values.astype(float)
    au, aulo, auhi = pat_boot_stat(y, s, safe_auroc)
    ap, aplo, aphi = pat_boot_stat(y, s, safe_auprc)
    br, brlo, brhi = pat_boot_stat(y, s, lambda a, b: brier_score_loss(a, np.clip(b, 0, 1)))
    if name == "isotonic":
        # DEGENERATE at 20 events: 57/210 patients sit at exact 0/1 (27 unique values), so the
        # Cox slope is an artifact of the logit clip constant. Suppressed (audit finding).
        sl = sllo = slhi = ic = iclo = ichi = np.nan
    else:
        sl, sllo, slhi = pat_boot_stat(y, s, slope_only)
        ic, iclo, ichi = pat_boot_stat(y, s, intercept_only)
    rows.append(dict(scores=name, patient_auroc=au, auroc_lo=aulo, auroc_hi=auhi,
                     patient_auprc=ap, auprc_lo=aplo, auprc_hi=aphi,
                     brier=br, brier_lo=brlo, brier_hi=brhi,
                     cal_slope=sl, slope_lo=sllo, slope_hi=slhi,
                     cal_intercept=ic, int_lo=iclo, int_hi=ichi,
                     note="slope/intercept suppressed: degenerate (eps-dependent) at 20 events" if name == "isotonic" else ""))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "task7_calibration_metrics.csv"), index=False)
print(res[["scores", "patient_auroc", "patient_auprc", "brier", "cal_slope", "cal_intercept"]].to_string(index=False), flush=True)

# operating points on the RAW locked patient score (PRIMARY — monotone-invariant, so
# identical under any single deployed calibrator). Pooled cross-fold Platt reorders
# patients across folds and understates sens/PPV (audit finding); kept as labeled secondary.
ci_rows = []
for scale, col in [("raw_locked (PRIMARY)", "s"),
                   ("pooled_crossfold_platt (secondary; cross-fold pooling artifact)", "s_platt")]:
    y = pat.y.values.astype(int); s = pat[col].values.astype(float)
    ops = operating_points(y, s)
    for _, r in ops.iterrows():
        sp = r.target_spec; v_sens = []
        idx = np.arange(len(y))
        for _ in range(2000):
            b = RNG.choice(idx, len(idx), True)
            yb, sb = y[b], s[b]
            if yb.sum() == 0 or (yb == 0).sum() == 0: continue
            thr = np.quantile(sb[yb == 0], sp)
            v_sens.append((sb[yb == 1] > thr).mean())
        lo, hi = np.percentile(v_sens, [2.5, 97.5])
        ci_rows.append(dict(score_scale=scale, target_spec=sp, threshold=round(r.threshold, 4),
                            sensitivity=round(r.sensitivity, 4), sens_lo=round(float(lo), 4),
                            sens_hi=round(float(hi), 4), achieved_spec=round(r.specificity, 4),
                            ppv=round(r.ppv, 4), n_flagged=int(r.n_flagged)))
opsci = pd.DataFrame(ci_rows)
opsci.to_csv(os.path.join(OUT, "task7_operating_points.csv"), index=False)
print(opsci.to_string(index=False), flush=True)
print("TASK7_DONE")
