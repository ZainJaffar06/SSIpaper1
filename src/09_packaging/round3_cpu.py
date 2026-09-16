#!/usr/bin/env python
"""Round-3 CPU items.
B3  Background-only confound battery: probes (POD / acquisition / past-only behavior /
    combined) evaluated against the background-only, wound-only and full-image models
    on the SAME 1,886-image annotated subset with MASTER patient folds: paired
    patient-level deltas (model minus probe) + Spearman correlation of patient scores.
S1  Recalibration + operating points for the POD7 risk-set primary (193 pts / 15 events),
    incl. a rule-out point (threshold at ~93% sensitivity) with flag rate.
S2  Patient-bootstrap 95% CIs for Brier / calibration slope+intercept, and decision-curve
    confidence bands, for clinical / image / combined.
S3  Task 10 pre-diagnosis restriction: CNN-vs-consensus on pre-dx images only.
P1  Masking cohort audit: per-image exclusion list (62 images / 4 patients) with reasons.
P3  Methods confirmation statement."""
import os, json, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from lib import dedup, safe_auroc, safe_auprc, calibration

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1"); MA = os.path.join(HERE, "out", "master")
FT = os.path.join(HERE, "out", "finetune")
RNG = np.random.default_rng(20260807)

master = pd.read_csv(os.path.join(OUT, "MASTER_map_1948_seed42.csv"))
X8 = np.load(os.path.join(FT, "sp224_uint8.npy"))
base = dedup(pd.read_csv(os.path.join(HERE, "base", "ssi_smartphone.csv")))
assert list(base.materialized_path) == list(master.materialized_path)

def pat_frame(df, score, pid="pid"):
    return df.groupby(pid).agg(y=("y_true", "max"), s=(score, "mean")).reset_index()

def boot(y, s, stat=safe_auroc, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = stat(y, s)
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True)
        a = stat(y[b], s[b])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

def paired(y, a, b, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = safe_auroc(y, a) - safe_auroc(y, b)
    for _ in range(nb):
        s = RNG.choice(idx, len(idx), True)
        da, db = safe_auroc(y[s], a[s]), safe_auroc(y[s], b[s])
        if not (np.isnan(da) or np.isnan(db)): v.append(da - db)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1)), 4)

# ---------------- B3: probe battery vs masked models ----------------
Xf = X8.astype(np.float32)
gray = Xf.mean(axis=3)
lap = (np.abs(4 * gray[:, 1:-1, 1:-1] - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1]
              - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:])).reshape(len(Xf), -1)
mx = Xf.max(axis=3); mn = Xf.min(axis=3)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
acq = np.column_stack([gray.reshape(len(Xf), -1).mean(1), gray.reshape(len(Xf), -1).std(1),
                       lap.var(1), lap.mean(1), sat.reshape(len(Xf), -1).mean(1),
                       Xf[..., 0].reshape(len(Xf), -1).mean(1) - Xf[..., 2].reshape(len(Xf), -1).mean(1)])
db = base.copy().sort_values(["pid", "pod"])
db["n_prior"] = db.groupby("pid").cumcount()
db["days_since_prev"] = db.groupby("pid").pod.diff().fillna(0)
db["is_first"] = (db.n_prior == 0).astype(float)
db = db.sort_index()
beh = db[["pod", "n_prior", "days_since_prev", "is_first"]].fillna(0).values
podf = base[["pod"]].fillna(base.pod.median()).values
y_all = master.y_true.values.astype(int); folds = master.fold.values; pids = master.pid.values

def master_oof_probe(X):
    X = np.asarray(X, float); oof = np.full(len(y_all), np.nan)
    for k in sorted(np.unique(folds)):
        tr, te = folds != k, folds == k
        mu = np.nanmean(X[tr], axis=0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Xtr = np.where(np.isnan(X[tr]), mu, X[tr]); Xte = np.where(np.isnan(X[te]), mu, X[te])
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(sc.transform(Xtr), y_all[tr])
        oof[te] = lr.predict_proba(sc.transform(Xte))[:, 1]
    return oof

probes = {"pod_only": master_oof_probe(podf), "acquisition": master_oof_probe(acq),
          "behavior_past_only": master_oof_probe(beh),
          "acq+beh+pod": master_oof_probe(np.column_stack([acq, beh]))}
mdl_frames = {}
for cond in ("full_image", "wound_only", "background_only"):
    f = pd.read_csv(os.path.join(MA, f"task8_oof_{cond}.csv"))
    mdl_frames[cond] = f
sub_paths = mdl_frames["full_image"].materialized_path
sub_idx = master.reset_index().set_index("materialized_path").loc[sub_paths, "index"].values
rows = []
from scipy.stats import spearmanr
for cond, f in mdl_frames.items():
    pm = pat_frame(f.rename(columns={"proba_masked": "s"}), "s")
    for pname, po in probes.items():
        fp = f.copy(); fp["probe"] = po[sub_idx]
        pp = fp.groupby("pid").agg(y=("y_true", "max"), sp=("probe", "mean"),
                                   sm=("proba_masked", "mean")).reset_index()
        o, lo, hi, p = paired(pp.y.values, pp.sm.values, pp.sp.values)
        rho = spearmanr(pp.sm, pp.sp).statistic
        rows.append(dict(model=cond, probe=pname,
                         probe_patient_auroc=round(float(safe_auroc(pp.y, pp.sp)), 4),
                         model_patient_auroc=round(float(safe_auroc(pp.y, pp.sm)), 4),
                         model_minus_probe=o, lo=lo, hi=hi, p_two_sided=p,
                         spearman_model_vs_probe=round(float(rho), 3)))
bat = pd.DataFrame(rows)
bat.to_csv(os.path.join(OUT, "r3_background_probe_battery.csv"), index=False)
print(bat[["model", "probe", "model_minus_probe", "p_two_sided", "spearman_model_vs_probe"]].to_string(index=False), flush=True)

# ---------------- S1: POD7 risk-set recalibration + operating points ----------------
dd = base.dropna(subset=["pod"])
dxmap = dd.dropna(subset=["dx_pod"]).groupby("pid").dx_pod.first()
r7 = dd[(dd.pod <= 7) & ~dd.pid.isin(set(dxmap[dxmap <= 7].index)) & ~dd.pid.isin(["RU-A1345", "RU-A1347"])]
p7 = r7.groupby("pid").agg(y=("y_true", "max"), s=("proba", "mean")).reset_index()
p7 = p7.merge(master.drop_duplicates("pid")[["pid", "fold"]], on="pid")
y7 = p7.y.values.astype(int); s7 = p7.s.values.astype(float); f7 = p7.fold.values
platt = np.full(len(y7), np.nan)
lo7 = np.log(np.clip(s7, 1e-7, 1 - 1e-7) / (1 - np.clip(s7, 1e-7, 1 - 1e-7))).reshape(-1, 1)
for k in sorted(np.unique(f7)):
    tr, te = f7 != k, f7 == k
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(lo7[tr], y7[tr])
    platt[te] = lr.predict_proba(lo7[te])[:, 1]
def slope_stat(y_, s_):
    try: return calibration(y_, s_)[0]
    except Exception: return np.nan
def icpt_stat(y_, s_):
    try: return calibration(y_, s_)[1]
    except Exception: return np.nan
au = boot(y7, s7); br = boot(y7, platt, lambda a, b: brier_score_loss(a, np.clip(b, 0, 1)))
sl = boot(y7, platt, slope_stat); ic = boot(y7, platt, icpt_stat)
cal7 = pd.DataFrame([dict(metric="patient_auroc(raw)", value=au[0], lo=au[1], hi=au[2]),
                     dict(metric="brier(platt)", value=br[0], lo=br[1], hi=br[2]),
                     dict(metric="cal_slope(platt)", value=sl[0], lo=sl[1], hi=sl[2]),
                     dict(metric="cal_intercept(platt)", value=ic[0], lo=ic[1], hi=ic[2])])
cal7.to_csv(os.path.join(OUT, "r3_pod7_calibration.csv"), index=False)
ops = []
neg = np.sort(s7[y7 == 0])
for sp in (0.80, 0.90):
    thr = np.quantile(neg, sp)
    ops.append(dict(rule=f"spec {sp:.2f}", threshold=round(float(thr), 4),
                    sensitivity=round(float((s7[y7 == 1] > thr).mean()), 3),
                    flag_rate=round(float((s7 > thr).mean()), 3)))
pos_sorted = np.sort(s7[y7 == 1])
for target, nmiss in (("rule-out sens 14/15 (0.93)", 1), ("rule-out sens 15/15 (1.00)", 0)):
    thr = pos_sorted[nmiss] - 1e-9 if nmiss < len(pos_sorted) else pos_sorted[0] - 1e-9
    ops.append(dict(rule=target, threshold=round(float(thr), 4),
                    sensitivity=round(float((s7[y7 == 1] > thr).mean()), 3),
                    flag_rate=round(float((s7 > thr).mean()), 3)))
for o in ops:
    thr = o["threshold"]; v = []
    for _ in range(2000):
        b = RNG.choice(len(y7), len(y7), True)
        if y7[b].sum() == 0: continue
        v.append((s7[b][y7[b] == 1] > thr).mean())
    o["sens_lo"], o["sens_hi"] = round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)
    o["specificity"] = round(float((s7[y7 == 0] <= thr).mean()), 3)
ops7 = pd.DataFrame(ops)
ops7.to_csv(os.path.join(OUT, "r3_pod7_operating_points.csv"), index=False)
print(ops7.to_string(index=False), flush=True)

# ---------------- S2: CIs for Brier/calibration + decision-curve bands ----------------
t4bcal = []
p210 = base.groupby("pid").agg(y=("y_true", "max"), img=("proba", "mean")).reset_index()
from task2_estimand import __name__ as _  # noqa (no side effects; placeholder)
clin_oof = None
# reuse saved decision-curve inputs: recompute model scores exactly as task4b did
IO = ["incision_cm", "closure_temp_c", "nadir_temp_c", "wound_packing", "mesh",
      "wound_protector", "intraop_cultures", "closure_min"]
iop = base.drop_duplicates("pid")[["pid"] + [c for c in IO if c in base.columns]]
X4 = p210.merge(iop, on="pid", how="left")
F4 = X4[[c for c in IO if c in X4.columns]].apply(pd.to_numeric, errors="coerce")
y4 = X4.y.values.astype(int); pid4 = X4.pid.values; img4 = X4.img.values
import importlib.util as _il
spec = _il.spec_from_file_location("t4b", os.path.join(HERE, "task4b_prospective.py"))
# avoid re-running task4b: reimplement nested_oof inline (identical)
def nested_oof(design, yv, pv, extra=None, seed=20260807, Cs=(0.01, 0.1, 1.0)):
    from sklearn.model_selection import StratifiedGroupKFold as SGK, GroupKFold as GK
    D = np.asarray(design, float); oof = np.full(len(yv), np.nan)
    for tr, te in SGK(5, shuffle=True, random_state=seed).split(D, yv, pv):
        mu = np.nanmean(D[tr], axis=0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Dtr = np.where(np.isnan(D[tr]), mu, D[tr]); Dte = np.where(np.isnan(D[te]), mu, D[te])
        if extra is not None:
            Dtr = np.column_stack([Dtr, np.asarray(extra)[tr]]); Dte = np.column_stack([Dte, np.asarray(extra)[te]])
        sc = StandardScaler().fit(Dtr); Str, Ste = sc.transform(Dtr), sc.transform(Dte)
        best_c, best_a = Cs[0], -1
        for C in Cs:
            aucs = []
            for itr, ite in GK(3).split(Str, yv[tr], pv[tr]):
                if len(np.unique(yv[tr][itr])) < 2 or len(np.unique(yv[tr][ite])) < 2: continue
                lr = LogisticRegression(class_weight="balanced", C=C, max_iter=2000).fit(Str[itr], yv[tr][itr])
                aucs.append(safe_auroc(yv[tr][ite], lr.predict_proba(Str[ite])[:, 1]))
            a = np.nanmean(aucs) if aucs else np.nan
            if not np.isnan(a) and a > best_a: best_a, best_c = a, C
        lr = LogisticRegression(class_weight="balanced", C=best_c, max_iter=2000).fit(Str, yv[tr])
        oof[te] = lr.predict_proba(Ste)[:, 1]
    return oof
clin4 = nested_oof(F4.values, y4, pid4); comb4 = nested_oof(F4.values, y4, pid4, extra=img4)
def platt210(s):
    from sklearn.model_selection import StratifiedGroupKFold as SGK
    out = np.full(len(s), np.nan)
    l = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    for tr, te in SGK(5, shuffle=True, random_state=20260807).split(l, y4, pid4):
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(l[tr], y4[tr])
        out[te] = lr.predict_proba(l[te])[:, 1]
    return out
thr_grid = np.round(np.arange(0.02, 0.32, 0.02), 2)
dc_rows = {"threshold": thr_grid}
for name, s in [("clinical_intraop", clin4), ("image", img4), ("combined", comb4)]:
    sp = platt210(np.asarray(s, float))
    br = boot(y4, sp, lambda a, b: brier_score_loss(a, np.clip(b, 0, 1)))
    sl = boot(y4, sp, slope_stat); ic = boot(y4, sp, icpt_stat)
    t4bcal.append(dict(model=name, brier=br[0], brier_lo=br[1], brier_hi=br[2],
                       cal_slope=sl[0], slope_lo=sl[1], slope_hi=sl[2],
                       cal_intercept=ic[0], int_lo=ic[1], int_hi=ic[2]))
    nb_obs, nb_lo, nb_hi = [], [], []
    for t in thr_grid:
        def netben(yv, sv):
            pred = sv >= t
            tp = float((pred & (yv == 1)).sum()); fp = float((pred & (yv == 0)).sum())
            return (tp - fp * t / (1 - t)) / len(yv)
        o = netben(y4, sp); v = []
        for _ in range(1000):
            b = RNG.choice(len(y4), len(y4), True)
            v.append(netben(y4[b], sp[b]))
        nb_obs.append(round(o, 4)); nb_lo.append(round(float(np.percentile(v, 2.5)), 4))
        nb_hi.append(round(float(np.percentile(v, 97.5)), 4))
    dc_rows[name] = nb_obs; dc_rows[name + "_lo"] = nb_lo; dc_rows[name + "_hi"] = nb_hi
pd.DataFrame(t4bcal).to_csv(os.path.join(OUT, "r3_calibration_with_cis.csv"), index=False)
pd.DataFrame(dc_rows).to_csv(os.path.join(OUT, "r3_decision_curves_with_cis.csv"), index=False)
print(pd.DataFrame(t4bcal).to_string(index=False), flush=True)

# ---------------- S3: task10 pre-dx restriction ----------------
i10 = pd.read_csv(os.path.join(OUT, "task10v2_image_level.csv"))
m10 = i10[i10.cnn_score.notna() & i10.y_adj.notna()].copy()
m10 = m10.merge(dxmap.rename("dx_pod"), left_on="pid", right_index=True, how="left")
pre = m10[(m10.y_adj == 0) | (m10.pod < m10.dx_pod)]
def clus(y_, s_, p_, nb=2000):
    df = pd.DataFrame({"p": p_}); groups = df.groupby("p").indices
    ps = list(groups); obs = safe_auroc(y_, s_); v = []
    for _ in range(nb):
        pick = RNG.choice(len(ps), len(ps), True)
        idx = np.concatenate([groups[ps[i]] for i in pick])
        a = safe_auroc(y_[idx], s_[idx])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)
yp = pre.y_adj.astype(int).values; pp_ = pre.pid.values
ca = clus(yp, pre.consensus_frac.values, pp_); na = clus(yp, pre.cnn_score.values, pp_)
dfp = pd.DataFrame({"p": pp_}); groups = dfp.groupby("p").indices; ps = list(groups); v = []
obs = safe_auroc(yp, pre.cnn_score.values) - safe_auroc(yp, pre.consensus_frac.values)
for _ in range(2000):
    pick = RNG.choice(len(ps), len(ps), True)
    idx = np.concatenate([groups[ps[i]] for i in pick])
    da, db_ = safe_auroc(yp[idx], pre.cnn_score.values[idx]), safe_auroc(yp[idx], pre.consensus_frac.values[idx])
    if not (np.isnan(da) or np.isnan(db_)): v.append(da - db_)
pv = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
s3 = pd.DataFrame([dict(analysis="pre-dx restricted CNN-vs-consensus", n_img=len(pre),
                        n_pos_img=int(yp.sum()), n_pos_pts=pre[pre.y_adj == 1].pid.nunique(),
                        n_dropped_postdx=int(len(m10) - len(pre)),
                        consensus_auroc=ca[0], consensus_lo=ca[1], consensus_hi=ca[2],
                        cnn_auroc=na[0], cnn_lo=na[1], cnn_hi=na[2],
                        delta=round(float(obs), 4), delta_lo=round(float(np.percentile(v, 2.5)), 4),
                        delta_hi=round(float(np.percentile(v, 97.5)), 4), p=round(float(min(pv, 1)), 4))])
s3.to_csv(os.path.join(OUT, "r3_task10_predx.csv"), index=False)
print(s3.to_string(index=False), flush=True)

# ---------------- P1: masking cohort audit ----------------
MAN = "/Volumes/Backup Plus/SSI_CNN_PUBLISHABLE_2026_08_07/manifests/locked_primary_manifest.csv"
man = pd.read_csv(MAN, encoding="utf-8-sig", low_memory=False).drop_duplicates("materialized_path")
mm = master.merge(man[["materialized_path", "best_source_type", "best_source_locator",
                       "best_source_member"]], on="materialized_path", how="left")
included = set(mdl_frames["full_image"].materialized_path)
exc = mm[~mm.materialized_path.isin(included)].copy()
INCISION_LABELS = ("incision_line", "incision_line_center", "incision_line_left",
                   "incision_line_right", "wound_bed")
reasons = []
for _, r in exc.iterrows():
    if r.best_source_type != "annotation_zip_embedded":
        reasons.append("no embedded annotation source (best_source_type=%s)" % r.best_source_type)
        continue
    try:
        with zipfile.ZipFile(r.best_source_locator) as zf:
            data = json.loads(zf.read(r.best_source_member))
        shapes = data.get("shapes") or []
        labs = [str(s.get("label", "")).strip().lower() for s in shapes]
        if not shapes: reasons.append("annotation JSON has no shapes")
        elif not any(l in INCISION_LABELS for l in labs):
            reasons.append("annotation has no incision-target shapes (labels: %s)" % ",".join(sorted(set(labs))[:4]))
        else: reasons.append("incision shapes present but mask empty/invalid")
    except Exception as e:
        reasons.append("annotation unreadable: %s" % type(e).__name__)
exc["exclusion_reason"] = reasons
audit = exc[["pid", "materialized_path", "y_true", "fold", "best_source_type", "exclusion_reason"]]
audit.to_csv(os.path.join(OUT, "r3_masking_cohort_audit.csv"), index=False)
print(f"P1: {len(audit)} excluded images, {audit.pid.nunique()} patients "
      f"({audit[audit.y_true==1].pid.nunique()} SSI+); reasons: {pd.Series(reasons).value_counts().head(3).to_dict()}", flush=True)
print("ROUND3_CPU_DONE", flush=True)
