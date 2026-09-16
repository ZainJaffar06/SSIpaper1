#!/usr/bin/env python
"""Reproducibility addendum (PI request 2026-09-13). Canonical files:
1. clinical_final_rangechecked_summary.csv + clinical_final_patient_oof.csv (full cohort)
2. clinical_pod7_patient_oof.csv (POD-7 risk-set equivalents)
3. primary_pod7_patient_scores.csv + primary_pod7_canonical_metrics.csv
   (one dedicated RNG -> ONE canonical CI; earlier 0.645/0.647/0.650 lower limits were
   bootstrap-stream artifacts of shared-RNG consumption order across scripts)
4. healing_stage_oof_predictions.csv + summary (supports the ~0.975 positive control)
5. sham_mask_overlap_audit.csv + summary (IoU between true and relocated sham masks,
   reproducing the exact RNG stream used in the sham retraining)
All patient-level files: pid, y_true, fold, scores."""
import os, json, zipfile, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lib import dedup, safe_auroc, safe_auprc

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.expanduser("~/Documents/New project/outputs/paper1_repro_addendum_2026-09-13")
os.makedirs(DEST, exist_ok=True)
IO = ["incision_cm","closure_temp_c","nadir_temp_c","wound_packing","mesh","wound_protector","intraop_cultures","closure_min"]
UNKNOWN_DX_POS = ["RU-A1345","RU-A1347"]

base = dedup(pd.read_csv(os.path.join(HERE, "base", "ssi_smartphone.csv")))
master = pd.read_csv(os.path.join(HERE, "out", "paper1", "MASTER_map_1948_seed42.csv"))
foldmap = master.drop_duplicates("pid")[["pid","fold"]]

def nested_oof(D, y, p, extra=None, seed=20260807, Cs=(0.01,0.1,1.0)):
    D = np.asarray(D, float); oof = np.full(len(y), np.nan); foldout = np.full(len(y), -1)
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(D, y, p)):
        mu = np.nanmean(D[tr], axis=0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Dtr = np.where(np.isnan(D[tr]), mu, D[tr]); Dte = np.where(np.isnan(D[te]), mu, D[te])
        if extra is not None:
            Dtr = np.column_stack([Dtr, np.asarray(extra)[tr]]); Dte = np.column_stack([Dte, np.asarray(extra)[te]])
        sc = StandardScaler().fit(Dtr); Str, Ste = sc.transform(Dtr), sc.transform(Dte)
        bc, ba = Cs[0], -1
        for C in Cs:
            au = []
            for itr, ite in GroupKFold(3).split(Str, y[tr], p[tr]):
                if len(np.unique(y[tr][itr]))<2 or len(np.unique(y[tr][ite]))<2: continue
                lr = LogisticRegression(class_weight="balanced", C=C, max_iter=2000).fit(Str[itr], y[tr][itr])
                au.append(safe_auroc(y[tr][ite], lr.predict_proba(Str[ite])[:,1]))
            a = np.nanmean(au) if au else np.nan
            if not np.isnan(a) and a > ba: ba, bc = a, C
        lr = LogisticRegression(class_weight="balanced", C=bc, max_iter=2000).fit(Str, y[tr])
        oof[te] = lr.predict_proba(Ste)[:,1]; foldout[te] = k
    return oof, foldout

def boot(rng, y, s, stat=safe_auroc, nb=2000):
    idx = np.arange(len(y)); obs = stat(y, s); v = []
    for _ in range(nb):
        b = rng.choice(idx, len(idx), True)
        a = stat(y[b], s[b])
        if not np.isnan(a): v.append(a)
    return round(float(obs),4), round(float(np.percentile(v,2.5)),4), round(float(np.percentile(v,97.5)),4)

def paired(rng, y, a, b, nb=2000, stat=safe_auroc):
    idx = np.arange(len(y)); obs = stat(y, a) - stat(y, b); v = []
    for _ in range(nb):
        s = rng.choice(idx, len(idx), True)
        da, db = stat(y[s], a[s]), stat(y[s], b[s])
        if not (np.isnan(da) or np.isnan(db)): v.append(da - db)
    p = 2*min(np.mean(np.array(v)<=0), np.mean(np.array(v)>=0))
    return round(float(obs),4), round(float(np.percentile(v,2.5)),4), round(float(np.percentile(v,97.5)),4), round(float(min(p,1)),4)

# ============ 1+2. FINAL RANGE-CHECKED CLINICAL: full cohort ============
pat = base.groupby("pid").agg(y_true=("y_true","max"), image_score=("proba","mean")).reset_index()
iop = base.drop_duplicates("pid")[["pid"]+IO]
X = pat.merge(iop, on="pid", how="left")
F = X[IO].apply(pd.to_numeric, errors="coerce")
n_removed = 0
for c in ("closure_temp_c","nadir_temp_c"):
    bad = (F[c]<30)|(F[c]>43); n_removed += int(bad.sum()); F.loc[bad, c] = np.nan
y = X.y_true.values.astype(int); pids = X.pid.values; img = X.image_score.values
clin, foldA = nested_oof(F.values, y, pids)
comb, _ = nested_oof(F.values, y, pids, extra=img)
oofA = pd.DataFrame(dict(pid=pids, y_true=y, fold=foldA, clinical_score=np.round(clin,6),
                         image_score=np.round(img,6), combined_score=np.round(comb,6)))
oofA.to_csv(os.path.join(DEST, "clinical_final_patient_oof.csv"), index=False)

rng = np.random.default_rng(913)   # dedicated, documented canonical RNG for this package
rows = []
for name, s in [("clinical_rangechecked", clin), ("image_locked_mean", img), ("combined", comb)]:
    a, lo, hi = boot(rng, y, s)
    ap, aplo, aphi = boot(rng, y, s, stat=safe_auprc)
    rows.append(dict(model=name, n_pts=len(y), n_events=int(y.sum()), auroc=a, auroc_lo=lo, auroc_hi=hi,
                     auprc=ap, auprc_lo=aplo, auprc_hi=aphi))
drows = []
for nm, a_, b_ in [("combined_minus_clinical", comb, clin), ("combined_minus_image", comb, img),
                   ("image_minus_clinical", img, clin)]:
    o, lo, hi, p = paired(rng, y, a_, b_)
    op, lop, hip, pp = paired(rng, y, a_, b_, stat=safe_auprc)
    drows.append(dict(contrast=nm, d_auroc=o, auroc_lo=lo, auroc_hi=hi, auroc_p=p,
                      d_auprc=op, auprc_lo=lop, auprc_hi=hip, auprc_p=pp))
pd.DataFrame(rows).to_csv(os.path.join(DEST, "clinical_final_rangechecked_summary.csv"), index=False)
pd.DataFrame(drows).to_csv(os.path.join(DEST, "clinical_final_rangechecked_deltas.csv"), index=False)
print("FULL-COHORT:", [(r["model"], r["auroc"]) for r in rows], "| removed temp entries:", n_removed)
print("  deltas:", [(r["contrast"], r["d_auroc"], r["auroc_p"]) for r in drows])

# ============ 3. CANONICAL POD-7 primary scores + POD-7 clinical equivalents ============
dd = base.dropna(subset=["pod"])
dxmap = dd.dropna(subset=["dx_pod"]).groupby("pid").dx_pod.first()
r7 = dd[(dd.pod<=7) & ~dd.pid.isin(set(dxmap[dxmap<=7].index)) & ~dd.pid.isin(UNKNOWN_DX_POS)].sort_values(["pid","pod"])
g = r7.groupby("pid")
p7 = g.agg(y_true=("y_true","max"), n_images_thru_pod7=("proba","size"),
           score_mean=("proba","mean"), score_latest=("proba","last")).reset_index()
closest = (r7.assign(gap=(7 - r7.pod)).sort_values(["pid","gap"]).groupby("pid").proba.first().rename("score_closest"))
p7 = p7.merge(closest, on="pid").merge(foldmap, on="pid", how="left")
p7[["pid","y_true","fold","n_images_thru_pod7","score_mean","score_latest","score_closest"]].round(6)\
  .to_csv(os.path.join(DEST, "primary_pod7_patient_scores.csv"), index=False)
y7 = p7.y_true.values.astype(int); s7 = p7.score_mean.values
a, lo, hi = boot(rng, y7, s7)
ap, aplo, aphi = boot(rng, y7, s7, stat=safe_auprc)
al, _, _ = boot(rng, y7, p7.score_latest.values); ac, _, _ = boot(rng, y7, p7.score_closest.values)
pd.DataFrame([dict(metric="patient_auroc_mean_agg", value=a, lo=lo, hi=hi),
              dict(metric="patient_auprc_mean_agg", value=ap, lo=aplo, hi=aphi),
              dict(metric="patient_auroc_latest_agg", value=al, lo=np.nan, hi=np.nan),
              dict(metric="patient_auroc_closest_agg", value=ac, lo=np.nan, hi=np.nan),
              dict(metric="n_pts", value=len(y7), lo=np.nan, hi=np.nan),
              dict(metric="n_events", value=int(y7.sum()), lo=np.nan, hi=np.nan)])\
  .to_csv(os.path.join(DEST, "primary_pod7_canonical_metrics.csv"), index=False)
print("POD7 CANONICAL: AUROC", a, f"({lo}-{hi})", "AUPRC", ap, "| latest", al, "closest", ac)

# POD-7 clinical/image/combined OOF (identical patients, info through POD 7)
X7 = p7[["pid","y_true","score_mean"]].merge(iop, on="pid", how="left")
F7 = X7[IO].apply(pd.to_numeric, errors="coerce")
for c in ("closure_temp_c","nadir_temp_c"):
    F7.loc[(F7[c]<30)|(F7[c]>43), c] = np.nan
img7 = X7.score_mean.values; y7v = X7.y_true.values.astype(int); pid7 = X7.pid.values
clin7, foldB = nested_oof(F7.values, y7v, pid7)
comb7, _ = nested_oof(F7.values, y7v, pid7, extra=img7)
pd.DataFrame(dict(pid=pid7, y_true=y7v, fold=foldB, clinical_score=np.round(clin7,6),
                  image_score_thru_pod7=np.round(img7,6), combined_score=np.round(comb7,6)))\
  .to_csv(os.path.join(DEST, "clinical_pod7_patient_oof.csv"), index=False)
o7, l7, h7, pp7 = paired(rng, y7v, comb7, img7)
print("POD7 clinical:", round(safe_auroc(y7v, clin7),4), "combined:", round(safe_auroc(y7v, comb7),4),
      "| comb-vs-img", o7, "p", pp7)

# ============ 4. Healing-stage OOF ============
stage = pd.read_csv(os.path.join(HERE, "base", "stage_smartphone.csv"))
stage_out = stage[["pid","materialized_path","phase_norm","y_true","proba"]].copy()
stage_out.to_csv(os.path.join(DEST, "healing_stage_oof_predictions.csv"), index=False)
sp = stage.groupby("pid").agg(y=("y_true","max"), s=("proba","mean")).reset_index()
si = round(float(safe_auroc(stage.y_true, stage.proba)),4)
spa = round(float(safe_auroc(sp.y, sp.s)),4)
pd.DataFrame([dict(level="image", n=len(stage), auroc=si),
              dict(level="patient_mean_agg", n=len(sp), auroc=spa)])\
  .to_csv(os.path.join(DEST, "healing_stage_summary.csv"), index=False)
print("HEALING STAGE: image", si, "patient", spa)

# ============ 5. Sham-mask overlap audit ============
MAN = "/Volumes/Backup Plus/SSI_CNN_PUBLISHABLE_2026_08_07/manifests/locked_primary_manifest.csv"
man = pd.read_csv(MAN, encoding="utf-8-sig", low_memory=False).drop_duplicates("materialized_path")
mm = master.merge(man[["materialized_path","best_source_type","best_source_locator","best_source_member"]],
                  on="materialized_path", how="left")
LAB = ("incision_line","incision_line_center","incision_line_left","incision_line_right","wound_bed")
def mask224(row):
    try:
        if row.best_source_type != "annotation_zip_embedded": return None
        with zipfile.ZipFile(row.best_source_locator) as zf:
            data = json.loads(zf.read(row.best_source_member))
        W, H = data.get("imageWidth"), data.get("imageHeight"); shapes = data.get("shapes") or []
        if not shapes or not W or not H: return None
        m = Image.new("L",(224,224),0); dr = ImageDraw.Draw(m); drew = False
        for s in shapes:
            if str(s.get("label","")).strip().lower() not in LAB: continue
            pts = s.get("points") or []
            if len(pts) < 2: continue
            sc = [(p[0]*224.0/W, p[1]*224.0/H) for p in pts]
            if s.get("shape_type") in ("polygon",) and len(sc) >= 3: dr.polygon(sc, fill=1)
            else: dr.line(sc, fill=1, width=10)
            drew = True
        if not drew: return None
        a = np.array(m, dtype=np.uint8)
        return (np.array(Image.fromarray(a*255).filter(ImageFilter.MaxFilter(11)))>0) if a.sum() else None
    except Exception:
        return None
masks, keep = {}, []
for i, row in mm.iterrows():
    mk = mask224(row)
    if mk is not None: masks[i] = mk; keep.append(i)
keep = np.array(keep)
# EXACT replication of the sham RNG stream used in round3_compute.py
rng_sham = np.random.default_rng(20260807)
rows5 = []
for i in keep:
    dy, dx = int(rng_sham.integers(56,168)), int(rng_sham.integers(56,168))
    t = masks[i]; s = np.roll(t, (dy, dx), axis=(0,1))
    inter = (t & s).sum(); union = (t | s).sum()
    rows5.append(dict(materialized_path=mm.loc[i,"materialized_path"], pid=mm.loc[i,"pid"],
                      mask_area_px=int(t.sum()), shift_y=dy, shift_x=dx,
                      iou=round(float(inter/union),4) if union else 0.0,
                      overlap_frac_of_true=round(float(inter/t.sum()),4) if t.sum() else 0.0))
ov = pd.DataFrame(rows5)
ov.to_csv(os.path.join(DEST, "sham_mask_overlap_audit.csv"), index=False)
summ = dict(n_images=len(ov), median_iou=round(float(ov.iou.median()),4),
            mean_iou=round(float(ov.iou.mean()),4), p90_iou=round(float(ov.iou.quantile(0.9)),4),
            max_iou=round(float(ov.iou.max()),4),
            frac_iou_lt_0_05=round(float((ov.iou<0.05).mean()),4),
            frac_iou_lt_0_20=round(float((ov.iou<0.20).mean()),4),
            median_overlap_frac_of_true=round(float(ov.overlap_frac_of_true.median()),4))
pd.DataFrame([summ]).to_csv(os.path.join(DEST, "sham_mask_overlap_summary.csv"), index=False)
print("SHAM OVERLAP:", summ)
print("ADDENDUM_DONE ->", DEST)
