#!/usr/bin/env python
"""Task 4: full-cohort clinical covariates on the LOCKED smartphone cohort (210 pts).
Union of chart-review baseline (parse_baseline output) + intraop covariates, with
missingness indicators and fold-internal imputation; nested-C logistic ("ridge");
clinical vs image vs combined with paired patient-bootstrap deltas.
Note: image patient score is the locked CNN OOF mean (CNN folds seed 42); the
clinical/combined folds here are seed 20260807 — combination is post-hoc (CAVEAT)."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lib import dedup, safe_auroc, safe_auprc

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "paper1")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260807)

d = dedup(pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv")))
img = d.groupby("pid").agg(y=("y_true", "max"), img_score=("proba", "mean")).reset_index()
assert len(img) == 210 and img.y.sum() == 20

chart = pd.read_csv(os.path.join(HERE, "clinical_extract", "baseline_covariates_deid.csv"))
CH_FEATS = ["age", "sex_male", "bmi", "asa", "wound_class_cr", "diabetes", "smoking_ever",
            "mi", "chf", "renal", "chronic_pulm", "steroid_hx", "immunosupp", "intraop_abx"]
CH_FEATS = [c for c in CH_FEATS if c in chart.columns]
io_cols = ["incision_cm", "closure_temp_c", "nadir_temp_c", "wound_packing", "mesh",
           "wound_protector", "intraop_cultures", "closure_min"]
iop = d.drop_duplicates("pid")[["pid"] + [c for c in io_cols if c in d.columns]]

X = img.merge(chart[["pid"] + CH_FEATS], on="pid", how="left").merge(iop, on="pid", how="left")
feat_cols = CH_FEATS + [c for c in io_cols if c in X.columns]
cov_from_chart = X[CH_FEATS].notna().any(axis=1)
cov_from_io = X[[c for c in io_cols if c in X.columns]].notna().any(axis=1)
coverage = pd.DataFrame([dict(
    n_locked=len(X), n_pos=int(X.y.sum()),
    chart_covered=int(cov_from_chart.sum()), chart_pos=int(X[cov_from_chart].y.sum()),
    intraop_covered=int(cov_from_io.sum()), intraop_pos=int(X[cov_from_io].y.sum()),
    union_covered=int((cov_from_chart | cov_from_io).sum()),
    union_pos=int(X[cov_from_chart | cov_from_io].y.sum()),
    neither=int((~(cov_from_chart | cov_from_io)).sum()))])
coverage.to_csv(os.path.join(OUT, "task4_coverage.csv"), index=False)
print(coverage.to_string(index=False), flush=True)

# AUDIT FINDING (blocker): chart-review coverage is OUTCOME-CORRELATED (18/20 positives
# covered vs 77/190 negatives), so missingness indicators leak ascertainment, not clinical
# signal. Primary design: intraop-only covariates (coverage ~outcome-neutral), NO indicators.
# Union-values design kept without indicators (partial mitigation, disclosed); the old
# union+indicators model is retained only as a labeled leak diagnostic.
F_union = X[feat_cols].apply(pd.to_numeric, errors="coerce")
io_feats = [c for c in io_cols if c in X.columns]
F_intraop = X[io_feats].apply(pd.to_numeric, errors="coerce")
Mset = F_union.isna().astype(float); Mset.columns = [c + "_miss" for c in F_union.columns]
y = X.y.values.astype(int); pids = X.pid.values

# ascertainment audit
cov_chart = cov_from_chart.astype(int).values
asc = pd.DataFrame([
    dict(probe="chart_coverage_indicator_alone", auroc=round(float(safe_auroc(y, cov_chart)), 4),
         note=f"covered {int(cov_chart.sum())}/210; positives covered {int(cov_chart[y==1].sum())}/20"),
    dict(probe="intraop_coverage_indicator_alone", auroc=round(float(safe_auroc(y, cov_from_io.astype(int).values)), 4),
         note=f"covered {int(cov_from_io.sum())}/210; positives covered {int(cov_from_io.astype(int).values[y==1].sum())}/20"),
])
asc.to_csv(os.path.join(OUT, "task4_ascertainment_audit.csv"), index=False)
print(asc.to_string(index=False), flush=True)


def nested_oof(design, y, pids, extra=None, seed=20260807, Cs=(0.01, 0.1, 1.0)):
    """Outer StratifiedGroupKFold(5); inner GroupKFold(3) on train picks C; fold-internal
    imputation+scaling. extra: additional column(s) appended AFTER imputation (e.g. img score)."""
    D = np.asarray(design, float)
    oof = np.full(len(y), np.nan)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in outer.split(D, y, pids):
        mu = np.nanmean(D[tr], axis=0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Dtr = np.where(np.isnan(D[tr]), mu, D[tr]); Dte = np.where(np.isnan(D[te]), mu, D[te])
        if extra is not None:
            Dtr = np.column_stack([Dtr, extra[tr]]); Dte = np.column_stack([Dte, extra[te]])
        sc = StandardScaler().fit(Dtr)
        Str, Ste = sc.transform(Dtr), sc.transform(Dte)
        best_c, best_a = Cs[0], -1
        inner = GroupKFold(3)
        for C in Cs:
            aucs = []
            for itr, ite in inner.split(Str, y[tr], pids[tr]):
                if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2: continue
                lr = LogisticRegression(class_weight="balanced", C=C, max_iter=2000).fit(Str[itr], y[tr][itr])
                aucs.append(safe_auroc(y[tr][ite], lr.predict_proba(Str[ite])[:, 1]))
            a = np.nanmean(aucs) if aucs else np.nan
            if not np.isnan(a) and a > best_a: best_a, best_c = a, C
        lr = LogisticRegression(class_weight="balanced", C=best_c, max_iter=2000).fit(Str, y[tr])
        oof[te] = lr.predict_proba(Ste)[:, 1]
    return oof


def boot_ci(y, s, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = safe_auroc(y, s)
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True)
        a = safe_auroc(y[b], s[b])
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


imgS = X.img_score.values
clin_io = nested_oof(F_intraop.values, y, pids)
clin_union = nested_oof(F_union.values, y, pids)
clin_leaky = nested_oof(pd.concat([F_union, Mset], axis=1).values, y, pids)
comb = nested_oof(F_intraop.values, y, pids, extra=imgS)
clin = clin_io   # primary clinical model for paired deltas
rows = []
for name, s in [("clinical_intraop_only_PRIMARY(210pts)", clin_io),
                ("clinical_union_values_no_indicators (ascertainment caveat)", clin_union),
                ("clinical_union+missing_indicators (LEAKY diagnostic - do not cite)", clin_leaky),
                ("image_only_locked_mean", imgS),
                ("combined_intraop+image", comb)]:
    a, lo, hi = boot_ci(y, s)
    rows.append(dict(model=name, n_pts=len(y), n_pos=int(y.sum()), patient_auroc=a, lo=lo, hi=hi,
                     auprc=round(float(safe_auprc(y, s)), 4)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "task4_clinical_image_combined.csv"), index=False)
print(res.to_string(index=False), flush=True)

drows = []
for name, a, b in [("combined_minus_clinical_intraop", comb, clin), ("combined_minus_image", comb, imgS),
                   ("image_minus_clinical_intraop", imgS, clin)]:
    o, lo, hi, p = paired(y, a, b)
    drows.append(dict(contrast=name, delta_auroc=o, lo=lo, hi=hi, p_two_sided=p))
dres = pd.DataFrame(drows)
dres.to_csv(os.path.join(OUT, "task4_paired_deltas.csv"), index=False)
print(dres.to_string(index=False), flush=True)
print("TASK4_DONE")
