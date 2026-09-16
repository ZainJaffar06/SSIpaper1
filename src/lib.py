#!/usr/bin/env python
"""Shared analysis library — Paper 1 (smartphone arm).
Reproduction gates: locked image AUROC 0.7554 / patient(mean) 0.7792 on the raw
ssi_smartphone OOF rows. All CV is StratifiedGroupKFold(5, shuffle, seed 20260807)
grouped by patient; CIs are patient-clustered bootstrap."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 20260807


def dedup(df):
    return df.drop_duplicates("materialized_path").reset_index(drop=True)


def safe_auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    m = ~(pd.isna(y) | pd.isna(s)); y, s = y[m], s[m]
    if len(np.unique(y)) < 2: return np.nan
    return roc_auc_score(y, s)


def safe_auprc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    m = ~(pd.isna(y) | pd.isna(s)); y, s = y[m], s[m]
    if len(np.unique(y)) < 2: return np.nan
    return average_precision_score(y, s)


def patient_agg(df, how="mean", score="proba", pid="pid", y="y_true", pod="pod"):
    g = df.sort_values([pid] + ([pod] if pod in df.columns else []))
    if how == "mean":
        agg = g.groupby(pid).agg(y=(y, "max"), s=(score, "mean"))
    elif how == "max":
        agg = g.groupby(pid).agg(y=(y, "max"), s=(score, "max"))
    elif how == "latest":
        agg = g.groupby(pid).agg(y=(y, "max"), s=(score, "last"))
    else:
        raise ValueError(how)
    return agg.reset_index()


def cluster_boot_ci(pids, y, s, stat=safe_auroc, nb=2000, seed=SEED):
    """Patient-clustered bootstrap: resample patients with replacement, pool their rows."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"pid": pids, "y": y, "s": s})
    groups = {p: g for p, g in df.groupby("pid")}
    ps = list(groups)
    obs = stat(df.y.values, df.s.values)
    vals = []
    for _ in range(nb):
        pick = rng.choice(ps, len(ps), replace=True)
        bb = pd.concat([groups[p] for p in pick], ignore_index=True)
        v = stat(bb.y.values, bb.s.values)
        if not np.isnan(v): vals.append(v)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return float(obs), float(lo), float(hi)


def calibration(y, p, eps=1e-7):
    """Cox recalibration: slope+intercept of logistic fit on logit(p)."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(lo, np.asarray(y).astype(int))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def operating_points(y, s, specs=(0.80, 0.90, 0.95)):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    neg = np.sort(s[y == 0]); rows = []
    for sp in specs:
        thr = np.quantile(neg, sp)
        sens = float((s[y == 1] > thr).mean())
        spec = float((s[y == 0] <= thr).mean())
        ppv_n = (s > thr)
        ppv = float(y[ppv_n].mean()) if ppv_n.sum() else np.nan
        rows.append(dict(target_spec=sp, threshold=float(thr), sensitivity=sens,
                         specificity=spec, ppv=ppv, n_flagged=int(ppv_n.sum())))
    return pd.DataFrame(rows)


def grouped_oof(X, y, groups, C=1.0, n_splits=5, seed=SEED):
    """Patient-grouped out-of-fold logistic predictions (scaled, balanced)."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    X = np.where(np.isfinite(X), X, np.nan)
    oof = np.full(len(y), np.nan)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups):
        mu = np.nanmean(X[tr], axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Xtr = np.where(np.isnan(X[tr]), mu, X[tr])
        Xte = np.where(np.isnan(X[te]), mu, X[te])
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(class_weight="balanced", C=C, max_iter=2000)
        lr.fit(sc.transform(Xtr), y[tr])
        oof[te] = lr.predict_proba(sc.transform(Xte))[:, 1]
    return oof
