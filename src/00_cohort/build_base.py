#!/usr/bin/env python
"""Build the shared analysis base for Paper 1: locked OOF predictions joined with
POD (from phase_norm), chart-review diagnosis timing, and intraop covariates.
Gates: ssi_smartphone must be 1979 rows / 210 pts / 20 pos pts and reproduce
image AUROC 0.7554 / patient(mean) 0.7792 on raw OOF rows, else abort."""
import os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lib import safe_auroc, patient_agg, dedup

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); os.makedirs(BASE, exist_ok=True)
MOD = "/Volumes/Backup Plus/SSI_CNN_PUBLISHABLE_2026_08_07/modeling_2026-08-07"
CHART = "/Volumes/Backup Plus/SSI chart review.xlsx"
INTRAOP = os.path.expanduser("~/Downloads/ProjectAiCCESSIntraO_DATA_LABELS_2026-07-01_1325.csv")


def pod_from_phase(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    m = re.search(r"d(\d+)", s)
    if m: return float(m.group(1))
    if "pre" in s or "post" in s: return 0.0
    return np.nan


# ---- chart review: dx POD + dx date ----
ch = pd.read_excel(CHART)
ch.columns = [c.strip().lower().replace(" ", "_") for c in ch.columns]
idc = [c for c in ch.columns if "patient" in c][0]
podc = [c for c in ch.columns if c == "pod" or "pod" in c][0]
dtc = [c for c in ch.columns if "diagnosis_date" in c or "date" in c][0]
chart = pd.DataFrame({
    "pid": ch[idc].astype(str).str.strip(),
    "dx_pod": pd.to_numeric(ch[podc], errors="coerce"),
    "dx_date": pd.to_datetime(ch[dtc], errors="coerce"),
}).dropna(subset=["pid"]).drop_duplicates("pid")

# ---- intraop covariates ----
cov = None
if os.path.exists(INTRAOP):
    io = pd.read_csv(INTRAOP)
    io.columns = [c.strip() for c in io.columns]
    pidc = [c for c in io.columns if "Patient ID" in c or c == "Record ID"][0]

    def col(sub):
        c = [c for c in io.columns if sub.lower() in c.lower()]
        return io[c[0]] if c else pd.Series(np.nan, index=io.index)

    cov = pd.DataFrame({"pid": io[pidc].astype(str).str.strip()})
    cov["cdc_wound_class"] = col("wound class").astype(str)
    cov["wound_type"] = col("What type of wound").astype(str)
    cov["incision_cm"] = pd.to_numeric(col("length of incision"), errors="coerce")
    cov["closure_temp_c"] = pd.to_numeric(col("of patient at wound closure"), errors="coerce")
    cov["nadir_temp_c"] = pd.to_numeric(col("lowest temperature"), errors="coerce")
    for name, sub in [("wound_packing", "packing"), ("mesh", "mesh"),
                      ("wound_protector", "protector"), ("intraop_cultures", "cultures")]:
        cov[name] = col(sub).astype(str).str.lower().str.startswith("yes").astype(float)
    def hhmm(s):
        s = s.astype(str).str.strip().str.replace(":", "", regex=False)
        s = s.where(s.str.fullmatch(r"\d{3,4}"), np.nan)
        s = s.str.zfill(4)
        return pd.to_numeric(s.str[:2], errors="coerce") * 60 + pd.to_numeric(s.str[2:], errors="coerce")
    m0, m1 = hhmm(col("Time closure began")), hhmm(col("Time closure ended"))
    mins = m1 - m0
    mins = mins.where(mins >= 0, mins + 1440)   # overnight wrap
    cov["closure_min"] = np.where((mins >= 0) & (mins < 600), mins, np.nan)
    cov = cov.drop_duplicates("pid")

ARMS = ["ssi_smartphone", "ssi_both", "stage_smartphone", "ssi_smartphone_expanded"]
for arm in ARMS:
    fp = os.path.join(MOD, f"{arm}_oof_predictions.csv")
    if not os.path.exists(fp):
        print(f"[{arm}] MISSING OOF CSV", flush=True); continue
    d = pd.read_csv(fp, encoding="utf-8-sig")
    d["pid"] = d["pid"].astype(str).str.strip()
    d["pod"] = d["phase_norm"].map(pod_from_phase)
    d = d.merge(chart, on="pid", how="left")
    if cov is not None:
        d = d.merge(cov, on="pid", how="left")
    d.to_csv(os.path.join(BASE, f"{arm}.csv"), index=False)
    ycol = "y_true"
    npos = d[d[ycol] == 1].pid.nunique()
    print(f"[{arm}] {len(d)} imgs | {d.pid.nunique()} pts | pos {npos} pts", flush=True)

# ---- reproduction gates on the primary arm ----
d = pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv"))
assert len(d) == 1979 and d.pid.nunique() == 210 and d[d.y_true == 1].pid.nunique() == 20
dd = dedup(d)   # locked headline is defined on the deduplicated frame (1,948 unique images)
img_auc = safe_auroc(dd.y_true, dd.proba)
pat = patient_agg(dd, "mean")
pat_auc = safe_auroc(pat.y, pat.s)
print(f"GATES: n_unique {len(dd)} (want 1948)  image {img_auc:.4f} (want 0.7554)  patient {pat_auc:.4f} (want 0.7792)", flush=True)
assert len(dd) == 1948 and abs(img_auc - 0.7554) < 5e-4 and abs(pat_auc - 0.7792) < 5e-4
# dx coverage among smartphone positives
pos = sorted(d[d.y_true == 1].pid.unique())
have = [p for p in pos if p in set(chart.dropna(subset=["dx_pod"]).pid)]
print(f"positives with dx_pod: {len(have)}/20; missing: {sorted(set(pos)-set(have))}", flush=True)
print("BASE_OK", flush=True)
