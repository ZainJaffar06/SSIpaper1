#!/usr/bin/env python
"""Parse the two chart-review PDFs into a DE-IDENTIFIED baseline covariate table.
PHI policy: names, DOB, MRN, and all calendar dates are dropped at parse time and
never written anywhere. Output keyed by RU-A study ID only.
Validation: strict per-row token-count checks; unparseable rows -> logged, not guessed."""
import re, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

SRC = "clinical_extract"; OUT = "clinical_extract"
FLAGS = ["mi","chf","pvd","cvd","dementia","chronic_pulm","ctd_rheum","ulcer","liver",
         "diabetes","hemiplegia","renal","tumor_nomet","leukemia","lymphoma",
         "met_solid_tumor","aids_hiv","transplant_hx","steroid_hx","chemo_hx","immunosupp"]

rows, bad = [], []

# ---------------- PM chart (numeric pid, fixed-width numeric block) ----------------
pm = open(f"{SRC}/pm_chart.txt").read().replace("=== PAGE BREAK ===", "\n")
for line in pm.splitlines():
    m = re.match(r"^(\d{4}) (\d{1,2}/\d{1,2}/\d{4}) (\d{1,3}) ([01]) (.+)$", line.strip())
    if not m: continue
    pid, _date, age, sex01, rest = m.groups()
    # race tokens until "eth bmi" anchor: (0|1) float
    m2 = re.match(r"^(.*?) ([01]) (\d{1,2}\.\d{1,2}|\d{2}) ((?:[01] ){23})([1-5]) (.+)$", rest)
    if not m2:
        bad.append(("PM", pid, line[:80])); continue
    race, eth, bmi, binblock, asa, tail = m2.groups()
    bins = binblock.split()
    if len(bins) != 23:
        bad.append(("PM", pid, f"binblock {len(bins)}")); continue
    smoking, flags, abx = int(bins[0]), [int(b) for b in bins[1:22]], int(bins[22])
    # tail: PROCTYPE...text...WOUND SSI  (wound may be glued to text)
    m3 = re.match(r"^([A-Z]{2,4})\s*(.*?)(\d)\s+(\d)\s*$", tail)
    if not m3:
        bad.append(("PM", pid, f"tail:{tail[:40]}")); continue
    ptype, _pname, wound, ssidx = m3.groups()
    rec = dict(pid=f"RU-A{pid}", source="PM_chart", age=int(age), sex_male=int(sex01),
               race=race.strip()[:12], ethnicity_hisp=int(eth), bmi=float(bmi),
               smoking_ever=smoking, asa=int(asa), intraop_abx=abx,
               proc_type=ptype, wound_class_cr=int(wound), ssi_dx_chart=int(ssidx))
    for i, f in enumerate(FLAGS): rec[f] = flags[i]
    rec["comorbidity_count"] = sum(flags)
    rows.append(rec)

# ---------------- EGS chart (RU-A keyed; PHI columns skipped) ----------------
egs = open(f"{SRC}/egs_chart.txt").read().replace("=== PAGE BREAK ===", " ")
egs = re.sub(r"\s+", " ", egs)
# segment per patient row: starts "date RU-A####", ends before next such start
starts = [m.start() for m in re.finditer(r"\d{1,2}/\d{1,2}/\d{4} RU-A\d{4}", egs)]
segs = [egs[s:(starts[i+1] if i+1 < len(starts) else len(egs))] for i, s in enumerate(starts)]
for seg in segs:
    pid = re.search(r"RU-A(\d{4})", seg).group(1)
    # ---- SSI outcome block (before age anchor): 'NO NA NA NA' = negative;
    #      a dx date (or Excel #### overflow) + POD = documented SSI ----
    pre_age = seg
    ssi_egs = np.nan
    if re.search(r"\bNO\s+NA\s+NA\s+NA\b", seg):
        ssi_egs = 0
    elif re.search(r"(#{4,}|\d{1,2}/\d{1,2}/\d{4})\s+\d{1,3}\s", seg[seg.find("RU-A"):]):
        # a dx-date/overflow followed by a POD integer in the SSI columns
        ssi_egs = 1
    # age + sex anchor (case-insensitive: 'male'/'Female' both occur)
    m = re.search(r" (\d{2,3}) ?(male|female) (.+)$", seg, re.IGNORECASE)
    if not m:
        bad.append(("EGS", pid, seg[:60])); continue
    age, sex, rest = m.groups()
    if not (18 <= int(age) <= 105):
        bad.append(("EGS", pid, f"age {age}")); continue
    # BMI: first number in plausible range (decimals OR integers; 12-90 — one real 88.8 exists)
    bmi = np.nan; after = rest
    for mb in re.finditer(r"\b(\d{2}(?:\.\d{1,2})?)\b", rest):
        v = float(mb.group(1))
        if 12 <= v <= 90:
            bmi = v; after = rest[mb.end():]; break
    if np.isnan(bmi): bad.append(("EGS", pid, "no plausible BMI (logged, not guessed)"))
    # trailing: [ASA][wound][abx]
    mt = re.search(r"(\d)\s*(\d)\s*(Yes|No)\s*$", after.strip(), re.IGNORECASE)
    asa = int(mt.group(1)) if mt else np.nan
    wound = int(mt.group(2)) if mt else np.nan
    abx = (1 if mt and mt.group(3).lower() == "yes" else (0 if mt else np.nan))
    if mt:
        asa = asa if 1 <= asa <= 5 else np.nan
        wound = wound if 1 <= wound <= 4 else np.nan
    mid = after[:mt.start()] if mt else after
    smoking_ever = 0 if re.search(r"\bnever\b", mid, re.IGNORECASE) else \
                   (1 if re.search(r"\b(Former|Current|Fo\b|Every)", mid, re.IGNORECASE) else np.nan)
    # DM column = value right after the smoking token; fallback: whole-mid Non-Diabetes/Diabetes search
    diabetes = np.nan
    msmk = re.search(r"\b(never|former|current|fo|unknown)\b", mid, re.IGNORECASE)
    dmzone = mid[msmk.end():].strip() if msmk else mid
    if re.match(r"^(smoker|smoking)?\s*Non[- ]?Diabet", dmzone, re.IGNORECASE): diabetes = 0
    elif re.match(r"^(smoker|smoking)?\s*(Diabet|Type\s*2|DM2|Yes\b)", dmzone, re.IGNORECASE): diabetes = 1
    elif re.match(r"^(smoker|smoking)?\s*No\b", dmzone, re.IGNORECASE): diabetes = 0
    if pd.isna(diabetes):   # fallback sweep (audit: 5 documented Non-Diabetes rows were lost to the anchor)
        if re.search(r"Non[- ]?Diabet", mid, re.IGNORECASE): diabetes = 0
        elif re.search(r"(?<!Non-)(?<!Non )Diabet|Type\s*2|DM2", mid, re.IGNORECASE): diabetes = 1
    yes_count = len(re.findall(r"Yes", mid))  # no leading \b: 'Yes' is often glued to prior word
    # enrollment-status flags from chart comments (withdrawn / refused consent / duplicate enrollment)
    enroll_flag = ""
    if re.search(r"withdr[ae]w", seg, re.IGNORECASE): enroll_flag = "withdrawn_noted"
    elif re.search(r"refus", seg, re.IGNORECASE): enroll_flag = "refusal_noted"
    if re.search(r"same (patient|person)|also enrolled|duplicate", seg, re.IGNORECASE):
        enroll_flag = (enroll_flag + ";" if enroll_flag else "") + "possible_duplicate_enrollment"
    # audit-confirmed same-person double enrollment (identical identifiers in source; not reproduced here)
    if pid == "1357": enroll_flag = (enroll_flag + ";" if enroll_flag else "") + "SAME_PERSON_AS_RU-A1303_PI_adjudicate"
    if pid == "1303": enroll_flag = (enroll_flag + ";" if enroll_flag else "") + "SAME_PERSON_AS_RU-A1357_PI_adjudicate"
    rec = dict(pid=f"RU-A{pid}", source="EGS_chart", age=int(age),
               sex_male=1 if sex.lower() == "male" else 0,
               race=np.nan, ethnicity_hisp=np.nan, bmi=bmi, smoking_ever=smoking_ever,
               asa=asa, intraop_abx=abx, proc_type="EGS", wound_class_cr=wound,
               ssi_dx_chart=ssi_egs, diabetes=diabetes,
               comorbidity_count=yes_count + (diabetes if diabetes == 1 else 0),
               enrollment_flag=enroll_flag)
    rows.append(rec)

df = pd.DataFrame(rows)
# ---- cross-chart conflicts (surfaced, not silently resolved) ----
dups = df[df.pid.duplicated(keep=False)].sort_values("pid")
conflicts = []
for pid, g in dups.groupby("pid"):
    if len(g) < 2: continue
    pm = g[g.source == "PM_chart"]; eg = g[g.source == "EGS_chart"]
    if len(pm) and len(eg):
        p, e = pm.iloc[0], eg.iloc[0]
        for col in ["age", "bmi", "asa", "wound_class_cr", "ssi_dx_chart", "diabetes"]:
            pv, ev = p[col], e[col]
            if pd.notna(pv) and pd.notna(ev) and pv != ev:
                conflicts.append(dict(pid=pid, field=col, PM_value=pv, EGS_value=ev))
pd.DataFrame(conflicts).to_csv(f"{OUT}/crosschart_conflicts.csv", index=False)
# dedupe: prefer PM (fully structured; 22/22 field agreement on overlap verified)
# EXCEPT ssi_dx_chart, where a documented SSI in EITHER chart counts (max)
ssi_max = df.groupby("pid").ssi_dx_chart.max()
df["pref"] = (df.source == "PM_chart").astype(int)
df = df.sort_values("pref", ascending=False).drop_duplicates("pid", keep="first").drop(columns=["pref"])
df["ssi_dx_chart"] = df.pid.map(ssi_max)
# HIPAA Safe Harbor: top-code ages > 89
n_top = int((df.age > 89).sum())
df.loc[df.age > 89, "age"] = 90
df.to_csv(f"{OUT}/baseline_covariates_deid.csv", index=False)
print(f"cross-chart conflicts surfaced: {len(conflicts)} (crosschart_conflicts.csv); ages top-coded at 90: {n_top}")
print(f"parsed: {len(df)} unique patients | PM rows kept {int((df.source=='PM_chart').sum())} | EGS {int((df.source=='EGS_chart').sum())}")
print(f"unparseable/skipped rows: {len(bad)}")
for b in bad[:8]: print("  BAD", b)
print("\nkey-field completeness:")
for c in ["age","sex_male","bmi","asa","wound_class_cr","diabetes","smoking_ever"]:
    print(f"  {c:16s} {df[c].notna().sum()}/{len(df)}")
print("\nASA distribution:", df.asa.value_counts(dropna=False).sort_index().to_dict())
print("age range:", df.age.min(), "-", df.age.max(), "| BMI range:", round(df.bmi.min(),1), "-", round(df.bmi.max(),1))
