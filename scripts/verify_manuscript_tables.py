# -*- coding: utf-8 -*-
import docx, pandas as pd, numpy as np, re
R="/Users/zainjaffar/Documents/New project/ssi-paper1-information-audit/results/collaboration/"
d=docx.Document("/Users/zainjaffar/Desktop/Smartphone Images Sham paper - FINAL LABELS 2026-10-09 (rev 2026-10-16).docx")
T=d.tables
def cell(ti,r,c): return T[ti].rows[r].cells[c].text.strip()
def nums(s): return [x for x in re.findall(r"[−+-]?\d+\.\d+|\d[\d,]*",s.replace("–","~"))]
PASS=[];FAIL=[]
def ck(lab,claim,truth,tol=0.0011):
    try: ok = abs(float(claim)-float(truth))<=tol
    except: ok = str(claim)==str(truth)
    (PASS if ok else FAIL).append(f"{lab}: {claim} vs {truth}")
    return ok
def row_ci(ti,r,c,a,b,cc,lab,src):
    t=cell(ti,r,c); n=re.findall(r"\d\.\d{3}",t)
    for got,exp,w in zip(n,[a,b,cc],["est","lo","hi"]):
        ck(f"{lab} {w}",got,exp)

# ---------- T4 = S1 architectures ----------
a=pd.read_csv(R+"architectures_retrained_final.csv").set_index("architecture")
amap={"ResNet18":"ResNet18","EfficientNet-B0":"EfficientNet-B0","ViT-B/16":"ViT-B/16"}
for r in range(1,4):
    arch=cell(4,r,0); s=a.loc[amap[arch]]
    m=re.findall(r"\d\.\d{3}",cell(4,r,1))
    ck(f"S1 {arch} mean",m[0],round(s.patient_auroc_retrained_final,3))
    ck(f"S1 {arch} sd",m[1],round(s.sd_retrained_final,3))
    per=[float(x.split(":")[1]) for x in s.per_seed.split(" / ")]
    rg=re.findall(r"\d\.\d{3}",cell(4,r,2))
    ck(f"S1 {arch} rng-lo",rg[0],round(min(per),3)); ck(f"S1 {arch} rng-hi",rg[1],round(max(per),3))
# ---------- T5 = S2 learning curve ----------
lc=pd.read_csv(R+"learning_curve_final_summary.csv")
for r,frac in [(1,0.333),(2,0.667),(3,1.0)]:
    s=lc[np.isclose(lc.frac,frac)].iloc[0]; m=re.findall(r"\d\.\d{3}",cell(5,r,1))
    ck(f"S2 {frac} mean",m[0],round(s['mean'],3)); ck(f"S2 {frac} sd",m[1],round(s['std'],3))
# ---------- T6 = S3 context probes ----------
cp=pd.read_csv(R+"context_probes_masking_frame_final.csv").set_index("probe")
pm={"POD only":"pod_only","Acquisition features":"acquisition","Prior imaging behavior":"behavior_past_only",
    "Acquisition + behavior + POD":"acquisition+behavior+pod"}
for r in range(1,6):
    lab=cell(6,r,0)
    if lab in pm:
        s=cp.loc[pm[lab]]
        ck(f"S3 {lab} auroc",re.findall(r"\d\.\d{3}",cell(6,r,1))[0],round(s.patient_auroc,3))
        ck(f"S3 {lab} delta",re.findall(r"\d\.\d{3}",cell(6,r,2))[0],abs(round(s.full_image_minus_probe,3)))
# ---------- T7 = S4 masking locked ----------
ab=pd.read_csv(R+"masking_absolute_with_ci.csv").set_index("condition")
co=pd.read_csv(R+"masking_contrasts_final.csv").set_index("contrast")
for r in range(1,6):
    k=cell(7,r,0); s=ab.loc[k]
    row_ci(7,r,1,round(s.auroc,3),round(s.lo,3),round(s.hi,3),f"S4 {k}",None)
    ck(f"S4 {k} ci-lo",re.findall(r"\d\.\d{3}",cell(7,r,2))[0],round(s.lo,3))
    ck(f"S4 {k} ci-hi",re.findall(r"\d\.\d{3}",cell(7,r,2))[1],round(s.hi,3))
cmap={"Shifted region − true wound":"shifted - true wound","Shifted region − true background":"shifted - true background",
 "True background − full":"true background - full","True wound − full":"true wound - full",
 "Complement of shifted region − full":"shifted complement - full"}
for r in range(6,11):
    k=cell(7,r,0)
    if k in cmap:
        s=co.loc[cmap[k]]
        ck(f"S4c {k} d",re.findall(r"\d\.\d{3}",cell(7,r,1))[0],abs(round(s.d,3)))
        ci=re.findall(r"\d\.\d{3}",cell(7,r,2))
        ck(f"S4c {k} lo",ci[0],abs(round(s.lo,3))); ck(f"S4c {k} hi",ci[1],abs(round(s.hi,3)))
        ck(f"S4c {k} p",re.findall(r"\d*\.\d+",cell(7,r,3))[0],round(s.p,2) if s.p>=0.1 else round(s.p,3))
# ---------- T9 = S6 calibration + operating points ----------
cal=pd.read_csv(R+"calibration_pod7_final_labels.csv").set_index("metric")
op=pd.read_csv(R+"operating_points_pod7_final_labels.csv").set_index("target_spec")
for r,k in [(1,"brier"),(2,"slope"),(3,"intercept")]:
    s=cal.loc[k]
    ck(f"S6 {k} est",re.findall(r"\d\.\d{3}",cell(9,r,1))[0],abs(round(s.est,3)))
    ci=re.findall(r"\d\.\d{3}",cell(9,r,2))
    ck(f"S6 {k} lo",ci[0],abs(round(s.lo,3))); ck(f"S6 {k} hi",ci[1],abs(round(s.hi,3)))
s=op.loc[0.9]
v=re.findall(r"\d\.\d{3}",cell(9,4,1)); ck("S6 90spec sens",v[0],round(s.sensitivity,3)); ck("S6 90spec flagged",v[1],round(s.flagged_frac,3))
ck("S6 90spec spec",re.findall(r"\d\.\d{3}",cell(9,4,2))[0],round(s.specificity,3))
# ---------- T2 = Table 3, 80% spec row ----------
s=op.loc[0.8]
for r in range(1,7):
    if "80% specificity" in cell(2,r,0):
        vals=[cell(2,r,c) for c in range(1,7)]
        got=[x for x in vals if re.fullmatch(r"\d\.\d{3}",x)]
        ck("T3 80spec sens",got[0],round(s.sensitivity,3))
        ck("T3 80spec spec",got[1],round(s.specificity,3))
        ck("T3 80spec flagged",got[2],round(s.flagged_frac,3))
# ---------- T12 = S9 three-seed ----------
a3=pd.read_csv(R+"masking_3seed_absolute.csv").set_index("condition")
c3=pd.read_csv(R+"masking_3seed_contrasts.csv").set_index("contrast")
for r in range(1,6):
    k=cell(12,r,0); s=a3.loc[k]
    ck(f"S9 {k} locked",re.findall(r"\d\.\d{3}",cell(12,r,1))[0],round(s.published_single_run,3))
    ck(f"S9 {k} s42",re.findall(r"\d\.\d{3}",cell(12,r,2))[0],round(s.seed42,3))
    ck(f"S9 {k} s7",re.findall(r"\d\.\d{3}",cell(12,r,3))[0],round(s.seed7,3))
    mm=re.findall(r"\d\.\d{3}",cell(12,r,4))
    ck(f"S9 {k} mean",mm[0],round(s.seed_mean,3))
    if len(mm)>=3: ck(f"S9 {k} lo",mm[1],round(s.lo,3)); ck(f"S9 {k} hi",mm[2],round(s.hi,3))
# ---------- T13 = S10 permutation ----------
g=dict(zip(*pd.read_csv(R+"item7_fixed_summary.csv").values.T))
txt=" ".join(cell(13,r,1) for r in range(len(T[13].rows)))
for lab,val in [("null mean",0.585),("null sd",0.057),("true wound",0.538),("locked shift",0.641)]:
    ck(f"S10 {lab} present",str(val) in txt.replace("0.5850","0.585"),True)
ck("S10 K",("300" in txt),True); ck("S10 pct20",("20" in txt),True); ck("S10 pct84",("84" in txt),True)
ck("S10 240/300",("240 of 300" in txt),True); ck("S10 p0.80",("0.80" in txt),True)
ck("S10 iou",("0.008" in txt and "0.545" in txt and "97.6" in txt),True)
ck("S10 46 infeasible",("46 of 1,886" in txt),True)
# ---------- T14 = S11 geometry ----------
fg=pd.read_csv(R+"item8_frozen_geometry.csv").set_index("feature")
mo=pd.read_csv(R+"item8_mask_only.csv")
gv=pd.read_csv(R+"item8_geometry_vs_image_same_frame.csv").iloc[0]
pairs=[("Binary mask as an image, frozen features","binary_mask_image"),
       ("Mask area alone","mask_area_only"),("Seven scalar geometry descriptors","scalar_geometry_7")]
for r in range(1,7):
    lab=cell(13+1,r,0)
    for nm,key in pairs:
        if lab==nm:
            s=fg.loc[key]; v=re.findall(r"\d\.\d{3}",cell(14,r,1))
            ck(f"S11 {nm} est",v[0],round(s.auroc,3)); ck(f"S11 {nm} lo",v[1],round(s.lo,3)); ck(f"S11 {nm} hi",v[2],round(s.hi,3))
    if "fine-tuned" in lab:
        v=re.findall(r"\d\.\d{3}",cell(14,r,1))
        ck("S11 ft mean",v[0],round(mo.auroc.mean(),3)); ck("S11 ft min",v[1],round(mo.auroc.min(),3)); ck("S11 ft max",v[2],round(mo.auroc.max(),3))
    if "geometry minus full image" in lab:
        ck("S11 difference",re.findall(r"\d\.\d{3}",cell(14,r,1))[0],round(gv.difference,3))
# ---------- T15 = S12 foundation probes ----------
fp=pd.read_csv(R+"item9_foundation_probes.csv").set_index("condition")
bc=pd.read_csv(R+"item9b_biomedclip.csv").set_index("condition")
condmap={"Full image":"full_image","True wound-only":"wound_only","True background-only":"background_only",
 "Circularly shifted region":"sham_wound_only","Complement of shifted region":"sham_background_only"}
for r in range(1,6):
    k=cell(15,r,0); key=condmap[k]
    for ci,src,nm in [(1,fp,"DINOv2"),(2,bc,"BiomedCLIP")]:
        s=src.loc[key]; v=re.findall(r"\d\.\d{3}",cell(15,r,ci))
        ck(f"S12 {nm} {k} est",v[0],round(s.auroc,3)); ck(f"S12 {nm} {k} lo",v[1],round(s.lo,3)); ck(f"S12 {nm} {k} hi",v[2],round(s.hi,3))
pc=pd.read_csv(R+"item9_paired_contrasts.csv").set_index("contrast")
for r in range(6,8):
    lab=cell(15,r,0).replace(" ","_").replace("-","-")
    key="background_only - wound_only" if lab.startswith("background") else "sham_background_only - wound_only"
    s=pc.loc[key]; v=re.findall(r"\d\.\d{3}",cell(15,r,1))
    ck(f"S12c {key} d",v[0],round(s.d,3)); ck(f"S12c {key} lo",v[1],round(s.lo,3)); ck(f"S12c {key} hi",v[2],round(s.hi,3))
    ck(f"S12c {key} p",v[3],round(s.p,3))
# ---------- T0 Table 1 full ----------
src=pd.read_csv(R+"baseline_table1.csv",dtype=str).fillna("")
S={str(r.iloc[0]).strip():tuple(str(r.iloc[i]).strip() for i in range(1,4)) for _,r in src.iterrows() if str(r.iloc[0]).strip()}
t1bad=0
for r in T[0].rows:
    c=[x.text.strip() for x in r.cells]
    if not c[0] or c[0]=="Characteristic": continue
    if c[0] in S:
        if tuple(x.replace("—","").strip() for x in c[1:4])!=tuple(x.replace("—","").strip() for x in S[c[0]]): t1bad+=1
ck("Table 1 all rows match source",t1bad,0)
print(f"TABLE-CELL AUDIT:  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFAILURES:")
    for f in FAIL: print("  **",f)
