#!/usr/bin/env python
"""Downstream analyses for the postoperative-only re-run.

Consumes out/postop_only/*.csv (one OOF file per run) and writes every table the
manuscript needs to out/postop_only/derived/. Each section is skipped when its
inputs are not yet present, so this can be run repeatedly while training proceeds.

Labels: final_ssi_list.txt only. Days: per_photo_frame_1948.csv. Diagnosis timing:
dx_pod from base/ssi_smartphone.csv. Never reads the superseded label sources.
"""
import os, sys, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import safe_auroc, safe_auprc, cluster_boot_ci, dedup

HERE=os.path.dirname(os.path.abspath(__file__))
PO=os.path.join(HERE,"out","postop_only"); DER=os.path.join(PO,"derived"); os.makedirs(DER,exist_ok=True)
REC=os.path.join(HERE,"out","recompute")
FINAL=set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())
NB=20000; SEED=913

frame=pd.read_csv(os.path.join(REC,"per_photo_frame_1948.csv"))
DAY=frame.set_index(["pid","image_id"]).day_after_surgery
b=dedup(pd.read_csv(os.path.join(HERE,"base","ssi_smartphone.csv")))
DX=b.assign(d=pd.to_numeric(b.dx_pod,errors="coerce")).groupby("pid").d.first()

def load(tag):
    fp=os.path.join(PO,f"{tag}.csv")
    if not os.path.exists(fp): return None
    d=pd.read_csv(fp).dropna(subset=["proba"])
    d["day"]=[DAY.get((p,i),np.nan) for p,i in zip(d.pid,d.image_id)]
    d["y"]=d.pid.isin(FINAL).astype(int)
    return d

def risk_set(d,L=7):
    out=[]
    for p,g in d.groupby("pid"):
        dp=DX.get(p); y=int(g.y.max())
        if y==1 and pd.isna(dp): continue        # SSI with unknown dx day
        if pd.notna(dp) and dp<=L: continue      # diagnosed by the landmark
        out.append(p)
    return set(out)

def agg(d,L=7,how="mean"):
    rs=risk_set(d,L); s=d[d.pid.isin(rs)&(d.day<=L)]
    if how=="mean":  g=s.groupby("pid").agg(y=("y","max"),p=("proba","mean"))
    elif how=="last":
        s=s.sort_values("day"); g=s.groupby("pid").agg(y=("y","max"),p=("proba","last"))
    else:
        s=s.assign(k=(s.day-L).abs()).sort_values("k")
        g=s.groupby("pid").agg(y=("y","max"),p=("proba","first"))
    return g

def row(g,lab,nb=NB):
    au=safe_auroc(g.y,g.p); ap=safe_auprc(g.y,g.p)
    _,lo,hi=cluster_boot_ci(g.index.values,g.y.values,g.p.values,safe_auroc,nb,SEED)
    return dict(analysis=lab,n_pts=len(g),n_events=int(g.y.sum()),
                auroc=round(au,4),lo=round(lo,4),hi=round(hi,4),auprc=round(ap,4))

# ---------- 1. PRIMARY + aggregation sensitivities + later landmarks ----------
d=load("resnet18_seed20260807")
if d is not None:
    R=[row(agg(d,7,"mean"),"POD 7, mean of photographs through POD 7 (PRIMARY)"),
       row(agg(d,7,"last"),"POD 7, latest photograph through POD 7"),
       row(agg(d,7,"near"),"POD 7, photograph closest to POD 7"),
       row(agg(d,14,"mean"),"POD 14, mean"),
       row(agg(d,30,"mean"),"POD 30, mean")]
    pd.DataFrame(R).to_csv(os.path.join(DER,"primary_and_landmarks.csv"),index=False)
    print(pd.DataFrame(R).to_string(index=False),flush=True)
    g=agg(d,7,"mean")
    g.reset_index().rename(columns={"y":"ssi_final","p":"score"}).merge(
        d[["pid","fold"]].drop_duplicates("pid"),on="pid",how="left"
    )[["pid","ssi_final","fold","score"]].to_csv(os.path.join(DER,"pod7_patient_scores.csv"),index=False)
    # operating points + calibration
    from lib import operating_points, calibration
    try:
        operating_points(g.y.values,g.p.values).to_csv(os.path.join(DER,"operating_points.csv"),index=False)
    except Exception as e: print("operating_points:",e)
    try:
        sl,ic=calibration(g.y.values,g.p.values)
        br=float(np.mean((g.p.values-g.y.values)**2)); pr=float(g.y.mean())
        ni=float(pr*(1-pr))
        pd.DataFrame([dict(metric="brier",value=round(br,4)),
                      dict(metric="no_information_brier",value=round(ni,4)),
                      dict(metric="brier_skill",value=round(1-br/ni,4)),
                      dict(metric="calibration_slope",value=round(sl,4)),
                      dict(metric="calibration_intercept",value=round(ic,4)),
                      dict(metric="prevalence",value=round(pr,4))]).to_csv(os.path.join(DER,"calibration.csv"),index=False)
    except Exception as e: print("calibration:",e)
else:
    print("primary run not finished yet",flush=True)

# ---------- 2. architectures x seeds ----------
rows=[]
for arch in ["resnet18","efficientnet_b0","vit_b_16"]:
    per=[]
    for s in (20260807,42,7):
        dd=load(f"{arch}_seed{s}")
        if dd is None: continue
        g=dd.groupby("pid").agg(y=("y","max"),p=("proba","mean"))
        per.append((s,safe_auroc(g.y,g.p)))
    if len(per)==3:
        v=np.array([x[1] for x in per])
        rows.append(dict(architecture=arch,n_seeds=3,mean=round(v.mean(),4),sd=round(v.std(ddof=1),4),
                         lo=round(v.min(),4),hi=round(v.max(),4),
                         per_seed=" / ".join(f"{s}:{a:.4f}" for s,a in per)))
if rows:
    pd.DataFrame(rows).to_csv(os.path.join(DER,"architectures.csv"),index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

# ---------- 3. learning curve ----------
lc=[]
for frac,tag in [(0.333,"lc_0.333"),(0.667,"lc_0.667"),(1.0,None)]:
    per=[]
    for s in (20260807,42,7):
        dd=load(f"efficientnet_b0_seed{s}") if tag is None else load(f"{tag}_seed{s}")
        if dd is None: continue
        g=dd.groupby("pid").agg(y=("y","max"),p=("proba","mean"))
        per.append(safe_auroc(g.y,g.p))
    if len(per)==3:
        v=np.array(per)
        lc.append(dict(frac=frac,mean=round(v.mean(),4),sd=round(v.std(ddof=1),4),
                       lo=round(v.min(),4),hi=round(v.max(),4)))
if lc:
    pd.DataFrame(lc).to_csv(os.path.join(DER,"learning_curve.csv"),index=False)
    print(pd.DataFrame(lc).to_string(index=False),flush=True)

# ---------- 4. masking absolutes + 3-seed contrasts ----------
CONDS=[("full_image","Full image"),("wound_only","True wound-only"),
       ("background_only","True background-only"),("sham_wound_only","Circularly shifted wound-size region"),
       ("sham_background_only","Complement of shifted region")]
SEEDS=(20260807,42,7)
pat={}
for c,_ in CONDS:
    for s in SEEDS:
        dd=load(f"mask_{c}_seed{s}")
        if dd is None: continue
        pat[(c,s)]=dd.groupby("pid").agg(y=("y","max"),p=("proba","mean"))
if len(pat)==len(CONDS)*len(SEEDS):
    pids=sorted(set.intersection(*[set(v.index) for v in pat.values()]))
    yv=pat[(CONDS[0][0],SEEDS[0])].loc[pids,"y"].values.astype(int)
    S={c:np.column_stack([pat[(c,s)].loc[pids,"p"].values for s in SEEDS]) for c,_ in CONDS}
    ab=[]
    for c,lab in CONDS:
        per=[safe_auroc(yv,S[c][:,k]) for k in range(3)]
        ab.append(dict(condition=lab,seed20260807=round(per[0],4),seed42=round(per[1],4),seed7=round(per[2],4),
                       seed_mean=round(np.mean(per),4),seed_sd=round(np.std(per,ddof=1),4)))
    pd.DataFrame(ab).to_csv(os.path.join(DER,"masking_absolute.csv"),index=False)
    print(pd.DataFrame(ab).to_string(index=False),flush=True)
    PAIRS=[("sham_wound_only","wound_only","shifted - true wound"),
           ("background_only","wound_only","true background - true wound"),
           ("sham_wound_only","background_only","shifted - true background"),
           ("background_only","full_image","true background - full"),
           ("wound_only","full_image","true wound - full"),
           ("sham_background_only","full_image","shifted complement - full")]
    rng=np.random.default_rng(SEED); pid_arr=np.array(pids); uq=np.unique(pid_arr)
    boot=[rng.choice(uq,len(uq),replace=True) for _ in range(NB)]
    idx=[np.concatenate([np.where(pid_arr==p)[0] for p in bs]) for bs in boot]
    out=[]
    for a,bb,lab in PAIRS:
        obs=np.mean([safe_auroc(yv,S[a][:,k])-safe_auroc(yv,S[bb][:,k]) for k in range(3)])
        reps=[]
        for ii in idx:
            if len(np.unique(yv[ii]))<2: continue
            reps.append(np.mean([safe_auroc(yv[ii],S[a][ii,k])-safe_auroc(yv[ii],S[bb][ii,k]) for k in range(3)]))
        reps=np.array(reps)
        p=2*min((reps<=0).mean(),(reps>=0).mean())
        out.append(dict(contrast=lab,d=round(obs,4),lo=round(np.percentile(reps,2.5),4),
                        hi=round(np.percentile(reps,97.5),4),p=round(float(p),4)))
    o=pd.DataFrame(out).sort_values("p").reset_index(drop=True)
    m=len(o); o["holm_p"]=[round(min(1.0,max(o.p[:i+1]*(m-np.arange(i+1)))),4) for i in range(m)]
    o.to_csv(os.path.join(DER,"masking_contrasts.csv"),index=False)
    print(o.to_string(index=False),flush=True)

# ---------- 5. geometry: mask-as-image ----------
gm=[]
for s in SEEDS:
    dd=load(f"maskimage_seed{s}")
    if dd is None: continue
    g=dd.groupby("pid").agg(y=("y","max"),p=("proba","mean"))
    gm.append(safe_auroc(g.y,g.p))
if len(gm)==3:
    v=np.array(gm)
    pd.DataFrame([dict(input="binary mask as image, fine-tuned",n_seeds=3,mean=round(v.mean(),4),
                       lo=round(v.min(),4),hi=round(v.max(),4))]).to_csv(os.path.join(DER,"maskimage.csv"),index=False)
    print("maskimage 3-seed:",np.round(v,4),flush=True)

print("\nDOWNSTREAM_DONE",flush=True)
