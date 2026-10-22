# -*- coding: utf-8 -*-
"""Clinical, image and stacked comparisons on the POSTOPERATIVE-ONLY frame.

Two clinical specifications, as requested:
  8-variable intraoperative set (as published)
  5-variable alternative: BMI, CDC wound class, intraoperative temperature nadir,
                          smoking, diabetes
The image score is the patient-level mean across all nine architecture-by-seed
postoperative-only runs, matching the published construction.

Labels from final_ssi_list.txt only. Nested CV with C selected inside each outer
fold. Patient-clustered bootstrap for paired differences.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import dedup, safe_auroc, safe_auprc, cluster_boot_ci
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

HERE=os.path.dirname(os.path.abspath(__file__))
DER=os.path.join(HERE,"out","postop_only","derived")
FINAL=set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())
NB=int(os.environ.get("NB","2000"))

# ---- clinical covariates ----
F="/Volumes/Backup Plus/Bluebelle_demographics_tracker(Bluebelle specific Cleaned DEMO).csv"
B=pd.read_csv(F,encoding="latin-1",low_memory=False); B.columns=[c.strip() for c in B.columns]
B["pid"]=B["SUBJECT ID"].astype(str).str.strip().str.upper().str.replace(" ","",regex=False)
B=B[B.pid.str.match(r"RU-A\d{4}")].copy()
B["bmi"]=pd.to_numeric(B["BMI"],errors="coerce"); B.loc[(B.bmi<12)|(B.bmi>80),"bmi"]=np.nan
B["wound_class"]=pd.to_numeric(B["CR: Wound classification"],errors="coerce")
B.loc[~B.wound_class.isin([1,2,3,4]),"wound_class"]=np.nan
sm=B["Smoking status"].astype(str).str.strip().str.lower()
B["smoking"]=sm.map({"never":0.0,"former":1.0,"current":2.0})
dm=B["Diabetes mellitus"].astype(str).str.strip().str.lower()
B["diabetes"]=np.where(dm.str.contains("non-diabetes"),0.0,np.where(dm.str.contains("diabet"),1.0,np.nan))
BB=B.groupby("pid")[["bmi","wound_class","smoking","diabetes"]].median()

IO8=["incision_cm","closure_temp_c","nadir_temp_c","wound_packing","mesh","wound_protector","intraop_cultures","closure_min"]
d=dedup(pd.read_csv(os.path.join(HERE,"base","ssi_smartphone.csv"),low_memory=False))
cov=d.drop_duplicates("pid").set_index("pid")[IO8].apply(pd.to_numeric,errors="coerce")
for c in ("closure_temp_c","nadir_temp_c"): cov.loc[(cov[c]<30)|(cov[c]>43),c]=np.nan

# ---- image score: mean over the nine postoperative-only runs ----
S=None
for f in sorted(glob.glob(os.path.join(HERE,"out","postop_only","*_seed*.csv"))):
    if os.path.basename(f).startswith(("mask_","maskimage","lc_")): continue
    r=pd.read_csv(f)[["pid","image_id","proba"]].rename(columns={"proba":os.path.basename(f)})
    S=r if S is None else S.merge(r,on=["pid","image_id"],how="outer")
runs=[c for c in S.columns if c not in ("pid","image_id")]
print(f"image score averaged over {len(runs)} runs",flush=True)
S["img"]=S[runs].mean(axis=1)
IM=S.groupby("pid").img.mean()

pf=pd.read_csv(os.path.join(HERE,"out","recompute","per_photo_frame_1948.csv"))
POSTOP=set(pf.loc[pf.photo_type=="postoperative","pid"])
X=cov.join(BB,how="outer").join(IM.rename("image_score"),how="inner")
X=X.loc[[p for p in X.index if p in POSTOP]]
X["y"]=[int(i in FINAL) for i in X.index]
y=X.y.values.astype(int); pid=X.index.values
print(f"full postoperative cohort: {len(X)} patients / {y.sum()} events",flush=True)

NEW5=["bmi","wound_class","nadir_temp_c","smoking","diabetes"]
def nested(cols,seed=20260807,Cs=(0.01,0.1,1.0)):
    D=X[cols].values.astype(float); oof=np.full(len(y),np.nan)
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(D,y,pid):
        mu=np.nanmean(D[tr],axis=0); mu=np.where(np.isfinite(mu),mu,0.0)
        P=np.where(np.isnan(D[tr]),mu,D[tr]); Q=np.where(np.isnan(D[te]),mu,D[te])
        sc=StandardScaler().fit(P); P,Q=sc.transform(P),sc.transform(Q)
        best,ba=Cs[0],-1
        for C in Cs:
            au=[]
            for t2,v2 in StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(P,y[tr],pid[tr]):
                if len(np.unique(y[tr][t2]))<2: continue
                m=LogisticRegression(C=C,max_iter=5000,class_weight="balanced").fit(P[t2],y[tr][t2])
                au.append(safe_auroc(y[tr][v2],m.predict_proba(P[v2])[:,1]))
            if au and np.mean(au)>ba: ba,best=np.mean(au),C
        m=LogisticRegression(C=best,max_iter=5000,class_weight="balanced").fit(P,y[tr])
        oof[te]=m.predict_proba(Q)[:,1]
    return oof

def rep(lab,s):
    au=safe_auroc(y,s); ap=safe_auprc(y,s)
    _,lo,hi=cluster_boot_ci(pid,y,s,safe_auroc,NB,913)
    return dict(model=lab,n_pts=len(y),n_events=int(y.sum()),auroc=round(au,4),
                lo=round(lo,4),hi=round(hi,4),auprc=round(ap,4))

cl8=nested(IO8); cl5=nested(NEW5); img=X.image_score.values
st8=nested(IO8+["image_score"]); st5=nested(NEW5+["image_score"])
M=[rep("clinical 8-variable",cl8),rep("clinical v2 5-variable",cl5),
   rep("image only",img),rep("stacked, 8-variable + image",st8),rep("stacked, v2 + image",st5)]
pd.DataFrame(M).to_csv(os.path.join(DER,"clinical_models.csv"),index=False)
print(pd.DataFrame(M).to_string(index=False),flush=True)

def paired(a,b,lab):
    rng=np.random.default_rng(913); uq=np.unique(pid); reps=[]
    for _ in range(NB):
        bs=rng.choice(uq,len(uq),replace=True)
        ii=np.concatenate([np.where(pid==p)[0] for p in bs])
        if len(np.unique(y[ii]))<2: continue
        reps.append(safe_auroc(y[ii],a[ii])-safe_auroc(y[ii],b[ii]))
    reps=np.array(reps); obs=safe_auroc(y,a)-safe_auroc(y,b)
    p=2*min((reps<=0).mean(),(reps>=0).mean())
    return dict(contrast=lab,d_auroc=round(obs,4),lo=round(np.percentile(reps,2.5),4),
                hi=round(np.percentile(reps,97.5),4),p=round(float(p),4))
C=[paired(st8,cl8,"image adds to clinical 8-variable"),
   paired(st5,cl5,"image adds to clinical v2 5-variable")]
pd.DataFrame(C).to_csv(os.path.join(DER,"clinical_contrasts.csv"),index=False)
print(pd.DataFrame(C).to_string(index=False),flush=True)

# partition stability of the 8-variable increment
ps=[]
for sd in (20260807,42,7,2024,99):
    a=nested(IO8+["image_score"],seed=sd); b=nested(IO8,seed=sd)
    ps.append(dict(partition_seed=sd,d_auroc=round(safe_auroc(y,a)-safe_auroc(y,b),4)))
pd.DataFrame(ps).to_csv(os.path.join(DER,"clinical_partition_stability.csv"),index=False)
print(pd.DataFrame(ps).to_string(index=False),flush=True)
print("CLINICAL_POSTOP_DONE",flush=True)
