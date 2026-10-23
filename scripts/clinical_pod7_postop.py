# -*- coding: utf-8 -*-
"""Table 4 POD-7 rows: clinical / image / stacked restricted to the 132-patient risk set,
image score = the POD-7 primary patient score (as in the published table)."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open("clinical_postop.py").read().split("# ---- image score")[0])   # reuse covariate build (BB, cov, IO8, NEW5, FINAL, nested deps)
from lib import safe_auroc, safe_auprc, cluster_boot_ci
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
DER=os.path.join(HERE,"out","postop_only","derived")
S=pd.read_csv(os.path.join(DER,"pod7_patient_scores.csv")).set_index("pid")
X=cov.join(BB,how="outer").join(S.score.rename("image_score"),how="inner")
X["y"]=[int(i in FINAL) for i in X.index]; y=X.y.values.astype(int); pid=X.index.values
print(f"POD-7 risk set with covariates: {len(X)} pts / {y.sum()} events",flush=True)
def nested(cols,seed=20260807,Cs=(0.01,0.1,1.0)):
    D=X[cols].values.astype(float); oof=np.full(len(y),np.nan)
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(D,y,pid):
        mu=np.nanmean(D[tr],axis=0); mu=np.where(np.isfinite(mu),mu,0.0)
        P=np.where(np.isnan(D[tr]),mu,D[tr]); Q=np.where(np.isnan(D[te]),mu,D[te])
        sc=StandardScaler().fit(P); P,Q=sc.transform(P),sc.transform(Q); best,ba=Cs[0],-1
        for C in Cs:
            au=[]
            for t2,v2 in StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(P,y[tr],pid[tr]):
                if len(np.unique(y[tr][t2]))<2 or len(np.unique(y[tr][v2]))<2: continue
                m=LogisticRegression(C=C,max_iter=5000,class_weight="balanced").fit(P[t2],y[tr][t2])
                au.append(safe_auroc(y[tr][v2],m.predict_proba(P[v2])[:,1]))
            if au and np.mean(au)>ba: ba,best=np.mean(au),C
        m=LogisticRegression(C=best,max_iter=5000,class_weight="balanced").fit(P,y[tr]); oof[te]=m.predict_proba(Q)[:,1]
    return oof
def rep(lab,s):
    au=safe_auroc(y,s); ap=safe_auprc(y,s); _,lo,hi=cluster_boot_ci(pid,y,s,safe_auroc,20000,913)
    return dict(model=lab,n_pts=len(y),n_events=int(y.sum()),auroc=round(au,4),lo=round(lo,4),hi=round(hi,4),auprc=round(ap,4))
cl=nested(IO8); im=X.image_score.values; st=nested(IO8+["image_score"])
M=pd.DataFrame([rep("clinical",cl),rep("image",im),rep("stacked",st)])
M.to_csv(os.path.join(DER,"clinical_pod7.csv"),index=False); print(M.to_string(index=False))
# stacked - image paired
rng=np.random.default_rng(913); uq=np.unique(pid); reps=[]
for _ in range(20000):
    bs=rng.choice(uq,len(uq),replace=True); ii=np.concatenate([np.where(pid==p)[0] for p in bs])
    if len(np.unique(y[ii]))<2: continue
    reps.append(safe_auroc(y[ii],st[ii])-safe_auroc(y[ii],im[ii]))
reps=np.array(reps); d=safe_auroc(y,st)-safe_auroc(y,im); p=2*min((reps<=0).mean(),(reps>=0).mean())
pd.DataFrame([dict(contrast="stacked - image (POD7)",d=round(d,4),lo=round(np.percentile(reps,2.5),4),hi=round(np.percentile(reps,97.5),4),p=round(float(p),4))]).to_csv(os.path.join(DER,"clinical_pod7_contrast.csv"),index=False)
print(f"stacked - image: {d:+.4f} ({np.percentile(reps,2.5):+.4f} to {np.percentile(reps,97.5):+.4f}) P={p:.3f}")
print("CLINICAL_POD7_DONE")
