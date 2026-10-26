# -*- coding: utf-8 -*-
"""Locked CNN versus the combined context probe on the FULL postoperative frame
(1,341 photographs / 171 patients / 13 SSI). Replaces the 210-patient-frame sentence
(0.685 image / 0.731 patient; +0.064 P=0.010; +0.034 P=0.47), which included intraoperative
photographs. Same probe features as context_probes_postop.py; locked CNN = resnet18 seed 20260807
out-of-fold probabilities on this frame. Paired bootstraps: 20,000 resamples, seed 913; image level
resamples images (as the original did), patient level resamples patients."""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import ndimage
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from lib import safe_auroc, cluster_boot_ci, grouped_oof
PO=os.path.join(HERE,"out","postop_only"); DER=os.path.join(PO,"derived")
FINAL=set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())
X=np.load(os.path.join(PO,"postop_uint8.npy"),mmap_mode="r")
master=pd.read_csv(os.path.join(HERE,"out","paper1","MASTER_map_1948_seed42.csv")); keep=np.load(os.path.join(HERE,"out","recompute","postop_row_mask.npy"))
rp=pd.read_csv("/Users/zainjaffar/Documents/New project/ssi-paper1-information-audit/results/MASTER_map_1948_seed42.csv")
P=master.loc[keep].reset_index(drop=True); P["image_id"]=rp.image_id.values[keep]
pf=pd.read_csv(os.path.join(HERE,"out","recompute","per_photo_frame_1948.csv"))[["image_id","day_after_surgery"]]
P=P.merge(pf,on="image_id",how="left"); assert P.day_after_surgery.notna().all()
y=P.pid.isin(FINAL).values.astype(int); pid=P.pid.values; N=len(P); assert N==1341 and len(X)==N
print(f"frame {N} imgs | {len(set(pid))} pts | {int(pd.Series(y).groupby(pid).max().sum())} SSI",flush=True)
acq=np.zeros((N,10),np.float32)
for i in range(N):
    g=np.asarray(X[i]).astype(np.float32); gray=g.mean(2)
    lap=ndimage.laplace(gray); gx=ndimage.sobel(gray,1); gy=ndimage.sobel(gray,0); mag=np.hypot(gx,gy)
    acq[i]=[gray.mean(),gray.std(),lap.var(),np.abs(lap).mean(),g[...,0].mean(),g[...,1].mean(),g[...,2].mean(),
            (g.max(2)-g.min(2)).mean(),mag.mean(),(mag>mag.mean()+2*mag.std()).mean()]
P["_i"]=np.arange(N); beh=np.zeros((N,4),np.float32)
for q,g in P.sort_values(["pid","day_after_surgery"]).groupby("pid"):
    days=g.day_after_surgery.values; idx=g._i.values
    for j,(d,i) in enumerate(zip(days,idx)): beh[i]=[d,j,(d-days[j-1]) if j>0 else 0.0,1.0 if j==0 else 0.0]
oof=grouped_oof(np.column_stack([acq,beh]),y,pid,C=1.0)
cnn=pd.read_csv(os.path.join(PO,"resnet18_seed20260807.csv")).dropna(subset=["proba"])
m=P[["image_id","pid"]].assign(y=y,probe=oof).merge(cnn[["image_id","proba"]],on="image_id",how="inner"); assert len(m)==N,len(m)
yi=m.y.values; a=m.proba.values; b=m.probe.values; pi=m.pid.values
rng=np.random.default_rng(913)
def paired(y_,a_,b_,groups=None,nb=20000):
    obs=safe_auroc(y_,a_)-safe_auroc(y_,b_); v=[]
    if groups is None:
        for _ in range(nb):
            s=rng.integers(0,len(y_),len(y_))
            if len(np.unique(y_[s]))<2: continue
            v.append(safe_auroc(y_[s],a_[s])-safe_auroc(y_[s],b_[s]))
    else:
        gi=pd.Series(np.arange(len(y_))).groupby(groups).indices; ps=list(gi)
        for _ in range(nb):
            pick=rng.integers(0,len(ps),len(ps)); s=np.concatenate([gi[ps[k]] for k in pick])
            if len(np.unique(y_[s]))<2: continue
            v.append(safe_auroc(y_[s],a_[s])-safe_auroc(y_[s],b_[s]))
    v=np.array(v); p=min(1.0,2*min((v<=0).mean(),(v>=0).mean()))
    return round(float(obs),4),round(float(np.percentile(v,2.5)),4),round(float(np.percentile(v,97.5)),4),round(float(p),4),len(v)
rows=[]
pi_auc,pi_lo,pi_hi=cluster_boot_ci(pi,yi,b,safe_auroc,20000,913)
pg=m.groupby("pid").agg(y=("y","max"),a=("proba","mean"),b=("probe","mean"))
pp_auc,pp_lo,pp_hi=cluster_boot_ci(pg.index.values,pg.y.values,pg.b.values,safe_auroc,20000,913)
rows.append(dict(level="image",n=N,probe_auroc=round(pi_auc,4),probe_lo=round(pi_lo,4),probe_hi=round(pi_hi,4),cnn_auroc=round(safe_auroc(yi,a),4)))
rows.append(dict(level="patient",n=len(pg),probe_auroc=round(pp_auc,4),probe_lo=round(pp_lo,4),probe_hi=round(pp_hi,4),cnn_auroc=round(safe_auroc(pg.y,pg.a),4)))
io=paired(yi,a,b); ic=paired(yi,a,b,groups=pi); po=paired(pg.y.values,pg.a.values,pg.b.values)
for r,(lab,res) in zip(rows,[("image, image-resampled",io),("patient",po)]):
    r["cnn_minus_probe"],r["d_lo"],r["d_hi"],r["p"],r["n_valid"]=res
rows.append(dict(level="image, patient-resampled",n=N,probe_auroc=round(pi_auc,4),probe_lo=round(pi_lo,4),probe_hi=round(pi_hi,4),cnn_auroc=round(safe_auroc(yi,a),4),
                 cnn_minus_probe=ic[0],d_lo=ic[1],d_hi=ic[2],p=ic[3],n_valid=ic[4]))
out=pd.DataFrame(rows); out.to_csv(os.path.join(DER,"cnn_vs_context_fullframe_postop.csv"),index=False); print(out.to_string(index=False),flush=True)
print("CONTEXT_FULL_DONE",flush=True)
