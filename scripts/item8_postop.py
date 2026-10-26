# -*- coding: utf-8 -*-
"""S11 frozen rows on the POSTOPERATIVE masking frame (1,296 imgs / 168 pts / 12 SSI).
Same procedure as item8_fast_frozen.py (frozen ImageNet EfficientNet-B0 features of the
binary mask as an image; seven scalar geometry descriptors; mask area alone), each with a
logistic probe fitted inside the locked folds, patient-mean aggregation, 20,000-resample
patient-clustered bootstrap. Also the geometry->operative-region check behind the sentence
'elongation and fill ratio separate chest from abdominal incisions'.
Labels: final_ssi_list.txt only. MASTER_map is read for pid/fold/path alignment, never its label column.
Memory: features are extracted per batch of 48 from the bool mask cache; nothing large is resident."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torchvision.models as M
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import cluster_boot_ci, safe_auroc
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"out","postop_only"); DER=os.path.join(OUT,"derived")
DEV="mps" if torch.backends.mps.is_available() else "cpu"
FINAL=set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())
master=pd.read_csv(os.path.join(HERE,"out","paper1","MASTER_map_1948_seed42.csv"))
keep=np.load(os.path.join(HERE,"out","recompute","postop_row_mask.npy")); assert keep.sum()==1341
P=master.loc[keep].reset_index(drop=True)
z=np.load(os.path.join(OUT,"postop_masks.npz")); Mk=z["Mk"]; kidx=z["kidx"]
pid=P.pid.values[kidx]; fold=P.fold.values[kidx]; y=pd.Series(pid).isin(FINAL).values.astype(int)
print(f"frame {len(Mk)} imgs | {len(set(pid))} pts | {int(pd.Series(y).groupby(pid).max().sum())} SSI",flush=True)
assert len(Mk)==1296 and len(set(pid))==168
enc=M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1); enc.classifier=nn.Identity(); enc=enc.eval().to(DEV)
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEV); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEV)
@torch.no_grad()
def feats(bs=48):
    out=[]
    for i in range(0,len(Mk),bs):
        b=np.repeat(Mk[i:i+bs].astype(np.float32)[:,:,:,None],3,axis=3)
        b=torch.from_numpy(b).permute(0,3,1,2).contiguous().to(DEV)
        out.append(enc((b-MEAN)/STD).float().cpu().numpy())
        if DEV=="mps": torch.mps.empty_cache()
    return np.concatenate(out)
def probe(F):
    oof=np.zeros(len(F))
    for k in sorted(set(fold)):
        tr,te=fold!=k,fold==k
        s=StandardScaler().fit(F[tr]); lr=LogisticRegression(max_iter=3000,class_weight="balanced").fit(s.transform(F[tr]),y[tr])
        oof[te]=lr.predict_proba(s.transform(F[te]))[:,1]
    return pd.DataFrame({"pid":pid,"y":y,"p":oof}).groupby("pid").agg(y=("y","max"),p=("p","mean"))
rows=[]
def rec(name,g):
    a,lo,hi=cluster_boot_ci(list(g.index),g.y.values,g.p.values,nb=20000,seed=913)
    print(f"  {name:36s} AUROC {a:.3f} ({lo:.3f}-{hi:.3f})",flush=True)
    rows.append(dict(feature=name,auroc=round(a,4),lo=round(lo,4),hi=round(hi,4)))
rec("binary_mask_image_frozen",probe(feats()))
nm=["area","centroid_y","centroid_x","bbox_h","bbox_w","elongation","fill_ratio"]; G=[]
for i in range(len(Mk)):
    ys_,xs_=np.where(Mk[i]); h=ys_.max()-ys_.min()+1; w=xs_.max()-xs_.min()+1
    G.append([Mk[i].mean(),ys_.mean()/224,xs_.mean()/224,h/224,w/224,h/max(w,1),Mk[i].sum()/max(h*w,1)])
G=np.array(G)
rec("scalar_geometry_7",probe(G))
rec("mask_area_only",probe(G[:,:1]))
ma=pd.read_csv(os.path.join(DER,"masking_absolute.csv")); fi=float(ma.loc[ma.condition=="Full image","seed_mean"].iloc[0])
rows.append(dict(feature="full_image_same_frame_3seed_mean",auroc=round(fi,4),lo=np.nan,hi=np.nan))
rows.append(dict(feature="geometry_minus_full_image",auroc=round(rows[1]["auroc"]-fi,4),lo=np.nan,hi=np.nan))
pd.DataFrame(rows).to_csv(os.path.join(DER,"frozen_geometry_postop.csv"),index=False)
# geometry -> operative region (patient means of the descriptors vs chest/abdomen)
rl=os.path.join(HERE,"out","recompute","region_labels_FOR_REVIEW.csv")
if os.path.exists(rl):
    r=pd.read_csv(rl); reg=dict(zip(r.patient_id,r.region))
    pm=pd.DataFrame(G,columns=nm); pm["pid"]=pid; pm=pm.groupby("pid").mean(); pm["region"]=pm.index.map(reg)
    k=pm[pm.region.isin(["Chest","Abdomen"])]; ab=(k.region=="Abdomen").astype(int)
    g2=pd.DataFrame([dict(feature=n,auroc_for_region=round(safe_auroc(ab,k[n]),4),chest_mean=round(k[k.region=="Chest"][n].mean(),4),
        abdomen_mean=round(k[k.region=="Abdomen"][n].mean(),4)) for n in nm])
    g2.attrs={}; g2.to_csv(os.path.join(DER,"geometry_encodes_region_postop.csv"),index=False)
    print(f"region check: {int((k.region=='Chest').sum())} chest / {int(ab.sum())} abdomen patients",flush=True); print(g2.to_string(index=False),flush=True)
else: print("region labels file missing; region check skipped",flush=True)
print("ITEM8_POSTOP_DONE",flush=True)
