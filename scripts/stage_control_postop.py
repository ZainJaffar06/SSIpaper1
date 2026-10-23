# -*- coding: utf-8 -*-
"""Table S7 positive control on the POSTOPERATIVE-ONLY frame.
Early = postoperative photographs at POD 0-2; late = POD 14-30; POD 3-13 excluded.
The published run's early class contained 321 intraoperative post-closure images, which
are gone here. EfficientNet-B0, locked patient folds, 12 epochs, image-level AUROC with a
patient-clustered bootstrap interval."""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision.models as M
from sklearn.metrics import roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from lib import cluster_boot_ci, safe_auroc
PO=os.path.join(HERE,"out","postop_only"); DER=os.path.join(PO,"derived")
DEV="mps" if torch.backends.mps.is_available() else "cpu"
MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X=np.load(os.path.join(PO,"postop_uint8.npy"),mmap_mode="r")
master=pd.read_csv(os.path.join(HERE,"out","paper1","MASTER_map_1948_seed42.csv"))
keep=np.load(os.path.join(HERE,"out","recompute","postop_row_mask.npy"))
P=master.loc[keep].reset_index(drop=True)
rp=pd.read_csv("/Users/zainjaffar/Documents/New project/ssi-paper1-information-audit/results/MASTER_map_1948_seed42.csv")
P["image_id"]=rp.image_id.values[keep]
pf=pd.read_csv(os.path.join(HERE,"out","recompute","per_photo_frame_1948.csv"))[["image_id","day_after_surgery"]]
P=P.merge(pf,on="image_id",how="left")
early=(P.day_after_surgery<=2).values; late=((P.day_after_surgery>=14)&(P.day_after_surgery<=30)).values
sel=np.where(early|late)[0]; y=late[sel].astype(np.int64); pid=P.pid.values[sel]; fold=P.fold.values[sel]
print(f"stage frame: {len(sel)} imgs | {len(set(pid))} pts | early {int((y==0).sum())} late {int(y.sum())}",flush=True)
def make():
    m=M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier[1]=nn.Linear(m.classifier[1].in_features,2); return m.to(DEV)
def batch(loc): return np.asarray(X[sel[loc]])
def run_fold(tr,te,epochs=12,bs=32):
    model=make(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    w=torch.tensor([1.0,float((y[tr]==0).sum()/max((y[tr]==1).sum(),1))]).to(DEV); crit=nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        model.train(); perm=np.random.permutation(len(tr))
        for i in range(0,len(tr),bs):
            loc=tr[perm[i:i+bs]]
            xb=torch.from_numpy(batch(loc)).to(DEV).float().permute(0,3,1,2)/255.0
            if torch.rand(1).item()<0.5: xb=torch.flip(xb,[3])
            xb=((xb-MEAN.to(DEV))/STD.to(DEV)).contiguous()
            opt.zero_grad(); crit(model(xb),torch.from_numpy(y[loc]).to(DEV)).backward(); opt.step()
        if DEV=="mps": torch.mps.empty_cache()
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(te),64):
            loc=te[i:i+64]; xb=torch.from_numpy(batch(loc)).to(DEV).float().permute(0,3,1,2)/255.0
            xb=((xb-MEAN.to(DEV))/STD.to(DEV)).contiguous(); out.append(torch.softmax(model(xb),1)[:,1].cpu().numpy())
    del model
    if DEV=="mps": torch.mps.empty_cache()
    return np.concatenate(out)
torch.manual_seed(20260807); np.random.seed(20260807); t0=time.time(); oof=np.full(len(y),np.nan)
for k in sorted(np.unique(fold)):
    tr=np.where(fold!=k)[0]; te=np.where(fold==k)[0]
    if len(np.unique(y[tr]))<2: continue
    oof[te]=run_fold(tr,te); print(f"  fold {int(k)} done ({time.time()-t0:.0f}s)",flush=True)
m=~np.isnan(oof); au=roc_auc_score(y[m],oof[m])
_,lo,hi=cluster_boot_ci(pid[m],y[m],oof[m],safe_auroc,20000,913)
pd.DataFrame({"pid":pid,"image_id":P.image_id.values[sel],"fold":fold,"late":y,"proba":oof}).to_csv(os.path.join(PO,"stage_control.csv"),index=False)
pd.DataFrame([dict(level="Image",n_images=int(m.sum()),n_patients=len(set(pid[m])),n_early=int((y[m]==0).sum()),n_late=int(y[m].sum()),
  definition="POD 0-2 vs POD 14-30, postoperative photographs only",auroc=round(au,4),lo=round(lo,4),hi=round(hi,4))]).to_csv(os.path.join(DER,"stage_control.csv"),index=False)
print(f"STAGE CONTROL image-level AUROC {au:.4f} ({lo:.3f}-{hi:.3f}) | published 0.976 with 321 intraoperative images in the early class",flush=True)
print("STAGE_CONTROL_DONE",flush=True)
