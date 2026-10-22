# -*- coding: utf-8 -*-
"""Constrained sham permutation test on the POSTOPERATIVE-ONLY masking frame.

Same exact-feasibility construction as item7_fixed.py: for every image the full set
of translations that keep the mask inside the frame and hold overlap with the true
mask at or below t is enumerated by valid-mode cross-correlation. Both masks have
equal area A, so union = 2A - inter and IoU <= t  <=>  inter <= 2At/(1+t).

Also re-scores the single circular shift used in the masked retraining on THIS
estimator, so its percentile is frame- and estimator-matched rather than being
compared across a fine-tuned and a frozen pipeline.

Memory: images stay uint8 and masking is applied per batch; nothing large is held
as float32 (the machine runs with ~1.4GB free swap).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision.models as M, os, time, sys
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import safe_auroc

HERE=os.path.dirname(os.path.abspath(__file__))
PO=os.path.join(HERE,"out","postop_only"); DER=os.path.join(PO,"derived"); os.makedirs(DER,exist_ok=True)
DEV="mps" if torch.backends.mps.is_available() else "cpu"
K=int(os.environ.get("NDRAW","300")); T=0.05
FINAL=set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())

z=np.load(os.path.join(PO,"postop_masks.npz")); Mk=z["Mk"]; kidx=z["kidx"]
SH=np.load(os.path.join(PO,"postop_sham.npz"))["SH"]
Xc=np.load(os.path.join(PO,"postop_uint8.npy"), mmap_mode="r")
master=pd.read_csv(os.path.join(HERE,"out","paper1","MASTER_map_1948_seed42.csv"))
keep=np.load(os.path.join(HERE,"out","recompute","postop_row_mask.npy"))
P=master.loc[keep].reset_index(drop=True).iloc[kidx]
y=P.pid.isin(FINAL).values.astype(int); pid=P.pid.values; fold=P.fold.values
N=len(kidx)
print(f"postop masking frame: {N} images | {len(set(pid))} patients | "
      f"{int(pd.Series(y).groupby(pid).max().sum())} SSI pts | K={K}",flush=True)

t0=time.time(); FEAS=[]; infeas=[]; minio=[]; BB=[]
for i in range(N):
    ys_,xs_=np.where(Mk[i]); r0,r1,c0,c1=ys_.min(),ys_.max()+1,xs_.min(),xs_.max()+1
    BB.append((r0,r1,c0,c1))
    patch=Mk[i][r0:r1,c0:c1].astype(np.float32); A=float(patch.sum())
    corr=fftconvolve(Mk[i].astype(np.float32),patch[::-1,::-1],mode="valid")
    inter=np.clip(corr,0,None); iou=inter/np.maximum(2*A-inter,1e-9)
    ok=np.argwhere(iou<=T); minio.append(float(iou.min()))
    if len(ok)==0:
        infeas.append(i); FEAS.append(np.array([np.unravel_index(np.argmin(iou),iou.shape)]))
    else: FEAS.append(ok)
print(f"feasible sets in {time.time()-t0:.0f}s | no feasible position: {len(infeas)}"
      + (f" (min achievable IoU median {np.median([minio[i] for i in infeas]):.3f})" if infeas else ""),flush=True)

enc=M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
enc.classifier=nn.Identity(); enc=enc.eval().to(DEV)
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEV)
STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEV)

@torch.no_grad()
def feats(mask,bs=48):
    """Frozen features with `mask` applied per batch; images never leave uint8 on disk."""
    out=[]
    for i in range(0,N,bs):
        g=np.asarray(Xc[kidx[i:i+bs]])
        if mask is not None: g=g*mask[i:i+bs,:,:,None]
        b=torch.from_numpy(np.ascontiguousarray(g)).permute(0,3,1,2).float().to(DEV)/255.0
        out.append(enc((b-MEAN)/STD).float().cpu().numpy())
        if DEV=="mps": torch.mps.empty_cache()
    return np.concatenate(out)

def patient_auroc(F):
    oof=np.zeros(len(F))
    for k in sorted(set(fold)):
        tr,te=fold!=k,fold==k
        if len(np.unique(y[tr]))<2: continue
        s=StandardScaler().fit(F[tr])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(s.transform(F[tr]),y[tr])
        oof[te]=lr.predict_proba(s.transform(F[te]))[:,1]
    g=pd.DataFrame({"pid":pid,"y":y,"p":oof}).groupby("pid").agg(y=("y","max"),p=("p","mean"))
    return safe_auroc(g.y,g.p)

obs=patient_auroc(feats(Mk));            print(f"TRUE WOUND  : {obs:.4f}",flush=True)
shift=patient_auroc(feats(SH));           print(f"LOCKED SHIFT (re-scored here): {shift:.4f}",flush=True)

rg=np.random.default_rng(913); null=[]; iou_all=[]
for dd in range(K):
    t1=time.time(); newm=np.zeros_like(Mk)
    for i in range(N):
        r0,r1,c0,c1=BB[i]; h,w=r1-r0,c1-c0
        nr,nc=FEAS[i][rg.integers(0,len(FEAS[i]))]
        newm[i][nr:nr+h,nc:nc+w]=Mk[i][r0:r1,c0:c1]
        u=(newm[i]|Mk[i]).sum()
        iou_all.append((newm[i]&Mk[i]).sum()/u if u else 1.0)
    a=patient_auroc(feats(newm)); null.append(a)
    if dd<3 or (dd+1)%25==0: print(f"  draw {dd+1:4d}/{K} AUROC {a:.4f} [{time.time()-t1:.0f}s]",flush=True)

null=np.array(null); io=np.array(iou_all)
ge=int((null>=obs).sum()); le=int((null<=obs).sum())
pr=(1+ge)/(K+1); pl=(1+le)/(K+1)
print("\n"+"="*70)
print(f"POSTOPERATIVE-ONLY PERMUTATION TEST, {K} exact-feasible relocations")
print("="*70)
print(f"  IoU: median {np.median(io):.5f} mean {io.mean():.5f} max {io.max():.4f} | <=0.05 {100*(io<=T).mean():.2f}%")
print(f"  null: mean {null.mean():.4f} SD {null.std(ddof=1):.4f} range {null.min():.4f}-{null.max():.4f}")
print(f"  TRUE WOUND {obs:.4f} -> percentile {100*(null<obs).mean():.1f}")
print(f"  relocations >= true wound: {ge}/{K}  one-sided P (wound better) {pr:.4f}")
print(f"  LOCKED SHIFT {shift:.4f} -> percentile {100*(null<shift).mean():.1f}")
med=np.median(null)
if med>obs: print(f"  locked-shift advantage over wound = {shift-obs:+.4f}; typical relocation {med-obs:+.4f}"
                  f"  -> ratio {(shift-obs)/(med-obs):.2f}x")
pd.DataFrame(dict(draw=np.arange(1,K+1),auroc=null)).to_csv(os.path.join(DER,"permutation_null.csv"),index=False)
pd.DataFrame([dict(quantity="K",value=K),
 dict(quantity="n_images",value=N),
 dict(quantity="n_patients",value=len(set(pid))),
 dict(quantity="n_ssi_patients",value=int(pd.Series(y).groupby(pid).max().sum())),
 dict(quantity="true_wound_auroc",value=round(obs,4)),
 dict(quantity="locked_shift_auroc_rescored",value=round(shift,4)),
 dict(quantity="locked_shift_percentile",value=round(100*float((null<shift).mean()),1)),
 dict(quantity="null_mean",value=round(float(null.mean()),4)),
 dict(quantity="null_sd",value=round(float(null.std(ddof=1)),4)),
 dict(quantity="null_min",value=round(float(null.min()),4)),
 dict(quantity="null_max",value=round(float(null.max()),4)),
 dict(quantity="true_wound_percentile",value=round(100*float((null<obs).mean()),1)),
 dict(quantity="n_ge_true_wound",value=ge),
 dict(quantity="one_sided_p_wound_better",value=round(pr,4)),
 dict(quantity="iou_median",value=round(float(np.median(io)),5)),
 dict(quantity="iou_mean",value=round(float(io.mean()),5)),
 dict(quantity="iou_max",value=round(float(io.max()),4)),
 dict(quantity="iou_frac_le_0.05",value=round(float((io<=T).mean()),4)),
 dict(quantity="n_images_no_feasible_position",value=len(infeas)),
]).to_csv(os.path.join(DER,"permutation_summary.csv"),index=False)
print("\nPERMUTATION_POSTOP_DONE",flush=True)
