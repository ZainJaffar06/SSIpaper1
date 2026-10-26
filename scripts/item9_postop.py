# -*- coding: utf-8 -*-
"""S12 on the POSTOPERATIVE masking frame: frozen DINOv2 ViT-B/14 and BiomedCLIP ViT-B/16
linear probes across the five masking conditions, locked folds, patient-level AUROC with
patient-clustered CIs, and the five paired contrasts (Holm across the family). Images are
read per batch from the memory-mapped compact cache; masks applied per batch."""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, timm
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from lib import safe_auroc, cluster_boot_ci
PO=os.path.join(HERE,"out","postop_only"); DER=os.path.join(PO,"derived")
DEV="mps" if torch.backends.mps.is_available() else "cpu"
FINAL=set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())
z=np.load(os.path.join(PO,"postop_masks.npz")); Mk=z["Mk"]; kidx=z["kidx"]; SH=np.load(os.path.join(PO,"postop_sham.npz"))["SH"]
X=np.load(os.path.join(PO,"postop_uint8.npy"),mmap_mode="r")
master=pd.read_csv(os.path.join(HERE,"out","paper1","MASTER_map_1948_seed42.csv")); keep=np.load(os.path.join(HERE,"out","recompute","postop_row_mask.npy"))
P=master.loc[keep].reset_index(drop=True).iloc[kidx]; y=P.pid.isin(FINAL).values.astype(int); pid=P.pid.values; fold=P.fold.values; N=len(kidx)
print(f"frame {N} imgs | {len(set(pid))} pts | {int(pd.Series(y).groupby(pid).max().sum())} SSI",flush=True)
COND={"full_image":None,"wound_only":("keep",Mk),"background_only":("drop",Mk),"sham_wound_only":("keep",SH),"sham_background_only":("drop",SH)}
ENC={}
dv=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",verbose=False).eval().to(DEV)
ENC["dinov2_vitb14"]=(dv,(0.485,0.456,0.406),(0.229,0.224,0.225))
ck=hf_hub_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224","open_clip_pytorch_model.bin")
sd=torch.load(ck,map_location="cpu"); trunk={k[len("visual.trunk."):]:v for k,v in sd.items() if k.startswith("visual.trunk.")}
bv=timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=0); r=bv.load_state_dict(trunk,strict=False)
assert not r.missing_keys and not r.unexpected_keys; bv=bv.eval().to(DEV)
ENC["biomedclip_vitb16"]=(bv,(0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
@torch.no_grad()
def embed(enc,mean,std,cond,bs=32):
    mu=torch.tensor(mean).view(1,3,1,1).to(DEV); sdv=torch.tensor(std).view(1,3,1,1).to(DEV); out=[]
    for i in range(0,N,bs):
        g=np.asarray(X[kidx[i:i+bs]])
        if cond is not None:
            mode,M=cond; m=M[i:i+bs]; g=g*(m if mode=="keep" else ~m)[...,None]
        b=torch.from_numpy(np.ascontiguousarray(g)).permute(0,3,1,2).float().to(DEV)/255.0
        f=enc((b-mu)/sdv); out.append((f if f.ndim==2 else f.mean(1)).float().cpu().numpy())
        if DEV=="mps": torch.mps.empty_cache()
    return np.concatenate(out)
def probe_oof(F):
    oof=np.zeros(len(F))
    for k in sorted(set(fold)):
        tr,te=fold!=k,fold==k
        if len(np.unique(y[tr]))<2: continue
        s=StandardScaler().fit(F[tr]); lr=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced").fit(s.transform(F[tr]),y[tr])
        oof[te]=lr.predict_proba(s.transform(F[te]))[:,1]
    return oof
rows=[]; PAT={}
for ename,(enc,mean,std) in ENC.items():
    for cname,cond in COND.items():
        t0=time.time(); F=embed(enc,mean,std,cond); oof=probe_oof(F)
        g=pd.DataFrame({"pid":pid,"y":y,"p":oof}).groupby("pid").agg(y=("y","max"),p=("p","mean")); PAT[(ename,cname)]=g
        au=safe_auroc(g.y,g.p); _,lo,hi=cluster_boot_ci(g.index.values,g.y.values,g.p.values,safe_auroc,20000,913)
        rows.append(dict(encoder=ename,condition=cname,dim=F.shape[1],auroc=round(au,4),lo=round(lo,4),hi=round(hi,4)))
        print(f"  {ename:18s} {cname:22s} {au:.4f} ({lo:.3f}-{hi:.3f}) [{time.time()-t0:.0f}s]",flush=True)
        if ename=="dinov2_vitb14": g.reset_index().to_csv(os.path.join(DER,f"item9_patientscores_dinov2_{cname}_postop.csv"),index=False)
out=pd.DataFrame(rows); out.to_csv(os.path.join(DER,"foundation_probes_postop.csv"),index=False)
# paired contrasts, DINOv2, same five pairs as the published family
pairs=[("background_only","wound_only"),("sham_background_only","wound_only"),("sham_wound_only","wound_only"),("full_image","wound_only"),("background_only","full_image")]
pids=sorted(set(pid)); yv=PAT[("dinov2_vitb14","full_image")].loc[pids,"y"].values; rng=np.random.default_rng(913); cr=[]
for a,b in pairs:
    A=PAT[("dinov2_vitb14",a)].loc[pids,"p"].values; B=PAT[("dinov2_vitb14",b)].loc[pids,"p"].values; reps=[]
    for _ in range(20000):
        ii=rng.integers(0,len(pids),len(pids))
        if len(np.unique(yv[ii]))<2: continue
        reps.append(safe_auroc(yv[ii],A[ii])-safe_auroc(yv[ii],B[ii]))
    reps=np.array(reps); d=safe_auroc(yv,A)-safe_auroc(yv,B); p=2*min((reps<=0).mean(),(reps>=0).mean())
    cr.append(dict(contrast=f"{a} - {b}",d=round(d,4),lo=round(np.percentile(reps,2.5),4),hi=round(np.percentile(reps,97.5),4),p=round(float(p),4)))
c=pd.DataFrame(cr).sort_values("p").reset_index(drop=True); m=len(c); c["holm_p"]=[round(min(1.0,max(c.p[:i+1]*(m-np.arange(i+1)))),4) for i in range(m)]
c.to_csv(os.path.join(DER,"foundation_paired_contrasts_postop.csv"),index=False); print(c.to_string(index=False),flush=True)
# rank agreement with the fine-tuned (3-seed mean) ordering on this frame
ma=pd.read_csv(os.path.join(DER,"masking_absolute.csv")).set_index("condition").seed_mean
order=["Full image","True wound-only","True background-only","Circularly shifted wound-size region","Complement of shifted region"]
ft=[ma[o] for o in order]; cond_order=["full_image","wound_only","background_only","sham_wound_only","sham_background_only"]
sp=[dict(encoder=e,spearman_vs_finetuned=round(spearmanr(ft,[out[(out.encoder==e)&(out.condition==c)].auroc.item() for c in cond_order]).correlation,3)) for e in ENC]
pd.DataFrame(sp).to_csv(os.path.join(DER,"foundation_rank_agreement_postop.csv"),index=False); print(sp,flush=True)
print("FOUNDATION_POSTOP_DONE",flush=True)
