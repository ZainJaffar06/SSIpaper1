#!/usr/bin/env python
"""Postoperative-only re-run (Dr. Kewalramani decision, 2026-10-21).

Removes the 598 intraoperative photographs (280 pre-closure, 318 post-closure) and
the 9 with no derivable day. The 41 day-of-surgery photographs taken after leaving
the operating room are RETAINED as postoperative.

Frame: 1,341 photographs | 171 patients | 13 final-label SSI patients.
Labels come from final_ssi_list.txt only. Folds are the locked per-patient
assignment restricted to this frame, so results stay comparable to the published
runs; no re-stratification. Fold 1 holds 0 events on this frame -- recorded, not
worked around: each fold-k model still trains on the other four folds (9-13 of the
13 positives) and a 0-event test fold contributes only negatives to the pooled
out-of-fold AUROC.

NEVER reads rgb_intraoperative_scores.csv, results/canonical/, or the y_true column
of MASTER_map_1948_seed42.csv (all carry superseded labels).

MEMORY: this machine runs with ~1.5GB free swap, so nothing large is held resident.
A compact uint8 cache of just the 1,341 frame images is written once and memory
mapped; masking conditions are applied per batch rather than materialised as five
full arrays. Peak footprint ~150MB.

Resumable: completed runs are skipped. Outputs to out/postop_only/.
"""
import os, json, time, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import torch, torch.nn as nn
import torchvision.models as M
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE,"out","postop_only"); os.makedirs(OUT,exist_ok=True)
FT   = os.path.join(HERE,"out","finetune"); REC = os.path.join(HERE,"out","recompute")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
FINAL= set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())

master = pd.read_csv(os.path.join(HERE,"out","paper1","MASTER_map_1948_seed42.csv"))
meta   = pd.read_csv(os.path.join(FT,"sp224_meta.csv"))
assert list(meta.materialized_path)==list(master.materialized_path), "cache/master misaligned"
keep = np.load(os.path.join(REC,"postop_row_mask.npy"))
assert len(keep)==1948 and keep.sum()==1341, "frame mask wrong"

# compact, memory-mapped cache of only the frame images
PC = os.path.join(OUT,"postop_uint8.npy")
if not os.path.exists(PC):
    src = np.load(os.path.join(FT,"sp224_uint8.npy"), mmap_mode="r")
    out = np.lib.format.open_memmap(PC, mode="w+", dtype=np.uint8, shape=(int(keep.sum()),224,224,3))
    g = np.where(keep)[0]
    for i in range(0,len(g),128): out[i:i+128] = src[g[i:i+128]]
    out.flush(); del out, src
    print("compact cache written", flush=True)
X = np.load(PC, mmap_mode="r")

P = master.loc[keep].reset_index(drop=True)
y = P.pid.isin(FINAL).values.astype(np.int64)
pid, fold = P.pid.values, P.fold.values
rp = pd.read_csv("/Users/zainjaffar/Documents/New project/ssi-paper1-information-audit/results/MASTER_map_1948_seed42.csv")
IMG = rp.image_id.values[keep]
print(f"POSTOP FRAME {len(P)} imgs | {P.pid.nunique()} pts | {int(pd.Series(y).groupby(pid).max().sum())} SSI pts",flush=True)
print(f"  events/fold {pd.DataFrame({'p':pid,'y':y,'f':fold}).drop_duplicates('p').groupby('f').y.sum().tolist()}",flush=True)

class Frame:
    """Serves uint8 batches for one condition without materialising it."""
    def __init__(self, idx, mk=None, invert=False, as_mask=False):
        self.idx=np.asarray(idx); self.mk=mk; self.invert=invert; self.as_mask=as_mask
    def __len__(self): return len(self.idx)
    def batch(self, loc):
        if self.as_mask:
            m=self.mk[loc].astype(np.uint8)*255
            return np.repeat(m[...,None],3,axis=3)
        g=np.asarray(X[self.idx[loc]])
        if self.mk is not None:
            m=self.mk[loc]
            if self.invert: m=~m
            g=g*m[...,None]
        return g

def make(arch):
    if arch=="resnet18":
        m=M.resnet18(weights=M.ResNet18_Weights.DEFAULT); m.fc=nn.Linear(m.fc.in_features,2); return m.to(DEV),1e-4,32
    if arch=="efficientnet_b0":
        m=M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1]=nn.Linear(m.classifier[1].in_features,2); return m.to(DEV),1e-4,32
    m=M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads.head=nn.Linear(m.heads.head.in_features,2); return m.to(DEV),3e-5,8

def run_fold(arch, fr, yv, tr, te, epochs=12):
    model,lr,bs = make(arch)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    w=torch.tensor([1.0,float((yv[tr]==0).sum()/max((yv[tr]==1).sum(),1))]).to(DEV)
    crit=nn.CrossEntropyLoss(weight=w); n=len(tr)
    for _ in range(epochs):
        model.train(); perm=np.random.permutation(n)
        for i in range(0,n,bs):
            loc=tr[perm[i:i+bs]]
            xb=torch.from_numpy(fr.batch(loc)).to(DEV).float().permute(0,3,1,2)/255.0
            if torch.rand(1).item()<0.5: xb=torch.flip(xb,[3])
            if torch.rand(1).item()<0.3: xb=torch.flip(xb,[2])
            xb=((xb-MEAN.to(DEV))/STD.to(DEV)).contiguous()
            yb=torch.from_numpy(yv[loc]).to(DEV)
            opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()
        if DEV=="mps": torch.mps.empty_cache()
    model.eval(); pr=[]
    with torch.no_grad():
        for i in range(0,len(te),64):
            loc=te[i:i+64]
            xb=torch.from_numpy(fr.batch(loc)).to(DEV).float().permute(0,3,1,2)/255.0
            xb=((xb-MEAN.to(DEV))/STD.to(DEV)).contiguous()
            pr.append(torch.softmax(model(xb),1)[:,1].cpu().numpy())
    del model
    if DEV=="mps": torch.mps.empty_cache()
    return np.concatenate(pr)

def pat_auc(p_,y_,s_):
    g=pd.DataFrame({"pid":p_,"y":y_,"p":s_}).groupby("pid").agg(y=("y","max"),p=("p","mean"))
    return float(roc_auc_score(g.y,g.p)) if g.y.nunique()>1 else float("nan")

def cv(tag, arch, fr, yv, p_, f_, img_, seed, frac=1.0):
    fp=os.path.join(OUT,f"{tag}.csv")
    if os.path.exists(fp): print("skip",tag,flush=True); return
    torch.manual_seed(seed); np.random.seed(seed); rng=np.random.default_rng(seed)
    t0=time.time(); oof=np.full(len(yv),np.nan)
    for k in sorted(np.unique(f_)):
        tr=np.where(f_!=k)[0]; te=np.where(f_==k)[0]
        if frac<1.0:
            tp=np.unique(p_[tr]); sel=set(rng.choice(tp,max(2,int(round(len(tp)*frac))),replace=False))
            tr=tr[np.isin(p_[tr],list(sel))]
        if len(np.unique(yv[tr]))<2:
            print(f"  {tag} fold {int(k)}: single-class training split, skipped",flush=True); continue
        oof[te]=run_fold(arch,fr,yv,tr,te)
        print(f"  {tag} fold {int(k)} done ({time.time()-t0:.0f}s)",flush=True)
    pd.DataFrame({"pid":p_,"image_id":img_,"fold":f_,"y_true":yv,"proba":oof}).to_csv(fp,index=False)
    m=~np.isnan(oof)
    print(f"{tag}: image {roc_auc_score(yv[m],oof[m]):.4f} patient {pat_auc(p_[m],yv[m],oof[m]):.4f} "
          f"({(time.time()-t0)/60:.1f} min)",flush=True)

ALL=np.arange(len(P))
# ---------------- STAGE 1 ----------------
print("\n=== STAGE 1: architectures x seeds (primary first) ===",flush=True)
cv("resnet18_seed20260807","resnet18",Frame(ALL),y,pid,fold,IMG,20260807)
for arch in ["resnet18","efficientnet_b0","vit_b_16"]:
    for seed in (20260807,42,7):
        cv(f"{arch}_seed{seed}",arch,Frame(ALL),y,pid,fold,IMG,seed)

# ---------------- STAGE 2: masks ----------------
print("\n=== STAGE 2: masking ===",flush=True)
MCACHE=os.path.join(OUT,"postop_masks.npz")
if os.path.exists(MCACHE):
    z=np.load(MCACHE); Mk=z["Mk"]; kidx=z["kidx"]
else:
    MAN="/Volumes/Backup Plus/SSI_CNN_PUBLISHABLE_2026_08_07/manifests/locked_primary_manifest.csv"
    LAB=("incision_line","incision_line_center","incision_line_left","incision_line_right","wound_bed")
    man=pd.read_csv(MAN,encoding="utf-8-sig",low_memory=False).drop_duplicates("materialized_path")
    mm=P.merge(man[["materialized_path","best_source_type","best_source_locator","best_source_member"]],
               on="materialized_path",how="left")
    def mask224(row):
        try:
            if row.best_source_type!="annotation_zip_embedded": return None
            with zipfile.ZipFile(row.best_source_locator) as zf: data=json.loads(zf.read(row.best_source_member))
            W,H=data.get("imageWidth"),data.get("imageHeight"); shapes=data.get("shapes") or []
            if not shapes or not W or not H: return None
            m=Image.new("L",(224,224),0); dr=ImageDraw.Draw(m); drew=False
            for s in shapes:
                if str(s.get("label","")).strip().lower() not in LAB: continue
                pts=s.get("points") or []
                if len(pts)<2: continue
                sc=[(p[0]*224.0/W,p[1]*224.0/H) for p in pts]
                if s.get("shape_type")=="polygon" and len(sc)>=3: dr.polygon(sc,fill=1)
                else: dr.line(sc,fill=1,width=10)
                drew=True
            if not drew: return None
            a=np.array(m,dtype=np.uint8)
            return (np.array(Image.fromarray(a*255).filter(ImageFilter.MaxFilter(11)))>0) if a.sum() else None
        except Exception: return None
    ms,kk=[],[]
    for i,row in mm.iterrows():
        v=mask224(row)
        if v is not None: ms.append(v); kk.append(i)
    Mk=np.stack(ms); kidx=np.array(kk)
    np.savez_compressed(MCACHE,Mk=Mk,kidx=kidx)
    print(f"masks built {Mk.shape}",flush=True)
ys=y[kidx]; ps=pid[kidx]; fs=fold[kidx]; ims=IMG[kidx]
print(f"masking subset: {len(kidx)} imgs | {len(set(ps))} pts | {int(pd.Series(ys).groupby(ps).max().sum())} SSI pts",flush=True)
print(f"  events/fold {pd.DataFrame({'p':ps,'y':ys,'f':fs}).drop_duplicates('p').groupby('f').y.sum().tolist()}",flush=True)
rng=np.random.default_rng(20260807)
SH=np.stack([np.roll(Mk[i],(int(rng.integers(56,168)),int(rng.integers(56,168))),axis=(0,1)) for i in range(len(Mk))])
np.savez_compressed(os.path.join(OUT,"postop_sham.npz"),SH=SH)
SUB=kidx
for seed in (20260807,42,7):
    cv(f"mask_full_image_seed{seed}","efficientnet_b0",Frame(SUB),ys,ps,fs,ims,seed)
    cv(f"mask_wound_only_seed{seed}","efficientnet_b0",Frame(SUB,Mk,False),ys,ps,fs,ims,seed)
    cv(f"mask_background_only_seed{seed}","efficientnet_b0",Frame(SUB,Mk,True),ys,ps,fs,ims,seed)
    cv(f"mask_sham_wound_only_seed{seed}","efficientnet_b0",Frame(SUB,SH,False),ys,ps,fs,ims,seed)
    cv(f"mask_sham_background_only_seed{seed}","efficientnet_b0",Frame(SUB,SH,True),ys,ps,fs,ims,seed)

# ---------------- STAGE 3: mask-as-image ----------------
print("\n=== STAGE 3: binary mask as image ===",flush=True)
for seed in (20260807,42,7):
    cv(f"maskimage_seed{seed}","efficientnet_b0",Frame(SUB,Mk,False,as_mask=True),ys,ps,fs,ims,seed)

# ---------------- STAGE 4: learning curve ----------------
print("\n=== STAGE 4: learning curve ===",flush=True)
for frac in (0.333,0.667):
    for seed in (20260807,42,7):
        cv(f"lc_{frac}_seed{seed}","efficientnet_b0",Frame(ALL),y,pid,fold,IMG,seed,frac=frac)

print("\nPOSTOP_RETRAIN_DONE",flush=True)
