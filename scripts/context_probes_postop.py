# -*- coding: utf-8 -*-
"""S3 / Figure 3B on the POSTOPERATIVE masking frame (1,296 images, 168 patients, 12 SSI).
Metadata-only probes without wound pixels: POD only; acquisition features (brightness,
contrast, sharpness, colour, edge density) computed from the image; prior-imaging behaviour
(photographs before this one, days since previous, first photograph); and their combination.
Patient-grouped logistic probes via lib.grouped_oof, patient-level mean, AUROC with
patient-clustered CI. Comparator: the three postoperative full-image fine-tuned seeds on the
same patients -> CNN minus combined probe per seed (mean, SD, range) for Figure 3B."""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import ndimage
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from lib import safe_auroc, cluster_boot_ci, grouped_oof
PO=os.path.join(HERE,"out","postop_only"); DER=os.path.join(PO,"derived")
FINAL=set(open(os.path.join(HERE,"final_ssi_list.txt")).read().split())
kidx=np.load(os.path.join(PO,"postop_masks.npz"))["kidx"]; X=np.load(os.path.join(PO,"postop_uint8.npy"),mmap_mode="r")
master=pd.read_csv(os.path.join(HERE,"out","paper1","MASTER_map_1948_seed42.csv")); keep=np.load(os.path.join(HERE,"out","recompute","postop_row_mask.npy"))
rp=pd.read_csv("/Users/zainjaffar/Documents/New project/ssi-paper1-information-audit/results/MASTER_map_1948_seed42.csv")
P=master.loc[keep].reset_index(drop=True); P["image_id"]=rp.image_id.values[keep]; P=P.iloc[kidx].reset_index(drop=True)
pf=pd.read_csv(os.path.join(HERE,"out","recompute","per_photo_frame_1948.csv"))[["image_id","day_after_surgery"]]
P=P.merge(pf,on="image_id",how="left"); y=P.pid.isin(FINAL).values.astype(int); pid=P.pid.values; N=len(P)
print(f"frame {N} imgs | {len(set(pid))} pts | {int(pd.Series(y).groupby(pid).max().sum())} SSI",flush=True)
# ---- acquisition features from pixels ----
acq=np.zeros((N,10),np.float32)
for i in range(N):
    g=np.asarray(X[kidx[i]]).astype(np.float32); gray=g.mean(2)
    lap=ndimage.laplace(gray); gx=ndimage.sobel(gray,1); gy=ndimage.sobel(gray,0); mag=np.hypot(gx,gy)
    acq[i]=[gray.mean(),gray.std(),lap.var(),np.abs(lap).mean(),g[...,0].mean(),g[...,1].mean(),g[...,2].mean(),
            (g.max(2)-g.min(2)).mean(),mag.mean(),(mag>mag.mean()+2*mag.std()).mean()]
# ---- behaviour features (postoperative photographs only, as the frame is) ----
P["_i"]=np.arange(N); beh=np.zeros((N,4),np.float32)
for q,g in P.sort_values(["pid","day_after_surgery"]).groupby("pid"):
    days=g.day_after_surgery.values; idx=g._i.values
    for j,(d,i) in enumerate(zip(days,idx)):
        beh[i]=[d, j, (d-days[j-1]) if j>0 else 0.0, 1.0 if j==0 else 0.0]
podo=beh[:,[0]]
probes={"pod_only":podo,"acquisition":acq,"behavior_past_only":beh,"acquisition+behavior+pod":np.column_stack([acq,beh])}
rows=[]; PS={}
for name,Xp in probes.items():
    oof=grouped_oof(Xp,y,pid,C=1.0); g=pd.DataFrame({"pid":pid,"y":y,"p":oof}).groupby("pid").agg(y=("y","max"),p=("p","mean")); PS[name]=g
    au=safe_auroc(g.y,g.p); _,lo,hi=cluster_boot_ci(g.index.values,g.y.values,g.p.values,safe_auroc,20000,913)
    rows.append(dict(probe=name,image_auroc=round(safe_auroc(y,oof),4),patient_auroc=round(au,4),lo=round(lo,4),hi=round(hi,4)))
    print(f"  {name:28s} patient {au:.4f} ({lo:.3f}-{hi:.3f})",flush=True)
# ---- comparator: postop full-image fine-tuned seeds on the same patients ----
pids=sorted(set(pid)); yv=PS["pod_only"].loc[pids,"y"].values; cnn={}
for s in (20260807,42,7):
    d=pd.read_csv(os.path.join(PO,f"mask_full_image_seed{s}.csv")).dropna(subset=["proba"]); d["y"]=d.pid.isin(FINAL).astype(int)
    cnn[s]=d.groupby("pid").agg(y=("y","max"),p=("proba","mean")).loc[pids]
full=np.mean([safe_auroc(yv,cnn[s].p.values) for s in cnn])
for r in rows: r["full_image_3seed_mean"]=round(full,4); r["full_minus_probe"]=round(full-r["patient_auroc"],4)
pd.DataFrame(rows).to_csv(os.path.join(DER,"context_probes_postop.csv"),index=False)
comb=PS["acquisition+behavior+pod"].loc[pids,"p"].values; deltas=[]
for s in cnn:
    deltas.append(dict(seed=s,cnn_auroc=round(safe_auroc(yv,cnn[s].p.values),4),combined_probe_auroc=round(safe_auroc(yv,comb),4),cnn_minus_probe=round(safe_auroc(yv,cnn[s].p.values)-safe_auroc(yv,comb),4)))
dd=pd.DataFrame(deltas); v=dd.cnn_minus_probe.values
dd.to_csv(os.path.join(DER,"seedaware_probe_deltas_postop.csv"),index=False)
summ=dict(mean=round(v.mean(),4),sd=round(v.std(ddof=1),4),lo=round(v.min(),4),hi=round(v.max(),4))
pd.DataFrame([summ]).to_csv(os.path.join(DER,"seedaware_probe_delta_summary_postop.csv"),index=False)
# paired CNN(seed 20260807) vs combined probe, patient-clustered
rng=np.random.default_rng(913); A=cnn[20260807].p.values; reps=[]
for _ in range(20000):
    ii=rng.integers(0,len(pids),len(pids))
    if len(np.unique(yv[ii]))<2: continue
    reps.append(safe_auroc(yv[ii],A[ii])-safe_auroc(yv[ii],comb[ii]))
reps=np.array(reps); d0=safe_auroc(yv,A)-safe_auroc(yv,comb); p=2*min((reps<=0).mean(),(reps>=0).mean())
pd.DataFrame([dict(contrast="CNN seed20260807 - combined probe",d=round(d0,4),lo=round(np.percentile(reps,2.5),4),hi=round(np.percentile(reps,97.5),4),p=round(float(p),4))]).to_csv(os.path.join(DER,"cnn_vs_probe_paired_postop.csv"),index=False)
print(pd.DataFrame(rows).to_string(index=False)); print(dd.to_string(index=False)); print("CNN - combined probe, 3-seed:",summ); print(f"paired (seed 20260807): {d0:+.4f} ({np.percentile(reps,2.5):+.4f} to {np.percentile(reps,97.5):+.4f}) P={p:.3f}")
print("CONTEXT_POSTOP_DONE",flush=True)
