#!/usr/bin/env python
"""Task 8: RETRAINED wound-only / background-only models across all 5 folds
(upgrades the inference-masking ablation: does the WOUND REGION CONTAIN the
signal, not just does the model use it). Annotated smartphone subset; EffB0
end-to-end per fold with identical hyperparameters to the main fine-tune;
masks are the exact incision-target geometry used by the audited Grad-CAM
pipeline (incision_line* linestrips width-10 + wound_bed polygons, 11px
dilation tolerance band), scaled to the same plain 224x224 resize as the
decoded tensor cache."""
import os, io, json, zipfile, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import torch, torch.nn as nn
import torchvision.models as M
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "out", "finetune"); OUT = os.path.join(HERE, "out", "paper1")
MAN = "/Volumes/Backup Plus/SSI_CNN_PUBLISHABLE_2026_08_07/manifests/locked_primary_manifest.csv"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(20260807); np.random.seed(20260807)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
INCISION_LABELS = ("incision_line", "incision_line_center", "incision_line_left",
                   "incision_line_right", "wound_bed")

meta = pd.read_csv(os.path.join(HERE, "out", "paper1", "MASTER_map_1948_seed42.csv"))
meta = meta.rename(columns={"y_true": "y_true"})  # frozen seed-42 patient folds (PI round-2)
OUT = os.path.join(HERE, "out", "master"); os.makedirs(OUT, exist_ok=True)
X8 = np.load(os.path.join(FT, "sp224_uint8.npy"))
man = pd.read_csv(MAN, encoding="utf-8-sig", low_memory=False)
man = man.drop_duplicates("materialized_path")
mm = meta.merge(man[["materialized_path", "best_source_type", "best_source_locator",
                     "best_source_member"]], on="materialized_path", how="left")
assert len(mm) == len(meta)

def polygon_mask_224(row):
    try:
        if row.best_source_type != "annotation_zip_embedded": return None
        with zipfile.ZipFile(row.best_source_locator) as zf:
            data = json.loads(zf.read(row.best_source_member))
        W, H = data.get("imageWidth"), data.get("imageHeight")
        shapes = data.get("shapes") or []
        if not shapes or not W or not H: return None
        mask = Image.new("L", (224, 224), 0)
        dr = ImageDraw.Draw(mask)
        drew = False
        for s in shapes:
            lab = str(s.get("label", "")).strip().lower()
            if lab not in INCISION_LABELS: continue
            pts = s.get("points") or []
            if len(pts) < 2: continue
            sc = [(p[0] * 224.0 / W, p[1] * 224.0 / H) for p in pts]
            if s.get("shape_type") in ("polygon",) and len(sc) >= 3:
                dr.polygon(sc, fill=1)
            else:
                dr.line(sc, fill=1, width=10)
            drew = True
        if not drew: return None
        a = np.array(mask, dtype=np.uint8)
        if a.sum() == 0: return None
        # 11x11 square dilation tolerance band (PIL MaxFilter == square dilation)
        a = np.array(Image.fromarray(a * 255).filter(ImageFilter.MaxFilter(11))) > 0
        return a
    except Exception:
        return None

masks, keep = {}, []
t0 = time.time()
for i, row in mm.iterrows():
    mk = polygon_mask_224(row)
    if mk is not None:
        masks[i] = mk; keep.append(i)
keep = np.array(keep)
sub = mm.loc[keep].reset_index(drop=True)
y_sub = sub.y_true.values.astype(np.int64); pid_sub = sub.pid.values
folds_sub = sub.fold.values
n_pos_pts = sub[sub.y_true == 1].pid.nunique()
print(f"annotated subset: {len(sub)} imgs | {sub.pid.nunique()} pts | {n_pos_pts} pos pts "
      f"({time.time()-t0:.0f}s mask build)", flush=True)

M_arr = np.stack([masks[i] for i in keep])          # (n,224,224) bool
Xs = X8[keep]
X_wound = Xs * M_arr[..., None]                     # background zeroed
X_backg = Xs * (~M_arr)[..., None]                  # wound zeroed

def run_fold(Xall, tr_idx, te_idx, epochs=12, bs=32):
    m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m = m.to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    w = torch.tensor([1.0, float((y_sub[tr_idx] == 0).sum() / max((y_sub[tr_idx] == 1).sum(), 1))]).to(DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    Xtr = torch.from_numpy(Xall[tr_idx]); ytr = torch.from_numpy(y_sub[tr_idx])
    Xte = torch.from_numpy(Xall[te_idx])
    n = len(tr_idx)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(DEV).float().permute(0, 3, 1, 2) / 255.0
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, [3])
            if torch.rand(1).item() < 0.3: xb = torch.flip(xb, [2])
            xb = ((xb - MEAN.to(DEV)) / STD.to(DEV)).contiguous()
            opt.zero_grad(); crit(m(xb), ytr[idx].to(DEV)).backward(); opt.step()
    m.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 64):
            xb = Xte[i:i + 64].to(DEV).float().permute(0, 3, 1, 2) / 255.0
            xb = ((xb - MEAN.to(DEV)) / STD.to(DEV)).contiguous()
            probs.append(torch.softmax(m(xb), 1)[:, 1].cpu().numpy())
    del m
    if DEV == "mps": torch.mps.empty_cache()
    return np.concatenate(probs)

conds = {"full_image": Xs, "wound_only": X_wound, "background_only": X_backg}
oofs = {}
for name, Xc in conds.items():
    fp = os.path.join(OUT, f"task8_oof_{name}.csv")
    if os.path.exists(fp):
        oofs[name] = pd.read_csv(fp).proba_masked.values
        print("skip", name, flush=True); continue
    t0 = time.time(); oof = np.full(len(y_sub), np.nan)
    for k in sorted(np.unique(folds_sub)):
        tr, te = np.where(folds_sub != k)[0], np.where(folds_sub == k)[0]
        if len(np.unique(y_sub[tr])) < 2 or len(np.unique(y_sub[te])) < 1: continue
        oof[te] = run_fold(Xc, tr, te)
        print(f"{name} fold {int(k)} done ({time.time()-t0:.0f}s)", flush=True)
    oofs[name] = oof
    o = sub[["pid", "materialized_path", "y_true", "fold"]].copy(); o["proba_masked"] = oof
    o.to_csv(fp, index=False)
    print(f"{name} image AUROC {roc_auc_score(y_sub, oof):.4f}", flush=True)

RNG = np.random.default_rng(20260807)
def pat_frame(s):
    return pd.DataFrame({"pid": pid_sub, "y": y_sub, "s": s}).groupby("pid").agg(
        y=("y", "max"), s=("s", "mean")).reset_index()

def boot(y, s, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = roc_auc_score(y, s)
    for _ in range(nb):
        b = RNG.choice(idx, len(idx), True)
        if len(np.unique(y[b])) < 2: continue
        v.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

def paired(y, a, b, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    for _ in range(2000):
        s = RNG.choice(idx, len(idx), True)
        if len(np.unique(y[s])) < 2: continue
        v.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1)), 4)

rows, drows = [], []
pf = {n: pat_frame(s) for n, s in oofs.items()}
for n in conds:
    ia, ilo, ihi = boot(y_sub, oofs[n])
    pa, plo, phi = boot(pf[n].y.values, pf[n].s.values)
    rows.append(dict(condition=n, n_img=len(y_sub), n_pts=len(pf[n]), n_pos_pts=n_pos_pts,
                     image_auroc=ia, img_lo=ilo, img_hi=ihi,
                     patient_auroc=pa, pat_lo=plo, pat_hi=phi))
for n in ("wound_only", "background_only"):
    o, lo, hi, p = paired(pf[n].y.values, pf[n].s.values, pf["full_image"].s.values)
    drows.append(dict(contrast=f"{n}_minus_full(patient)", delta=o, lo=lo, hi=hi, p_two_sided=p))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "task8_masked_retrain_performance.csv"), index=False)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "task8_masked_retrain_deltas.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False), flush=True)
print(pd.DataFrame(drows).to_string(index=False), flush=True)
print("TASK8_DONE", flush=True)
