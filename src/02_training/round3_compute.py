#!/usr/bin/env python
"""Round-3 blocking compute, all on the frozen master map (MASTER_map_1948_seed42.csv):
(A) ViT-B/16 seeds 42 and 7 (completes 3-seed distributions for every architecture).
(B) Multi-seed learning curve: EffB0, training fractions 1/3 and 2/3 x seeds
    {20260807,42,7}; the 1.0 points are the existing master EffB0 seed runs.
(C) Masking NEGATIVE CONTROL: sham masks = each image's own incision mask randomly
    relocated (np.roll, fixed RNG; area/shape preserved, position destroyed).
    Conditions sham_wound_only / sham_background_only, identical protocol+folds.
Every run writes an OOF CSV; summaries with mean±SD across seeds."""
import os, time, json, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import torch, torch.nn as nn
import torchvision.models as M
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "master"); FT = os.path.join(HERE, "out", "finetune")
os.makedirs(OUT, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

master = pd.read_csv(os.path.join(HERE, "out", "paper1", "MASTER_map_1948_seed42.csv"))
meta = pd.read_csv(os.path.join(FT, "sp224_meta.csv"))
X8 = np.load(os.path.join(FT, "sp224_uint8.npy"))
assert list(meta.materialized_path) == list(master.materialized_path)
y_all = master.y_true.values.astype(np.int64)
pid_all = master.pid.values
folds = master.fold.values

def make_model(arch):
    if arch == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2); lr = 1e-4; bs = 32
    elif arch == "vit_b_16":
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2); lr = 3e-5; bs = 8
    else: raise ValueError(arch)
    return m.to(DEV), lr, bs

def run_fold(arch, X, yv, tr_idx, te_idx, epochs=12):
    model, lr, bs = make_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor([1.0, float((yv[tr_idx] == 0).sum() / max((yv[tr_idx] == 1).sum(), 1))]).to(DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    Xtr = torch.from_numpy(X[tr_idx]); ytr = torch.from_numpy(yv[tr_idx])
    Xte = torch.from_numpy(X[te_idx]); n = len(tr_idx)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(DEV).float().permute(0, 3, 1, 2) / 255.0
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, [3])
            if torch.rand(1).item() < 0.3: xb = torch.flip(xb, [2])
            xb = ((xb - MEAN.to(DEV)) / STD.to(DEV)).contiguous()
            opt.zero_grad(); crit(model(xb), ytr[idx].to(DEV)).backward(); opt.step()
        if DEV == "mps": torch.mps.empty_cache()
    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 64):
            xb = Xte[i:i + 64].to(DEV).float().permute(0, 3, 1, 2) / 255.0
            xb = ((xb - MEAN.to(DEV)) / STD.to(DEV)).contiguous()
            probs.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    del model
    if DEV == "mps": torch.mps.empty_cache()
    return np.concatenate(probs)

def pat_auc(pids, yv, p):
    df = pd.DataFrame({"pid": pids, "y": yv, "p": p}).groupby("pid").agg(y=("y", "max"), p=("p", "mean"))
    return float(roc_auc_score(df.y, df.p))

# ---------- (A) ViT seeds 42, 7 ----------
for seed in (42, 7):
    fp = os.path.join(OUT, f"vit_b_16_seed{seed}_oof.csv")
    if os.path.exists(fp): print("skip vit", seed, flush=True); continue
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time(); oof = np.full(len(y_all), np.nan)
    for k in sorted(np.unique(folds)):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        oof[te] = run_fold("vit_b_16", X8, y_all, tr, te)
        print(f"vit s{seed} fold {int(k)} done ({time.time()-t0:.0f}s)", flush=True)
    o = master[["pid", "materialized_path", "y_true", "fold"]].copy(); o["proba"] = oof
    o.to_csv(fp, index=False)
    print(f"vit_b_16 seed{seed}: image {roc_auc_score(y_all, oof):.4f} patient {pat_auc(pid_all, y_all, oof):.4f}", flush=True)

# ---------- (B) multi-seed learning curve ----------
lc_fp = os.path.join(OUT, "learning_curve_multiseed.csv")
if not os.path.exists(lc_fp):
    rows = []
    for seed in (20260807, 42, 7):
        # 1.0 point = existing master EffB0 run
        f = pd.read_csv(os.path.join(OUT, f"efficientnet_b0_seed{seed}_oof.csv"))
        rows.append(dict(frac=1.0, seed=seed, image_auroc=round(float(roc_auc_score(y_all, f.proba)), 4),
                         patient_auroc=round(pat_auc(pid_all, y_all, f.proba.values), 4)))
        for frac in (1/3, 2/3):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            oof = np.full(len(y_all), np.nan); t0 = time.time()
            for k in sorted(np.unique(folds)):
                tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
                trp = pd.unique(pid_all[tr])
                posp = [p for p in trp if y_all[pid_all == p].max() == 1]
                negp = [p for p in trp if y_all[pid_all == p].max() == 0]
                keep = set(rng.choice(posp, max(2, int(round(len(posp) * frac))), replace=False)) | \
                       set(rng.choice(negp, int(round(len(negp) * frac)), replace=False))
                tr2 = tr[[pid_all[i] in keep for i in tr]]
                oof[te] = run_fold("efficientnet_b0", X8, y_all, tr2, te)
            rows.append(dict(frac=round(frac, 3), seed=seed,
                             image_auroc=round(float(roc_auc_score(y_all, oof)), 4),
                             patient_auroc=round(pat_auc(pid_all, y_all, oof), 4)))
            pd.DataFrame(rows).to_csv(lc_fp, index=False)
            print(f"LC frac {frac:.2f} seed {seed} done ({(time.time()-t0)/60:.0f} min):", rows[-1], flush=True)
    lc = pd.DataFrame(rows)
    lc.to_csv(lc_fp, index=False)
    s = lc.groupby("frac").patient_auroc.agg(["mean", "std", "min", "max"]).round(4)
    s.to_csv(os.path.join(OUT, "learning_curve_multiseed_summary.csv"))
    print(s.to_string(), flush=True)

# ---------- (C) sham-mask negative control ----------
MAN = "/Volumes/Backup Plus/SSI_CNN_PUBLISHABLE_2026_08_07/manifests/locked_primary_manifest.csv"
INCISION_LABELS = ("incision_line", "incision_line_center", "incision_line_left",
                   "incision_line_right", "wound_bed")
sham_fp = os.path.join(OUT, "sham_mask_summary.csv")
if not os.path.exists(sham_fp):
    man = pd.read_csv(MAN, encoding="utf-8-sig", low_memory=False).drop_duplicates("materialized_path")
    mm = master.merge(man[["materialized_path", "best_source_type", "best_source_locator",
                           "best_source_member"]], on="materialized_path", how="left")
    def polygon_mask_224(row):
        try:
            if row.best_source_type != "annotation_zip_embedded": return None
            with zipfile.ZipFile(row.best_source_locator) as zf:
                data = json.loads(zf.read(row.best_source_member))
            W, H = data.get("imageWidth"), data.get("imageHeight")
            shapes = data.get("shapes") or []
            if not shapes or not W or not H: return None
            mask = Image.new("L", (224, 224), 0); dr = ImageDraw.Draw(mask); drew = False
            for s in shapes:
                lab = str(s.get("label", "")).strip().lower()
                if lab not in INCISION_LABELS: continue
                pts = s.get("points") or []
                if len(pts) < 2: continue
                sc = [(p[0] * 224.0 / W, p[1] * 224.0 / H) for p in pts]
                if s.get("shape_type") in ("polygon",) and len(sc) >= 3: dr.polygon(sc, fill=1)
                else: dr.line(sc, fill=1, width=10)
                drew = True
            if not drew: return None
            a = np.array(mask, dtype=np.uint8)
            if a.sum() == 0: return None
            return np.array(Image.fromarray(a * 255).filter(ImageFilter.MaxFilter(11))) > 0
        except Exception:
            return None
    masks, keep = {}, []
    for i, row in mm.iterrows():
        mk = polygon_mask_224(row)
        if mk is not None: masks[i] = mk; keep.append(i)
    keep = np.array(keep)
    sub = mm.loc[keep].reset_index(drop=True)
    y_s = sub.y_true.values.astype(np.int64); pid_s = sub.pid.values; fold_s = sub.fold.values
    rng = np.random.default_rng(20260807)
    SH = np.stack([np.roll(masks[i], (int(rng.integers(56, 168)), int(rng.integers(56, 168))), axis=(0, 1))
                   for i in keep])
    Xs = X8[keep]
    conds = {"sham_wound_only": Xs * SH[..., None], "sham_background_only": Xs * (~SH)[..., None]}
    print(f"sham subset: {len(sub)} imgs | {sub.pid.nunique()} pts | {sub[sub.y_true==1].pid.nunique()} pos pts", flush=True)
    rows = []
    for name, Xc in conds.items():
        fp = os.path.join(OUT, f"task8_oof_{name}.csv")
        if os.path.exists(fp):
            oof = pd.read_csv(fp).proba_masked.values
        else:
            torch.manual_seed(20260807); np.random.seed(20260807)
            t0 = time.time(); oof = np.full(len(y_s), np.nan)
            for k in sorted(np.unique(fold_s)):
                tr, te = np.where(fold_s != k)[0], np.where(fold_s == k)[0]
                oof[te] = run_fold("efficientnet_b0", Xc, y_s, tr, te)
                print(f"{name} fold {int(k)} done ({time.time()-t0:.0f}s)", flush=True)
            o = sub[["pid", "materialized_path", "y_true", "fold"]].copy(); o["proba_masked"] = oof
            o.to_csv(fp, index=False)
        rows.append(dict(condition=name, image_auroc=round(float(roc_auc_score(y_s, oof)), 4),
                         patient_auroc=round(pat_auc(pid_s, y_s, oof), 4)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(sham_fp, index=False)
print("ROUND3_COMPUTE_DONE", flush=True)
