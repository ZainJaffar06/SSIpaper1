#!/usr/bin/env python
"""Task 5: END-TO-END fine-tunes on identical patient-grouped folds (smartphone arm).
EfficientNet-B0 + ViT-B/16, published fold map, OOF predictions, + EffB0
training-size learning curve. Images decoded once to an in-memory uint8 cache.
Seeds fixed; folds = StratifiedGroupKFold(5, shuffle, seed=20260807) on patients."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision.models as M
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "base"); OUT = os.path.join(HERE, "out", "finetune")
os.makedirs(OUT, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(20260807); np.random.seed(20260807)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ---------- decode once to uint8 cache ----------
CACHE = os.path.join(OUT, "sp224_uint8.npy")
d = pd.read_csv(os.path.join(BASE, "ssi_smartphone.csv")).drop_duplicates("materialized_path").reset_index(drop=True)
meta = d[["materialized_path", "pid", "y_true", "pod"]].copy()
meta.to_csv(os.path.join(OUT, "sp224_meta.csv"), index=False)
if os.path.exists(CACHE):
    X8 = np.load(CACHE)
    assert len(X8) == len(meta), "cache/meta mismatch"
    print(f"cache loaded {X8.shape}", flush=True)
else:
    from PIL import Image
    try:
        import pillow_heif; pillow_heif.register_heif_opener()
    except Exception: pass
    from multiprocessing.dummy import Pool   # threads: safe under macOS spawn; decode is C-bound
    def dec(p):
        try:
            im = Image.open(p).convert("RGB").resize((224, 224))
            return np.asarray(im, dtype=np.uint8)
        except Exception:
            return np.zeros((224, 224, 3), np.uint8)
    t0 = time.time()
    with Pool(6) as pool:
        arrs = pool.map(dec, meta.materialized_path.tolist(), chunksize=16)
    X8 = np.stack(arrs)
    np.save(CACHE, X8)
    print(f"decoded {X8.shape} in {time.time()-t0:.0f}s", flush=True)

y_all = meta.y_true.values.astype(np.int64)
pid_all = meta.pid.values
# ---------- folds (published) ----------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260807)
folds = np.full(len(y_all), -1)
for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(y_all)), y_all, pid_all)):
    folds[te] = k
meta["fold"] = folds
meta.to_csv(os.path.join(OUT, "finetune_fold_map.csv"), index=False)
for k in range(5):
    assert len(set(pid_all[folds == k]) & set(pid_all[folds != k])) == 0
print("folds published + patient-disjoint verified", flush=True)

def make_model(arch):
    if arch == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        lr = 1e-4
    elif arch == "vit_b_16":
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
        lr = 3e-5
    else: raise ValueError(arch)
    return m.to(DEV), lr

def run_fold(arch, tr_idx, te_idx, epochs=12, bs=32):
    if arch == "vit_b_16": bs = 8    # MPS ~9GB cap + tight disk (swap): keep ViT footprint small
    model, lr = make_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor([1.0, float((y_all[tr_idx] == 0).sum() / max((y_all[tr_idx] == 1).sum(), 1))]).to(DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    Xtr = torch.from_numpy(X8[tr_idx]); ytr = torch.from_numpy(y_all[tr_idx])
    Xte = torch.from_numpy(X8[te_idx])
    n = len(tr_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(DEV).float().permute(0, 3, 1, 2) / 255.0
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, [3])
            if torch.rand(1).item() < 0.3: xb = torch.flip(xb, [2])
            xb = ((xb - MEAN.to(DEV)) / STD.to(DEV)).contiguous()
            opt.zero_grad()
            loss = crit(model(xb), ytr[idx].to(DEV))
            loss.backward(); opt.step()
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

def patient_auc(pids, y, p):
    df = pd.DataFrame({"pid": pids, "y": y, "p": p}).groupby("pid").agg(y=("y", "max"), p=("p", "mean"))
    return roc_auc_score(df.y, df.p)

# ---------- stage 1: full fine-tunes ----------
for arch in ["efficientnet_b0", "vit_b_16"]:
    fp = os.path.join(OUT, f"{arch}_oof.csv")
    if os.path.exists(fp): print("skip", arch, flush=True); continue
    t0 = time.time(); oof = np.full(len(y_all), np.nan)
    for k in range(5):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        oof[te] = run_fold(arch, tr, te)
        print(f"{arch} fold {k} done ({time.time()-t0:.0f}s)", flush=True)
    mm = meta.copy(); mm["proba_ft"] = oof
    mm.to_csv(fp, index=False)
    print(f"{arch} OOF image AUROC {roc_auc_score(y_all, oof):.4f} patient {patient_auc(pid_all, y_all, oof):.4f} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

# ---------- stage 2: EffB0 learning curve (fractions of TRAINING patients) ----------
lc_fp = os.path.join(OUT, "learning_curve.csv")
if not os.path.exists(lc_fp):
    rows = []
    rng = np.random.default_rng(20260807)
    for frac in [0.33, 0.66, 1.0]:
        oof = np.full(len(y_all), np.nan); t0 = time.time()
        for k in range(5):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            if frac < 1.0:
                trp = pd.unique(pid_all[tr])
                pos_p = [p for p in trp if y_all[pid_all == p].max() == 1]
                neg_p = [p for p in trp if y_all[pid_all == p].max() == 0]
                keep = set(rng.choice(pos_p, max(2, int(len(pos_p) * frac)), replace=False)) | \
                       set(rng.choice(neg_p, int(len(neg_p) * frac), replace=False))
                tr = tr[[pid_all[i] in keep for i in tr]]
            oof[te] = run_fold("efficientnet_b0", tr, te)
        rows.append(dict(frac=frac, n_train_last=int(len(tr)), image_auroc=round(float(roc_auc_score(y_all, oof)), 4),
                         patient_auroc=round(float(patient_auc(pid_all, y_all, oof)), 4),
                         minutes=round((time.time() - t0) / 60, 1)))
        pd.DataFrame(rows).to_csv(lc_fp, index=False)
        print("LC", rows[-1], flush=True)
print("FINETUNE_ALL_DONE", flush=True)
