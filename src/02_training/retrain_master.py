#!/usr/bin/env python
"""PI round-2, point 1: retrain ResNet18, EfficientNet-B0, ViT-B/16 end-to-end on the
FROZEN master map (1,948 images, locked seed-42 patient folds) under one identical
protocol (12 epochs, AdamW, flips, class-weighted CE). ResNet18 and EffB0 run at
3 training seeds to quantify seed variability; ViT runs one seed (stated). Outputs:
per-model-seed OOF CSVs + summary with paired patient-bootstrap deltas vs ResNet18."""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
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
# cache order must equal master order (both are first-occurrence dedups of the locked frame)
assert list(meta.materialized_path) == list(master.materialized_path), "cache/master order mismatch"
y_all = master.y_true.values.astype(np.int64)
pid_all = master.pid.values
folds = master.fold.values

def make_model(arch):
    if arch == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2); lr = 1e-4; bs = 32
    elif arch == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2); lr = 1e-4; bs = 32
    elif arch == "vit_b_16":
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2); lr = 3e-5; bs = 8
    else: raise ValueError(arch)
    return m.to(DEV), lr, bs

def run_fold(arch, tr_idx, te_idx, epochs=12):
    model, lr, bs = make_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor([1.0, float((y_all[tr_idx] == 0).sum() / max((y_all[tr_idx] == 1).sum(), 1))]).to(DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    Xtr = torch.from_numpy(X8[tr_idx]); ytr = torch.from_numpy(y_all[tr_idx])
    Xte = torch.from_numpy(X8[te_idx]); n = len(tr_idx)
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

def pat_frame(p):
    return pd.DataFrame({"pid": pid_all, "y": y_all, "p": p}).groupby("pid").agg(
        y=("y", "max"), p=("p", "mean")).reset_index()

JOBS = [("resnet18", s) for s in (20260807, 42, 7)] + \
       [("efficientnet_b0", s) for s in (20260807, 42, 7)] + \
       [("vit_b_16", 20260807)]
for arch, seed in JOBS:
    fp = os.path.join(OUT, f"{arch}_seed{seed}_oof.csv")
    if os.path.exists(fp): print("skip", arch, seed, flush=True); continue
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time(); oof = np.full(len(y_all), np.nan)
    for k in sorted(np.unique(folds)):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        oof[te] = run_fold(arch, tr, te)
        print(f"{arch} s{seed} fold {int(k)} done ({time.time()-t0:.0f}s)", flush=True)
    o = master[["pid", "materialized_path", "y_true", "fold"]].copy(); o["proba"] = oof
    o.to_csv(fp, index=False)
    pf = pat_frame(oof)
    print(f"{arch} seed{seed}: image {roc_auc_score(y_all, oof):.4f} patient "
          f"{roc_auc_score(pf.y, pf.p):.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)

# ---- summary: seed variability + paired deltas vs ResNet18 (seed 20260807) ----
RNG = np.random.default_rng(20260807)
def paired(y, a, b, nb=2000):
    idx = np.arange(len(y)); v = []
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    for _ in range(nb):
        s = RNG.choice(idx, len(idx), True)
        if len(np.unique(y[s])) < 2: continue
        v.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1)), 4)

rows, drows = [], []
ref = pat_frame(pd.read_csv(os.path.join(OUT, "resnet18_seed20260807_oof.csv")).proba.values)
lockpat = master.groupby("pid").agg(y=("y_true", "max")).reset_index()
for arch, seed in JOBS:
    f = pd.read_csv(os.path.join(OUT, f"{arch}_seed{seed}_oof.csv"))
    pf = pat_frame(f.proba.values)
    rows.append(dict(model=arch, seed=seed, image_auroc=round(float(roc_auc_score(y_all, f.proba)), 4),
                     patient_auroc=round(float(roc_auc_score(pf.y, pf.p)), 4)))
    if not (arch == "resnet18" and seed == 20260807):
        o, lo, hi, p = paired(ref.y.values, pf.p.values, ref.p.values)
        drows.append(dict(contrast=f"{arch}_s{seed}_minus_resnet18_s20260807(patient)",
                          delta=o, lo=lo, hi=hi, p_two_sided=p))
res = pd.DataFrame(rows)
sv = res.groupby("model").patient_auroc.agg(["mean", "std", "min", "max"]).round(4).reset_index()
res.to_csv(os.path.join(OUT, "master_architectures.csv"), index=False)
sv.to_csv(os.path.join(OUT, "master_seed_variability.csv"), index=False)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "master_paired_deltas.csv"), index=False)
print(res.to_string(index=False), flush=True)
print(sv.to_string(index=False), flush=True)
print("RETRAIN_MASTER_DONE", flush=True)
