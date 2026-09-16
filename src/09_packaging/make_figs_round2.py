#!/usr/bin/env python
"""Round-2 figures. Guards on source CSVs so it can run mid-pipeline and at the end."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1"); MA = os.path.join(HERE, "out", "master")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREEN, RED, GRAY = "#2C6FBB", "#E8873A", "#3B8E5A", "#C44536", "#7A7A7A"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    print("wrote", name, flush=True)

# R2-F1: fold-paired architectures with seed bands
fp = os.path.join(MA, "master_architectures.csv")
if os.path.exists(fp):
    ma = pd.read_csv(fp)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    order = ["resnet18", "efficientnet_b0", "vit_b_16"]
    for i, arch in enumerate([a for a in order if a in set(ma.model)]):
        s = ma[ma.model == arch]
        ax.scatter([i] * len(s), s.patient_auroc, color=BLUE, zorder=3,
                   label="individual seeds" if i == 0 else None)
        ax.plot([i - 0.18, i + 0.18], [s.patient_auroc.mean()] * 2, color=RED, lw=2,
                label="seed mean" if i == 0 else None)
    ax.axhline(0.7792, color=GRAY, ls="--", lw=0.9, label="locked ResNet18 (0.779, ref)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["ResNet18\n(3 seeds)", "EfficientNet-B0\n(3 seeds)", "ViT-B/16\n(1 seed)"], fontsize=8)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Patient AUROC")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Fold-paired architectures on the frozen master map\n(identical folds, protocol; seed spread ≈ architecture differences)")
    save(fig, "R2_F1_architectures_master.png")

# R2-F2: POD7 risk-set primary estimand
fp = os.path.join(OUT, "task2v2_riskset_primary.csv")
if os.path.exists(fp):
    rs = pd.read_csv(fp)
    prim = rs[rs.aggregation.str.contains("mean")]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(prim))
    ax.errorbar(x, prim.auroc, yerr=[prim.auroc - prim.lo, prim.hi - prim.auroc],
                fmt="o", capsize=4, color=BLUE)
    for i, r in enumerate(prim.itertuples()):
        ax.text(i, r.hi + 0.02, f"{r.auroc:.3f}\n({r.n_events} events)", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"POD{int(r.landmark)}" for r in prim.itertuples()])
    ax.axhline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.set_ylim(0.1, 1.05); ax.set_ylabel("Patient AUROC (95% CI)")
    ax.set_title("Risk-set analyses: predict subsequent SSI among not-yet-diagnosed\n(mean of photos through landmark; POD7 = primary)")
    save(fig, "R2_F2_pod7_riskset.png")

# R2-F3: decision curves
fp = os.path.join(OUT, "task4b_decision_curves.csv")
if os.path.exists(fp):
    dc = pd.read_csv(fp)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for col, c in [("clinical_intraop", GREEN), ("image", BLUE), ("combined", ORANGE), ("treat_all", GRAY)]:
        ax.plot(dc.threshold, dc[col], color=c, label=col.replace("_", " "),
                ls="--" if col == "treat_all" else "-")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.legend(fontsize=7); ax.set_ylim(-0.02, 0.11)
    ax.set_title("Decision curves (cross-fold Platt-calibrated, 210 pts / 20 events)")
    save(fig, "R2_F3_decision_curves.png")

# R2-F4: masked retrains on master map
fp = os.path.join(MA, "task8_masked_retrain_performance.csv")
if os.path.exists(fp):
    t8 = pd.read_csv(fp)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(t8))
    ax.bar(x, t8.patient_auroc, color=[BLUE, GREEN, RED], width=0.55)
    ax.errorbar(x, t8.patient_auroc, yerr=[t8.patient_auroc - t8.pat_lo, t8.pat_hi - t8.patient_auroc],
                fmt="none", ecolor="k", capsize=4)
    for i, r in t8.iterrows():
        ax.text(i, r.pat_hi + 0.02, f"{r.patient_auroc:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(t8.condition.str.replace("_", " "), fontsize=8)
    ax.set_ylim(0.3, 1.0); ax.axhline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.set_ylabel("Patient AUROC")
    ax.set_title(f"Masked retrains, frozen master folds\n(annotated subset {int(t8.n_img.iloc[0])} img / "
                 f"{int(t8.n_pts.iloc[0])} pts / {int(t8.n_pos_pts.iloc[0])} SSI+; RU-A1308 unmaskable)")
    save(fig, "R2_F4_masked_master.png")
print("R2_FIGS_DONE", flush=True)
