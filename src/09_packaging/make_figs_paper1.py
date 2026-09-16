#!/usr/bin/env python
"""Paper-1 figures. Each block guards on its source CSV existing, so the script
can run mid-pipeline and again at the end."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1"); FIG = os.path.join(OUT, "figures")
FT = os.path.join(HERE, "out", "finetune")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREEN, RED, GRAY = "#2C6FBB", "#E8873A", "#3B8E5A", "#C44536", "#7A7A7A"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    print("wrote", name, flush=True)

def ex(f): return os.path.exists(os.path.join(OUT, f))

# F1: estimand comparison (locked vs detect-before-diagnosis)
if ex("task2_estimand_primary.csv") and ex("task2_landmark_single.csv"):
    e = pd.read_csv(os.path.join(OUT, "task2_estimand_primary.csv"))
    lm = pd.read_csv(os.path.join(OUT, "task2_landmark_single.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    lab = ["Locked estimand\n(all images incl. post-dx)", "Detect-before-diagnosis\n(pre-dx images only)"]
    vals = e.patient_auroc.values[:2]
    los = [np.nan, e.lo.values[1]]; his = [np.nan, e.hi.values[1]]
    ax[0].bar(lab, vals, color=[GRAY, BLUE], width=0.55)
    ax[0].errorbar([1], [vals[1]], yerr=[[vals[1] - los[1]], [his[1] - vals[1]]], fmt="none", ecolor="k", capsize=4)
    for i, v in enumerate(vals): ax[0].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("Patient AUROC")
    ax[0].set_title(f"Leak-free estimand (paired Δ {e.patient_auroc.values[2]:+.4f}, NS)")
    for lblname, c in [("eventual_SSI", BLUE), ("not_yet_diagnosed_at_t", ORANGE)]:
        s = lm[lm.label == lblname]
        ax[1].errorbar(s.landmark, s.patient_auroc, yerr=[s.patient_auroc - s.lo, s.hi - s.patient_auroc],
                       marker="o", capsize=3, color=c,
                       label=lblname.replace("_", " ") + " (n_pos " + "/".join(map(str, s.n_pos)) + ")")
    ax[1].axhline(0.5, color=GRAY, ls=":", lw=0.8)
    ax[1].set_xlabel("Landmark day (POD)"); ax[1].set_ylabel("Patient AUROC")
    ax[1].set_ylim(0.4, 1.0); ax[1].legend(fontsize=7); ax[1].set_title("Landmark analysis")
    save(fig, "P1_F1_estimand_landmark.png")

# F2: probe ladder
if ex("task6_probe_performance.csv") and ex("task6_cnn_vs_probe_paired.csv"):
    pr = pd.read_csv(os.path.join(OUT, "task6_probe_performance.csv"))
    dl = pd.read_csv(os.path.join(OUT, "task6_cnn_vs_probe_paired.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(pr.probe) + ["CNN (locked)"]
    pats = list(pr.patient_auroc) + [0.7792]
    cols = [ORANGE] * len(pr) + [BLUE]
    ypos = np.arange(len(names))
    ax.barh(ypos, pats, color=cols, height=0.6)
    for i, v in enumerate(pats): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    for i, (_, r) in enumerate(dl.iterrows()):
        ax.text(0.315, i, f"CNN Δ {r.patient_delta:+.3f}  p={r.patient_p}", va="center", fontsize=7, color=GRAY)
    ax.set_yticks(ypos); ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    ax.axvline(0.5, color=GRAY, ls=":", lw=0.8); ax.set_xlim(0.3, 0.9)
    ax.set_xlabel("Patient AUROC")
    ax.set_title("Confound probes (past-only features) vs CNN — patient level")
    save(fig, "P1_F2_probe_ladder.png")

# F3: calibration + operating points
if ex("task7_calibration_metrics.csv") and ex("task7_operating_points.csv"):
    cm = pd.read_csv(os.path.join(OUT, "task7_calibration_metrics.csv"))
    op = pd.read_csv(os.path.join(OUT, "task7_operating_points.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(cm)); w = 0.35
    ax[0].bar(x - w / 2, cm.cal_slope, w, label="calibration slope (1=ideal)", color=BLUE)
    ax[0].bar(x + w / 2, cm.brier * 10, w, label="Brier ×10", color=ORANGE)
    ax[0].axhline(1.0, color=GRAY, ls=":", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(cm.scores); ax[0].legend(fontsize=7)
    ax[0].set_title("Cross-fold recalibration (patient level)")
    if "score_scale" in op.columns:
        op = op[op.score_scale.str.contains("PRIMARY")]
    ax[1].errorbar(op.target_spec, op.sensitivity, yerr=[op.sensitivity - op.sens_lo, op.sens_hi - op.sensitivity],
                   marker="s", capsize=4, color=BLUE)
    for _, r in op.iterrows():
        ax[1].text(r.target_spec, r.sensitivity + 0.06, f"sens {r.sensitivity:.2f}\nPPV {r.ppv:.2f}", ha="center", fontsize=7)
    ax[1].set_xlabel("Target specificity"); ax[1].set_ylabel("Sensitivity (95% CI)")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("Operating points (raw locked score, 20 events)")
    save(fig, "P1_F3_calibration_ops.png")

# F4: clinical vs image vs combined (full cohort, ascertainment-clean)
if ex("task4_clinical_image_combined.csv"):
    t4 = pd.read_csv(os.path.join(OUT, "task4_clinical_image_combined.csv"))
    keep = t4[t4.model.str.contains("PRIMARY|image_only|combined")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(keep))
    ax.bar(x, keep.patient_auroc, color=[GREEN, BLUE, ORANGE][:len(keep)], width=0.55)
    ax.errorbar(x, keep.patient_auroc, yerr=[keep.patient_auroc - keep.lo, keep.hi - keep.patient_auroc],
                fmt="none", ecolor="k", capsize=4)
    for i, r in keep.iterrows():
        ax.text(i, r.hi + 0.02, f"{r.patient_auroc:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["clinical only\n(intraop, ascertainment-clean)", "image only\n(locked CNN)", "combined\n(intraop + image)"], fontsize=8)
    ax.set_ylim(0.4, 1.0); ax.axhline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.set_ylabel("Patient AUROC")
    ax.set_title("Full locked cohort (210 pts, 20 events) — deltas NS;\nchart-coverage ascertainment leak excluded (coverage alone = 0.747)")
    save(fig, "P1_F4_clinical_image_combined.png")

# F5: three-subset table figure
if ex("task1_three_subset_table.csv"):
    t1 = pd.read_csv(os.path.join(OUT, "task1_three_subset_table.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.0))
    y = np.arange(len(t1))[::-1]
    ax.errorbar(t1.patient_auroc, y, xerr=[t1.patient_auroc - t1.lo, t1.hi - t1.patient_auroc],
                fmt="o", capsize=4, color=BLUE)
    for i, (yy, r) in enumerate(zip(y, t1.itertuples())):
        ax.text(0.42, yy + 0.18, f"{r.subset}  (n={r.n_patients}, {r.n_pos} SSI+, prev {r.prevalence:.1%})", fontsize=8)
    ax.axvline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.set_yticks([]); ax.set_xlim(0.4, 1.0); ax.set_xlabel("Patient AUROC (95% CI)")
    ax.set_title("One primary number, cohort definitions explicit")
    save(fig, "P1_F5_three_subsets.png")

# F6: architecture comparison (locked ResNet18 vs end-to-end fine-tunes)
eff_fp = os.path.join(FT, "efficientnet_b0_oof.csv"); vit_fp = os.path.join(FT, "vit_b_16_oof.csv")
if os.path.exists(eff_fp):
    from sklearn.metrics import roc_auc_score
    entries = [("ResNet18\n(locked, trained)", 0.7792, GRAY)]
    for fp, nm, c in [(eff_fp, "EfficientNet-B0\n(end-to-end)", BLUE), (vit_fp, "ViT-B/16\n(end-to-end)", ORANGE)]:
        if not os.path.exists(fp): continue
        f = pd.read_csv(fp)
        pat = f.groupby("pid").agg(y=("y_true", "max"), s=("proba_ft", "mean"))
        entries.append((nm, roc_auc_score(pat.y, pat.s), c))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar([e[0] for e in entries], [e[1] for e in entries], color=[e[2] for e in entries], width=0.55)
    for i, e in enumerate(entries): ax.text(i, e[1] + 0.01, f"{e[1]:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Patient AUROC")
    ax.set_title("End-to-end fine-tunes land on the same plateau")
    save(fig, "P1_F6_architecture.png")

# F7: learning curve
lc_fp = os.path.join(FT, "learning_curve.csv")
if os.path.exists(lc_fp):
    lc = pd.read_csv(lc_fp)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(lc.frac * 100, lc.patient_auroc, marker="o", color=BLUE, label="patient AUROC")
    ax.plot(lc.frac * 100, lc.image_auroc, marker="s", color=ORANGE, label="image AUROC")
    ax.set_xlabel("% of training patients"); ax.set_ylabel("AUROC"); ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0); ax.axhline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.set_title("EfficientNet-B0 learning curve (5-fold OOF)")
    save(fig, "P1_F7_learning_curve.png")

# F8: masked retrains
if ex("task8_masked_retrain_performance.csv"):
    t8 = pd.read_csv(os.path.join(OUT, "task8_masked_retrain_performance.csv"))
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
    ax.set_title(f"Retrained masked models ({int(t8.n_pos_pts.iloc[0])} SSI+ pts, all folds)")
    save(fig, "P1_F8_masked_retrain.png")
# F9: clinicians vs CNN on the same images
if ex("task10_consensus_vs_cnn.csv") and ex("task10_per_rater_performance.csv") and ex("task10_image_level.csv"):
    from sklearn.metrics import roc_curve
    t10 = pd.read_csv(os.path.join(OUT, "task10_consensus_vs_cnn.csv"))
    rr = pd.read_csv(os.path.join(OUT, "task10_image_level.csv"))
    pr = pd.read_csv(os.path.join(OUT, "task10_per_rater_performance.csv"))
    m = rr[rr.cnn_score.notna()]
    yv = m.y_cnn.astype(int).values
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for col, nm, c in [("cnn_score", f"CNN ({t10.iloc[1].auroc:.3f})", BLUE),
                       ("consensus_frac", f"15-clinician consensus ({t10.iloc[0].auroc:.3f})", ORANGE)]:
        fpr, tpr, _ = roc_curve(yv, m[col].values)
        ax.plot(fpr, tpr, color=c, label=nm)
    ax.scatter(1 - pr.spec, pr.sens, marker="x", color=GRAY, s=28, label="individual raters (n=15)")
    ax.plot([0, 1], [0, 1], ls=":", color=GRAY, lw=0.8)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Same {len(m)} photographs: CNN vs clinicians\n"
                 f"Δ{t10.iloc[2].auroc:+.3f} (patient-clustered p=0.044); Fleiss κ=0.105")
    save(fig, "P1_F9_clinicians_vs_cnn.png")
print("FIGS_DONE", flush=True)
