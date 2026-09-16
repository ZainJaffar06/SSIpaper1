#!/usr/bin/env python
"""Task 10: per-rating human-vs-CNN analysis from the recovered REDCap export
('Clean Real Study Rows' sheet, extracted via authenticated SharePoint session).

Data: rater_data/compact_ratings.txt — 18 RATER lines (code, profession, group,
completed flag, n_ratings) then 468 image lines: "<pid4>_D<pod> <is_ssi> <votes18>"
where votes18 is one char per rater in the SAME alphabetical rater order as the
RATER lines ('1' infected, '0' not_infected, '.' not rated).

GATES (must reproduce the workbook's own Summary sheet, else abort):
  total real rows 7062; 468 images; 18 raters; 15 completed;
  Fleiss kappa (completed) 0.105; Pbar 0.687; Pe 0.650; mean image agreement 0.798.

Analyses: kappa reproduction; per-rater sens/spec (patient-level SSI label);
consensus vote-fraction ROC vs locked CNN on the same images (patient-clustered
paired bootstrap); surgeon vs non-surgeon; per-image difficulty."""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lib import dedup, safe_auroc, safe_auprc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1"); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260807)

raters, imgs = [], []
for line in open(os.path.join(HERE, "rater_data", "compact_ratings.txt")):
    line = line.rstrip("\n")
    if line.startswith("RATER "):
        parts = line.split()
        raters.append(dict(code=parts[1], group=parts[-3], completed=int(parts[-2]),
                           n=int(parts[-1]), profession=" ".join(parts[2:-3])))
    elif line.strip():
        m = re.match(r"^(\d{4})_D(\d+) ([01]) ([01.?]+)$", line)
        assert m, f"bad line: {line}"
        imgs.append(dict(pid=f"RU-A{m.group(1)}", pod=int(m.group(2)),
                         is_ssi=int(m.group(3)), votes=m.group(4)))
R = pd.DataFrame(raters); I = pd.DataFrame(imgs)
assert len(R) == 18 and len(I) == 468, (len(R), len(I))
assert all(len(v) == 18 for v in I.votes)
comp_idx = [i for i, r in R.iterrows() if r.completed == 1]
assert len(comp_idx) == 15
total_rows = int(sum((np.array(list(v)) != '.').sum() for v in I.votes))
print(f"rows={total_rows} (want 7062)  images={len(I)}  raters={len(R)}  completed={len(comp_idx)}")
assert total_rows == 7062

# votes matrix for completed raters: (468, 15) of 0/1
V = np.array([[int(v[i]) for i in comp_idx] for v in I.votes])
n = V.shape[1]
n_inf = V.sum(1); n_not = n - n_inf
Pi = (n_inf * (n_inf - 1) + n_not * (n_not - 1)) / (n * (n - 1))
Pbar = Pi.mean()
p_inf = V.sum() / V.size
Pe = p_inf ** 2 + (1 - p_inf) ** 2
kappa = (Pbar - Pe) / (1 - Pe)
agree = np.maximum(n_inf, n_not) / n
print(f"GATES: kappa={kappa:.3f} (0.105)  Pbar={Pbar:.3f} (0.687)  Pe={Pe:.3f} (0.650)  mean_agree={agree.mean():.3f} (0.798)")
assert abs(kappa - 0.105) < 0.0015 and abs(Pbar - 0.687) < 0.0015 and abs(Pe - 0.650) < 0.0015
assert abs(agree.mean() - 0.798) < 0.0015
pd.DataFrame([dict(n_images=468, n_raters=15, fleiss_kappa=round(kappa, 4), Pbar=round(Pbar, 4),
                   Pe=round(Pe, 4), mean_image_agreement=round(float(agree.mean()), 4),
                   infected_vote_fraction=round(float(p_inf), 4))]).to_csv(
    os.path.join(OUT, "task10_kappa_reproduction.csv"), index=False)

# ---- link to locked CNN scores by (pid, pod) ----
d = dedup(pd.read_csv(os.path.join(HERE, "base", "ssi_smartphone.csv")))
cnn = d.groupby(["pid", "pod"]).agg(cnn_score=("proba", "mean"), y_cnn=("y_true", "max")).reset_index()
I = I.merge(cnn, on=["pid", "pod"], how="left")
matched = I.cnn_score.notna()
lab_mismatch = I[matched & (I.is_ssi != I.y_cnn)]
print(f"CNN match: {int(matched.sum())}/468 images; label mismatches: {len(lab_mismatch)}")
I["consensus_frac"] = V.sum(1) / n

M = I[matched].reset_index(drop=True)
# truth for the CNN comparison = LOCKED roster label (chart-review adjudicated).
# 3 rated images of RU-A1042 carry a stale is_ssi=0 in the REDCap export; roster says positive.
y = M.y_cnn.astype(int).values; pids = M.pid.values

def clus_boot(y_, s_, p_, nb=2000):
    df = pd.DataFrame({"p": p_}); groups = df.groupby("p").indices
    ps = list(groups); obs = safe_auroc(y_, s_); v = []
    for _ in range(nb):
        pick = RNG.choice(len(ps), len(ps), True)
        idx = np.concatenate([groups[ps[i]] for i in pick])
        a = safe_auroc(y_[idx], s_[idx])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

def paired_clus(y_, a_, b_, p_, nb=2000):
    df = pd.DataFrame({"p": p_}); groups = df.groupby("p").indices
    ps = list(groups); obs = safe_auroc(y_, a_) - safe_auroc(y_, b_); v = []
    for _ in range(nb):
        pick = RNG.choice(len(ps), len(ps), True)
        idx = np.concatenate([groups[ps[i]] for i in pick])
        da, db = safe_auroc(y_[idx], a_[idx]), safe_auroc(y_[idx], b_[idx])
        if not (np.isnan(da) or np.isnan(db)): v.append(da - db)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1)), 4)

ca, cl, ch = clus_boot(y, M.consensus_frac.values, pids)
na, nl, nh = clus_boot(y, M.cnn_score.values, pids)
do, dl, dh, dp = paired_clus(y, M.cnn_score.values, M.consensus_frac.values, pids)
cmp_rows = [dict(model="clinician_consensus (frac of 15 voting infected)", n_img=len(M),
                 n_pos_img=int(y.sum()), auroc=ca, lo=cl, hi=ch,
                 auprc=round(float(safe_auprc(y, M.consensus_frac.values)), 4)),
            dict(model="CNN (locked OOF, mean per pid-day)", n_img=len(M),
                 n_pos_img=int(y.sum()), auroc=na, lo=nl, hi=nh,
                 auprc=round(float(safe_auprc(y, M.cnn_score.values)), 4)),
            dict(model="paired delta CNN minus consensus (clustered)", n_img=len(M),
                 n_pos_img=int(y.sum()), auroc=do, lo=dl, hi=dh, auprc=np.nan)]
pd.DataFrame(cmp_rows).assign(note=["", "", f"p={dp}"]).to_csv(
    os.path.join(OUT, "task10_consensus_vs_cnn.csv"), index=False)
print("consensus AUROC", ca, "CNN", na, "delta", do, "p", dp)

# ---- per-rater sens/spec (all 468, patient-level label) ----
yy = I.is_ssi.values
rrows = []
for k, ri in enumerate(comp_idx):
    votes = V[:, k]
    sens = float(votes[yy == 1].mean()); spec = float(1 - votes[yy == 0].mean())
    rrows.append(dict(rater=R.loc[ri, "code"], group=R.loc[ri, "group"],
                      profession=R.loc[ri, "profession"],
                      sens=round(sens, 4), spec=round(spec, 4),
                      flag_rate=round(float(votes.mean()), 4),
                      youden=round(sens + spec - 1, 4)))
RR = pd.DataFrame(rrows)
RR.to_csv(os.path.join(OUT, "task10_per_rater_performance.csv"), index=False)
for g, sub in RR.groupby("group"):
    print(f"{g}: n={len(sub)} sens {sub.sens.mean():.3f} spec {sub.spec.mean():.3f} youden {sub.youden.mean():.3f}")

# surgeon vs non-surgeon consensus
srows = []
for g in ("Surgeon", "Non-surgeon"):
    cols = [k for k, ri in enumerate(comp_idx) if R.loc[ri, "group"] == g]
    frac = V[:, cols].mean(1)
    fm = frac[matched.values]
    a, lo, hi = clus_boot(y, fm, pids)
    srows.append(dict(group=g, n_raters=len(cols), consensus_auroc=a, lo=lo, hi=hi))
pd.DataFrame(srows).to_csv(os.path.join(OUT, "task10_surgeon_vs_nonsurgeon.csv"), index=False)
print(pd.DataFrame(srows).to_string(index=False))
I.drop(columns=["votes"]).to_csv(os.path.join(OUT, "task10_image_level.csv"), index=False)
print("TASK10_DONE")
