#!/usr/bin/env python
"""Task 10 v2 (PI round-2, point 4): ALL rater analyses regenerated from ONE
adjudicated patient-level outcome = the locked roster label (chart-review adjudicated;
RU-A1042 positive). The REDCap export's stale is_ssi column is not used anywhere.
Adds the paired cohort flow (rated -> label-known -> CNN-matched) with positive image
AND patient counts at each stage. Kappa gates are label-free and re-asserted."""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lib import dedup, safe_auroc, safe_auprc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "paper1")
RNG = np.random.default_rng(20260807)

raters, imgs = [], []
for line in open(os.path.join(HERE, "rater_data", "compact_ratings.txt")):
    line = line.rstrip("\n")
    if line.startswith("RATER "):
        p = line.split()
        raters.append(dict(code=p[1], group=p[-3], completed=int(p[-2]), n=int(p[-1])))
    elif line.strip():
        m = re.match(r"^(\d{4})_D(\d+) ([01]) ([01.?]+)$", line)
        imgs.append(dict(pid=f"RU-A{m.group(1)}", pod=int(m.group(2)), votes=m.group(4)))
R = pd.DataFrame(raters); I = pd.DataFrame(imgs)
comp_idx = [i for i, r in R.iterrows() if r.completed == 1]
V = np.array([[int(v[i]) for i in comp_idx] for v in I.votes])
assert len(I) == 468 and V.shape == (468, 15)

# label-free kappa gates (unchanged from v1)
n = 15; n_inf = V.sum(1)
Pi = (n_inf * (n_inf - 1) + (n - n_inf) * (n - n_inf - 1)) / (n * (n - 1))
p_inf = V.sum() / V.size
Pe = p_inf ** 2 + (1 - p_inf) ** 2
kappa = (Pi.mean() - Pe) / (1 - Pe)
assert abs(kappa - 0.105) < 0.0015

# ---- ONE adjudicated outcome: locked roster labels across all arms ----
lab = {}
for arm in ("ssi_smartphone", "ssi_both", "ssi_smartphone_expanded"):
    b = pd.read_csv(os.path.join(HERE, "base", f"{arm}.csv"))
    for pid, y in b.groupby("pid").y_true.max().items():
        lab.setdefault(pid, int(y))
I["y_adj"] = I.pid.map(lab)
sp = dedup(pd.read_csv(os.path.join(HERE, "base", "ssi_smartphone.csv")))
cnn = sp.groupby(["pid", "pod"]).proba.mean().rename("cnn_score").reset_index()
I = I.merge(cnn, on=["pid", "pod"], how="left")
I["consensus_frac"] = V.sum(1) / n

known = I.y_adj.notna()
matched = known & I.cnn_score.notna()
flow = pd.DataFrame([
    dict(stage="rated images", n_img=468, n_pos_img=int(I.y_adj.fillna(0).sum()),
         n_pts=I.pid.nunique(), n_pos_pts=I[I.y_adj == 1].pid.nunique(),
         note="positives counted where outcome known"),
    dict(stage="outcome known (locked roster, all arms)", n_img=int(known.sum()),
         n_pos_img=int(I[known].y_adj.sum()), n_pts=I[known].pid.nunique(),
         n_pos_pts=I[known & (I.y_adj == 1)].pid.nunique(),
         note=f"{int((~known).sum())} images from {I[~known].pid.nunique()} pids not in any locked arm -> excluded from label metrics"),
    dict(stage="matched to locked smartphone (pid,POD)", n_img=int(matched.sum()),
         n_pos_img=int(I[matched].y_adj.sum()), n_pts=I[matched].pid.nunique(),
         n_pos_pts=I[matched & (I.y_adj == 1)].pid.nunique(),
         note="paired CNN-vs-consensus cohort; POSITIVE PATIENTS ARE FEW - see caveat"),
])
flow.to_csv(os.path.join(OUT, "task10v2_cohort_flow.csv"), index=False)
print(flow.to_string(index=False), flush=True)

def clus_boot(y, s, p, stat=safe_auroc, nb=2000):
    df = pd.DataFrame({"p": p}); groups = df.groupby("p").indices
    ps = list(groups); obs = stat(y, s); v = []
    for _ in range(nb):
        pick = RNG.choice(len(ps), len(ps), True)
        idx = np.concatenate([groups[ps[i]] for i in pick])
        a = stat(y[idx], s[idx])
        if not np.isnan(a): v.append(a)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4)

def paired_clus(y, a, b, p, nb=2000):
    df = pd.DataFrame({"p": p}); groups = df.groupby("p").indices
    ps = list(groups); obs = safe_auroc(y, a) - safe_auroc(y, b); v = []
    for _ in range(nb):
        pick = RNG.choice(len(ps), len(ps), True)
        idx = np.concatenate([groups[ps[i]] for i in pick])
        da, db = safe_auroc(y[idx], a[idx]), safe_auroc(y[idx], b[idx])
        if not (np.isnan(da) or np.isnan(db)): v.append(da - db)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p2 = 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))
    return round(float(obs), 4), round(float(lo), 4), round(float(hi), 4), round(float(min(p2, 1)), 4)

# ---- per-rater sens/spec on ADJUDICATED labels (label-known images) ----
K = I[known].reset_index(drop=True)
VK = V[known.values]
yk = K.y_adj.astype(int).values
rr = []
for k, ri in enumerate(comp_idx):
    votes = VK[:, k]
    rr.append(dict(rater=R.loc[ri, "code"], group=R.loc[ri, "group"],
                   sens=round(float(votes[yk == 1].mean()), 4),
                   spec=round(float(1 - votes[yk == 0].mean()), 4),
                   n_img=len(yk), n_pos_img=int(yk.sum())))
RR = pd.DataFrame(rr)
RR.to_csv(os.path.join(OUT, "task10v2_per_rater_performance.csv"), index=False)
for g, s in RR.groupby("group"):
    print(f"{g}: sens {s.sens.mean():.3f} spec {s.spec.mean():.3f}", flush=True)

# ---- consensus vs CNN on the matched cohort ----
M = I[matched].reset_index(drop=True)
ym = M.y_adj.astype(int).values; pm = M.pid.values
ca, cl, ch = clus_boot(ym, M.consensus_frac.values, pm)
na, nl, nh = clus_boot(ym, M.cnn_score.values, pm)
do, dl, dh, dp = paired_clus(ym, M.cnn_score.values, M.consensus_frac.values, pm)
n_pos_pts = M[M.y_adj == 1].pid.nunique()
cmp_df = pd.DataFrame([
    dict(model="clinician_consensus", auroc=ca, lo=cl, hi=ch,
         auprc=round(float(safe_auprc(ym, M.consensus_frac.values)), 4)),
    dict(model="CNN_locked", auroc=na, lo=nl, hi=nh,
         auprc=round(float(safe_auprc(ym, M.cnn_score.values)), 4)),
    dict(model="paired_delta_CNN_minus_consensus", auroc=do, lo=dl, hi=dh, auprc=np.nan),
])
cmp_df["n_img"] = len(M); cmp_df["n_pos_img"] = int(ym.sum()); cmp_df["n_pos_PATIENTS"] = n_pos_pts
cmp_df["note"] = ["", "", f"p={dp}; SUPPLEMENT-GRADE: only {n_pos_pts} SSI+ patients in the paired cohort"]
cmp_df.to_csv(os.path.join(OUT, "task10v2_consensus_vs_cnn.csv"), index=False)
print(cmp_df[["model", "auroc", "lo", "hi", "note"]].to_string(index=False), flush=True)

srows = []
for g in ("Surgeon", "Non-surgeon"):
    cols = [k for k, ri in enumerate(comp_idx) if R.loc[ri, "group"] == g]
    frac = V[:, cols].mean(1)[matched.values]
    a, lo, hi = clus_boot(ym, frac, pm)
    srows.append(dict(group=g, n_raters=len(cols), consensus_auroc=a, lo=lo, hi=hi))
pd.DataFrame(srows).to_csv(os.path.join(OUT, "task10v2_surgeon_vs_nonsurgeon.csv"), index=False)
I.drop(columns=["votes"]).to_csv(os.path.join(OUT, "task10v2_image_level.csv"), index=False)
print("TASK10V2_DONE", flush=True)
