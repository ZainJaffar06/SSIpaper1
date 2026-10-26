import sys, os,re; sys.path.insert(0,'.'); sys.path.insert(0,'/Users/zainjaffar/ssi_work/paper1_2026-08-15')
import pandas as pd, numpy as np
from dxml import Doc
D='/Users/zainjaffar/ssi_work/paper1_2026-08-15/out/postop_only/derived/'
t=Doc(os.environ['OUTDOC']).build_index()
P=[];F=[]
def ck(lab,s): (P if s in t else F).append(f'{lab}: {s!r}')
pl=pd.read_csv(D+'primary_and_landmarks.csv'); p7=pl.iloc[0]
ck('primary n/events','132 patients with 8 subsequent infections'); ck('primary AUROC',f'{p7.auroc:.3f} (95% confidence interval [CI], {p7.lo:.3f}–{p7.hi:.3f})')
ck('primary interval includes 0.5','the AUROC interval includes 0.5')
ck('cohort frame','1,341 postoperative photographs from 171 patients'); ck('folds','31, 34, 36, 38, and 32 patients, with 3, 0, 4, 2, and 4 SSI events')
a=pd.read_csv(D+'architectures.csv').set_index('architecture')
for k,l in [('resnet18','ResNet18'),('efficientnet_b0','EfficientNet-B0'),('vit_b_16','ViT-B/16')]: ck(f'arch {l}',f'{a.loc[k,"mean"]:.3f}±{a.loc[k,"sd"]:.3f} for {l}' if l!='ViT-B/16' else f'{a.loc[k,"mean"]:.3f}±{a.loc[k,"sd"]:.3f} for ViT-B/16')
ma=pd.read_csv(D+'masking_absolute.csv').set_index('condition'); mc=pd.read_csv(D+'masking_contrasts.csv').set_index('contrast')
ck('masking frame','1,296-image, 168-patient mask-eligible cohort with 12 events')
for c in ma.index: ck(f'mask {c[:20]}',f'{ma.loc[c,"seed_mean"]:.3f}')
sw=mc.loc['shifted - true wound']; ck('headline contrast',f'{sw.d:+.3f} (95% CI, {sw.lo:+.3f} to {sw.hi:+.3f}; P={sw.p:.2f}')
ck('no contrast significant','No contrast in the family reached significance')
pm=dict(zip(*pd.read_csv(D+'permutation_summary.csv').values.T))
ck('perm pct',f'{pm["true_wound_percentile"]:.0f}th percentile'); ck('perm P',f'one-sided P={pm["one_sided_p_wound_better"]:.2f}'); ck('perm n',f'{int(pm["n_ge_true_wound"])} of 300')
ck('perm null',f'{pm["null_mean"]:.3f} (SD {pm["null_sd"]:.3f}'); ck('shift rescored',f'{pm["locked_shift_auroc_rescored"]:.3f} on this estimator, at the {pm["locked_shift_percentile"]:.0f}th percentile')
ck('infeasible 1','One of the 1,296 images admits no such position')
cm=pd.read_csv(D+'clinical_models.csv').set_index('model'); cc=pd.read_csv(D+'clinical_contrasts.csv').set_index('contrast')
ck('clinical 8var',f'{cm.loc["clinical 8-variable","auroc"]:.3f} (95% CI'); ck('image full',f'{cm.loc["image only","auroc"]:.3f} ('); ck('stacked',f'{cm.loc["stacked, 8-variable + image","auroc"]:.3f} (')
x8=cc.loc['image adds to clinical 8-variable']; ck('increment 8var',f'{x8.d_auroc:+.3f} (95% CI, {x8.lo:+.3f} to {x8.hi:+.3f}; P={x8.p:.3f})')
xv=cc.loc['image adds to clinical v2 5-variable']; ck('increment v2',f'{xv.d_auroc:+.3f} ({xv.lo:+.3f} to {xv.hi:+.3f}; P={xv.p:.2f})')
c7=pd.read_csv(D+'clinical_pod7.csv').set_index('model'); ck('POD7 clinical prose',f'{c7.loc["clinical","auroc"]:.3f}, {c7.loc["image","auroc"]:.3f}, and {c7.loc["stacked","auroc"]:.3f}')
rs=pd.read_csv(D+'region_strata.csv').set_index('stratum'); ck('region indicator',f'{rs.loc["BENCHMARK region indicator alone","auroc"]:.3f} ({rs.loc["BENCHMARK region indicator alone","lo"]:.3f}')
ck('abdomen',f'{rs.loc["Abdomen","auroc"]:.3f} (95% CI, {rs.loc["Abdomen","lo"]:.3f}')
ca=pd.read_csv(D+'calibration.csv').set_index('metric').value; ck('brier',f'Brier score of {ca["brier"]:.3f}'); ck('slope',f'calibration slope of {ca["calibration_slope"]:.2f}')
ck('S13 cited','(Supplementary Table S13)'); ck('S13 table','Supplementary Table S13 | Image quality-flag codebook')
ck('intraop excluded','all 598 of them (280 obtained before closure and 318 after) were excluded'); ck('41 retained','Forty-one photographs taken on the day of surgery')
ck('comment 10','its direction is not known'); ck('gate claim fixed','share no file identity')
# must be GONE
for lab,s in [('old primary 0.754','AUROC of 0.754'),('old CI','0.632–0.861'),('old masking','1,886-image'),('old +0.138 significant','+0.138 that survives'),
              ('old 20th pct','20th percentile'),('old 240/300','240 of 300'),('old 52.1%','passed the gate'),('intraop 0.706','0.706 (0.594'),('old fold 7,2,4,3,4','7, 2, 4, 3, and 4'),
              ('serial','serial analyses'),('old region 0.793','0.793 (0.680'),('old brier','Brier score of 0.070'),('study codes','RU-A1')]:
    (F if s in t else P).append(f'GONE {lab}')
print(f'PASS {len(P)} / FAIL {len(F)}')
for f in F: print('  **FAIL**',f)
em=t.count('—'); print(f'em-dashes remaining: {em}')
print(f'words: {len(t.split())}')
