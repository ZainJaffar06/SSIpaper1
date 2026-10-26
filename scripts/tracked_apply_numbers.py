# -*- coding: utf-8 -*-
import sys, os, re, os, shutil
sys.path.insert(0,'.'); sys.path.insert(0,'/Users/zainjaffar/ssi_work/paper1_2026-08-15')
import pandas as pd, numpy as np
from dxml_tracked import TrackedDoc as Doc
from lib import safe_auroc, safe_auprc, cluster_boot_ci
D='/Users/zainjaffar/ssi_work/paper1_2026-08-15/out/postop_only/derived/'
SRC=os.environ['NEWDOC']
OUT=os.environ['OUTDOC']
shutil.copy(SRC,OUT)
doc=Doc(OUT); LOG=[]
def R(pat,new,lab,count=1):
    n=doc.replace(pat,new,lab,count=count)
    LOG.append(('OK  ' if n else 'MISS')+f' [{n}] {lab}')

# ---------- load results ----------
pl=pd.read_csv(D+'primary_and_landmarks.csv'); p7,lt,cl,p14,p30=[pl.iloc[i] for i in range(5)]
S=pd.read_csv(D+'pod7_patient_scores.csv')
_,aplo,aphi=cluster_boot_ci(S.pid.values,S.ssi_final.values,S.score.values,safe_auprc,20000,913)
prev=p7.n_events/p7.n_pts
a=pd.read_csv(D+'architectures.csv').set_index('architecture')
ma=pd.read_csv(D+'masking_absolute.csv').set_index('condition')
mc=pd.read_csv(D+'masking_contrasts.csv').set_index('contrast')
pm=dict(zip(*pd.read_csv(D+'permutation_summary.csv').values.T))
cm=pd.read_csv(D+'clinical_models.csv').set_index('model')
cc=pd.read_csv(D+'clinical_contrasts.csv').set_index('contrast')
ps=pd.read_csv(D+'clinical_partition_stability.csv')
rs=pd.read_csv(D+'region_strata.csv').set_index('stratum')
ca=pd.read_csv(D+'calibration.csv').set_index('metric').value
op=pd.read_csv(D+'operating_points.csv').set_index('target_spec')
inc=pd.read_csv(D+'incisional_sensitivity.csv')
f3=lambda x:f'{x:.3f}'; s3=lambda x:f'{x:+.3f}'

# ---------- PRIMARY ----------
R(r'In the POD 7 risk set of 192 patients with 14 subsequent infections',
  f'In the POD 7 risk set of {p7.n_pts} patients with {p7.n_events} subsequent infections','primary n/events')
R(r'yielded an AUROC of 0\.754 \(95% confidence interval \[CI\], 0\.632–0\.861\) and an AUPRC of 0\.162 \(95% CI, 0\.087–0\.318\) at 7\.3% prevalence, against a no-information AUPRC of 0\.073, so the lower confidence bound corresponds to 1\.2 times chance',
  f'yielded an AUROC of {f3(p7.auroc)} (95% confidence interval [CI], {f3(p7.lo)}–{f3(p7.hi)}) and an AUPRC of {f3(p7.auprc)} (95% CI, {f3(aplo)}–{f3(aphi)}) at {100*prev:.1f}% prevalence, against a no-information AUPRC of {f3(prev)}; the AUROC interval includes 0.5','primary AUROC/AUPRC sentence')
R(r'The latest photograph through POD 7 yielded an AUROC of 0\.737 and an AUPRC of 0\.150; the photograph closest to POD 7 yielded 0\.748 and 0\.153',
  f'The latest photograph through POD 7 yielded an AUROC of {f3(lt.auroc)} ({f3(lt.lo)}–{f3(lt.hi)}) and an AUPRC of {f3(lt.auprc)}; the photograph closest to POD 7 yielded {f3(cl.auroc)} ({f3(cl.lo)}–{f3(cl.hi)}) and {f3(cl.auprc)}. Both sensitivities exceeded the prespecified mean, which is retained as the primary because it was prespecified','aggregation sensitivities')
R(r'The patient-level AUROC was 0\.754, but the AUPRC was 0\.162 at a prevalence of 7\.3%, and a threshold capturing 13 of 14 infections flagged approximately half of patients',
  f'The patient-level AUROC was {f3(p7.auroc)} with an interval that included 0.5, and the AUPRC was {f3(p7.auprc)} at a prevalence of {100*prev:.1f}%; with eight events, sensitivity could only move in steps of 0.125, and it was 0.50 at 80%, 90% and 95% specificity alike','Discussion opening numbers')

# ---------- COHORT ----------
R(r'The locked cohort included 1,948 quality-gated photographs from 210 patients; 20 patients \(9\.5%\) developed an SSI',
  'The locked cohort included 1,948 quality-gated photographs from 210 patients; after exclusion of 598 intraoperative photographs and 9 without a derivable postoperative day, the analytic frame comprised 1,341 postoperative photographs from 171 patients, of whom 13 (7.6%) developed an SSI','cohort opening')
R(r'The five folds contained 42, 41, 43, 42, and 42 patients, with 7, 2, 4, 3, and 4 SSI events, respectively\.',
  'On the postoperative frame the five locked folds contained 31, 34, 36, 38, and 32 patients, with 3, 0, 4, 2, and 4 SSI events, respectively; fold assignment was retained from the locked cohort rather than re-stratified, so one fold holds no event, and each fold-specific model therefore trained on 9 to 13 of the 13 positive patients.','fold composition')
R(r'Analysis-specific event counts therefore differ: 20 in the full cohort, 17 where diagnosis timing is required, 14 in the POD 7 landmark risk set, and 19 in the mask-eligible frame',
  'Analysis-specific event counts therefore differ: 13 in the postoperative frame, 12 where diagnosis timing is required, 8 in the POD 7 landmark risk set, and 12 in the mask-eligible frame','analysis-specific counts')

# ---------- ARCHITECTURES ----------
r18,eb,vt=a.loc['resnet18'],a.loc['efficientnet_b0'],a.loc['vit_b_16']
R(r'mean patient-level AUROC was 0\.756±0\.038 for ResNet18, 0\.733±0\.049 for EfficientNet-B0, and 0\.684±0\.045 for ViT-B/16',
  f'mean patient-level AUROC was {f3(r18["mean"])}±{f3(r18.sd)} for ResNet18, {f3(eb["mean"])}±{f3(eb.sd)} for EfficientNet-B0, and {f3(vt["mean"])}±{f3(vt.sd)} for ViT-B/16','architecture means')
R(r'observed seed variability was similar in magnitude to observed architecture differences',
  f'observed seed variability within an architecture ({f3(min(r18.hi-r18.lo,eb.hi-eb.lo,vt.hi-vt.lo))} to {f3(max(r18.hi-r18.lo,eb.hi-eb.lo,vt.hi-vt.lo))}) exceeded the spread across architecture means ({f3(a["mean"].max()-a["mean"].min())})','seed vs architecture spread')

# ---------- MASKING ----------
R(r'In the 1,886-image, 206-patient mask-eligible cohort, AUROC was 0\.636 for the full image, 0\.619 for the true wound-only condition, 0\.749 for the true background-only condition, 0\.776 for the circularly shifted wound-size region, and 0\.607 for the complement of the shifted region',
  f'In the 1,296-image, 168-patient mask-eligible cohort with 12 events, the mean within-seed AUROC across three training seeds was {f3(ma.loc["Full image","seed_mean"])} for the full image, {f3(ma.loc["True wound-only","seed_mean"])} for the true wound-only condition, {f3(ma.loc["True background-only","seed_mean"])} for the true background-only condition, {f3(ma.loc["Circularly shifted wound-size region","seed_mean"])} for the circularly shifted wound-size region, and {f3(ma.loc["Complement of shifted region","seed_mean"])} for the complement of the shifted region','masking absolutes')
sw=mc.loc['shifted - true wound']
R(r'Consistent with that direction, the shifted region scored higher than the true wound region in the locked run.*?\(Supplementary Table S9\)\.',
  f'The shifted region did not differ from the true wound region: the mean within-seed difference was {s3(sw.d)} (95% CI, {s3(sw.lo)} to {s3(sw.hi)}; P={sw.p:.2f}; Holm-adjusted P={sw.holm_p:.2f} across the six contrasts in this family). No contrast in the family reached significance (Supplementary Table S9).','headline contrast collapsed')
tw,tb,bf,sc,sb=mc.loc['true wound - full'],mc.loc['true background - true wound'],mc.loc['true background - full'],mc.loc['shifted complement - full'],mc.loc['shifted - true background']
R(r'The remaining contrasts were not significant\. The true wound region did not differ from the full image \(−0\.016; −0\.126 to \+0\.101; P=0\.76\), true background exceeded the true wound region by \+0\.098 \(−0\.063 to \+0\.254; P=0\.22\) and the full image by \+0\.082 \(−0\.041 to \+0\.201; P=0\.18\), and the shifted region approximated true background \(\+0\.040; −0\.119 to \+0\.208; P=0\.63\)',
  f'The true wound region scored below the full image ({s3(tw.d)}; {s3(tw.lo)} to {s3(tw.hi)}; P={tw.p:.2f}), true background exceeded the true wound region by {s3(tb.d)} ({s3(tb.lo)} to {s3(tb.hi)}; P={tb.p:.2f}) and matched the full image ({s3(bf.d)}; {s3(bf.lo)} to {s3(bf.hi)}; P={bf.p:.2f}), and the shifted region did not differ from true background ({s3(sb.d)}; {s3(sb.lo)} to {s3(sb.hi)}; P={sb.p:.2f})','remaining contrasts')

# ---------- PERMUTATION ----------
R(r'Forty-six of the 1,886 images admit no such position at any placement \(median minimum achievable overlap 0\.145\) and retained their minimum-overlap position, so across all draws the overlap was 0\.000 at the median and 0\.008 at the mean, with 97\.6% of image-draws at or below 0\.05 and a maximum of 0\.545',
  f'One of the 1,296 images admits no such position at any placement and retained its minimum-overlap position, so across all draws the overlap was {pm["iou_median"]:.3f} at the median and {pm["iou_mean"]:.3f} at the mean, with {100*pm["iou_frac_le_0.05"]:.1f}% of image-draws at or below 0.05 and a maximum of {pm["iou_max"]:.3f}','permutation IoU diagnostics')
R(r'this estimator is weaker than the fine-tuned model and scores the true wound region at 0\.538',
  f'this estimator is weaker than the fine-tuned model and scores the true wound region at {pm["true_wound_auroc"]:.3f}','permutation true wound')
R(r'The relocation distribution had a mean of 0\.585 \(SD 0\.057; range 0\.397 to 0\.778\), and the true wound region fell at its 20th percentile, with 240 of 300 relocations scoring at least as high \(one-sided P=0\.80',
  f'The relocation distribution had a mean of {pm["null_mean"]:.3f} (SD {pm["null_sd"]:.3f}; range {pm["null_min"]:.3f} to {pm["null_max"]:.3f}), and the true wound region fell at its {pm["true_wound_percentile"]:.0f}th percentile, with {int(pm["n_ge_true_wound"])} of 300 relocations scoring at least as high (one-sided P={pm["one_sided_p_wound_better"]:.2f}','permutation null')
R(r'The single circular shift used in the locked analysis scores 0\.641 on this estimator, at the 84th percentile of the same distribution\. Its advantage over the true wound region is consequently about twice that of a typical relocation',
  f'The single circular shift used in the masked retraining scores {pm["locked_shift_auroc_rescored"]:.3f} on this estimator, at the {pm["locked_shift_percentile"]:.0f}th percentile of the same distribution. Its advantage over the true wound region is consequently about {(pm["locked_shift_auroc_rescored"]-pm["true_wound_auroc"])/(pm["null_mean"]-pm["true_wound_auroc"]):.1f} times that of a typical relocation','locked shift re-scored')
R(r'placed the true wound region at the 20th percentile: four fifths of randomly positioned regions of the same size and shape performed at least as well \(one-sided P=0\.80\)',
  f'placed the true wound region at the {pm["true_wound_percentile"]:.0f}th percentile: more than three quarters of randomly positioned regions of the same size and shape performed at least as well (one-sided P={pm["one_sided_p_wound_better"]:.2f})','Discussion permutation')
R(r'to have been an 84th-percentile draw, so its apparent advantage over the wound region was roughly twice a typical relocation’s',
  f'to have been an {pm["locked_shift_percentile"]:.0f}th-percentile draw, so its apparent advantage over the wound region was roughly {(pm["locked_shift_auroc_rescored"]-pm["true_wound_auroc"])/(pm["null_mean"]-pm["true_wound_auroc"]):.1f} times a typical relocation’s','Discussion shift draw')
R(r'The direction of that contrast did replicate across three training seeds, with a mean within-seed difference of \+0\.138 that survives family-wise correction, but all three seeds shared one shift realisation, so it speaks to training stability rather than to spatial specificity\.',
  f'On the postoperative frame the single-shift contrast itself was null ({s3(sw.d)}; P={sw.p:.2f}), so it no longer provides even supporting evidence.','Discussion single-shift null')
R(r'places the true wound region at the 20th percentile of that distribution',f'places the true wound region at the {pm["true_wound_percentile"]:.0f}th percentile of that distribution','Discussion pct 2')
R(r'placed the true wound region at the 20th percentile, so an arbitrary region',f'placed the true wound region at the {pm["true_wound_percentile"]:.0f}th percentile, so an arbitrary region','conclusion pct')

# ---------- CLINICAL ----------
c8,cv,im,st=cm.loc['clinical 8-variable'],cm.loc['clinical v2 5-variable'],cm.loc['image only'],cm.loc['stacked, 8-variable + image']
x8,xv=cc.loc['image adds to clinical 8-variable'],cc.loc['image adds to clinical v2 5-variable']
R(r'AUROC was 0\.664 \(95% CI, 0\.494–0\.813\) for the eight-variable clinical model, 0\.765 \(0\.666–0\.847\) for the image model, and 0\.800 \(0\.691–0\.890\) for the stacked model',
  f'AUROC was {f3(c8.auroc)} (95% CI, {f3(c8.lo)}–{f3(c8.hi)}) for the eight-variable clinical model, {f3(im.auroc)} ({f3(im.lo)}–{f3(im.hi)}) for the image model, and {f3(st.auroc)} ({f3(st.lo)}–{f3(st.hi)}) for the stacked model','clinical full-cohort AUROCs')
R(r'changed AUROC by \+0\.135 \(95% CI, \+0\.037 to \+0\.253; P=0\.005\) and AUPRC by −0\.024 \(P=0\.83\)',
  f'changed AUROC by {s3(x8.d_auroc)} (95% CI, {s3(x8.lo)} to {s3(x8.hi)}; P={x8.p:.3f})','clinical increment 8-var')
R(r'repeating the full nested procedure under five partitions gave differences of \+0\.025 to \+0\.135 and P values of 0\.004 to 0\.42, significant in two of five, while the AUPRC difference was negative in every specification',
  f'repeating the full nested procedure under five partitions gave differences of {s3(ps.d_auroc.min())} to {s3(ps.d_auroc.max())}, positive in all five','partition stability')
R(r'which reached an AUROC of 0\.705 \(95% CI, 0\.540–0\.849\) in 200 patients with 17 events, adding the image score changed AUROC by −0\.003 \(−0\.037 to \+0\.034; P=0\.82\)',
  f'which reached an AUROC of {f3(cv.auroc)} (95% CI, {f3(cv.lo)}–{f3(cv.hi)}) in {cv.n_pts} patients with {cv.n_events} events, adding the image score changed AUROC by {s3(xv.d_auroc)} ({s3(xv.lo)} to {s3(xv.hi)}; P={xv.p:.2f})','clinical v2')
R(r'The range-validated comparison remained nominally significant \(\+0\.135; P=0\.005\) but did not reproduce across cross-validation partitions, reaching significance in two of five, and the AUPRC difference was negative in every specification\.',
  f'Against the eight-variable intraoperative model the image score added {s3(x8.d_auroc)} (P={x8.p:.3f}), and the increment was positive in all five cross-validation partitions; against a five-variable specification containing established preoperative determinants it added {s3(xv.d_auroc)} (P={xv.p:.2f}).','Discussion clinical')

# ---------- REGION ----------
ab,ch,ri,wc=rs.loc['Abdomen'],rs.loc['Chest'],rs.loc['BENCHMARK region indicator alone'],rs.loc['BENCHMARK CDC wound class alone']
wtd=(ab.events*ab.auroc+ch.events*ch.auroc)/(ab.events+ch.events)
R(r'where the pooled estimate is 0\.754, stratifying by region gave 0\.466 \(95% CI, 0\.297–0\.635\) in the abdominal stratum, which contained 12 of the 14 events, and 0\.647 \(0\.272–0\.992\) in the chest stratum, which contained 2; weighting the two strata by their event counts gives 0\.492',
  f'where the pooled estimate is {f3(p7.auroc)}, stratifying by region gave {f3(ab.auroc)} (95% CI, {f3(ab.lo)}–{f3(ab.hi)}) in the abdominal stratum, which contained {int(ab.events)} of the 8 events, and {f3(ch.auroc)} ({f3(ch.lo)}–{f3(ch.hi)}) in the chest stratum, which contained {int(ch.events)}; weighting the two strata by their event counts gives {f3(wtd)}','region strata')
R(r'A single binary region indicator achieved 0\.793 \(0\.680–0\.879\) in the 187 patients with a classifiable region and 14 events, and CDC wound class alone 0\.781 \(0\.632–0\.919\) in the 163 patients with a recorded class and 11 events, each at least as high as the image model on its own frame',
  f'A single binary region indicator achieved {f3(ri.auroc)} ({f3(ri.lo)}–{f3(ri.hi)}) in the {int(ri.n)} patients with a classifiable region and {int(ri.events)} events, and CDC wound class alone {f3(wc.auroc)} ({f3(wc.lo)}–{f3(wc.hi)}) in the {int(wc.n)} patients with a recorded class and {int(wc.events)} events; the region indicator exceeded the image model and its interval excluded 0.5 where the image model’s did not','region benchmarks')

# ---------- CALIBRATION ----------
R(r'had a Brier score of 0\.070 \(95% CI, 0\.042–0\.102\) against a no-information value of 0\.068 for a model assigning every patient the cohort prevalence, giving a Brier skill score of −0\.03, calibration slope of 0\.67 \(95% CI, 0\.30–1\.15\), and calibration intercept of −0\.79',
  f'had a Brier score of {f3(ca["brier"])} against a no-information value of {f3(ca["no_information_brier"])} for a model assigning every patient the cohort prevalence, giving a Brier skill score of {ca["brier_skill"]:+.2f}, calibration slope of {ca["calibration_slope"]:.2f}, and calibration intercept of {ca["calibration_intercept"]:.2f}','calibration')

# ---------- INCISIONAL ----------
R(r'excluding organ-space 0\.714 \(0\.562–0\.843\), 10 events',f'excluding organ-space {f3(inc.iloc[1].auroc)} ({f3(inc.iloc[1].lo)}–{f3(inc.iloc[1].hi)}), {int(inc.iloc[1].events)} events','incisional 1')

# ---------- INTRAOPERATIVE REMOVAL ----------
R(r'In addition, intraoperative photographs were obtained before and after wound closure as part of the parent study, and these were retained in the analytic frame with a recorded postoperative day of zero\..*?they are reported here as a leakage control\.',
  'Intraoperative photographs were also obtained before and after wound closure as part of the parent study. Because such images cannot contain postoperative wound appearance, all 598 of them (280 obtained before closure and 318 after) were excluded from the analyses reported here. Forty-one photographs taken on the day of surgery after the participant had left the operating room are counted as postoperative and retained. A further nine photographs carried no derivable postoperative day and were also excluded.','intraoperative paragraph')

# ---------- GATE CLAIM ----------
R(r'Of the 3,739 photographs in the expanded, ungated frame, 1,948 \(52\.1%\) passed the gate: 48\.0% of submissions from SSI-positive patients and 52\.6% from SSI-negative patients\.',
  'The gate was applied during annotation, upstream of this analysis. A separately materialised uncurated export of 3,739 smartphone photographs from 228 patients, overlapping this cohort in 202 patients and including closure photographs, is used as a sensitivity frame; the two collections share no file identity, so a per-photograph gate-passage rate between them is not defined.','gate claim')

# ---------- Mahdi comment 10: flatters -> direction unknown ----------
R(r'this attrition is outcome-dependent in the direction that flatters performance',
  'this attrition is outcome-dependent, and its direction is not known: packed or vacuum-managed wounds may be the most obviously infected, so their exclusion could lower performance as readily as raise it','comment 10 direction')
# ---------- Mahdi comment 86: serial ----------
R(r'The POD 7 landmark and prediagnosis truncation were the principal prospective analyses\.',
  'The POD 7 landmark and prediagnosis truncation were the principal prospective analyses, and the constrained permutation test (Supplementary Table S10) is the primary spatial analysis.','comment 86')

doc.save(OUT)
print('\n'.join(LOG))
print(f'\n{sum(1 for l in LOG if l.startswith("OK"))} applied, {sum(1 for l in LOG if l.startswith("MISS"))} missed')
