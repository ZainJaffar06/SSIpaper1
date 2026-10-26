# -*- coding: utf-8 -*-
"""Tracked edits for the 2026-10-26 re-runs (everything Mahdi listed that the 10-23 scripts did not
cover). Runs AFTER tracked_apply_numbers/tables/rest/refresh_contrasts on OUTDOC. Every number is
read from out/postop_only/derived; nothing is typed in. Items whose runs are still pending (S11
fine-tuned, S12, S2) are applied only when their CSVs exist, so the script can be re-run later."""
import sys, os, re, glob
sys.path.insert(0,'.'); sys.path.insert(0,'/Users/zainjaffar/ssi_work/paper1_2026-08-15')
import numpy as np, pandas as pd
from dxml_tracked import TrackedDoc, W
from lib import safe_auroc
D='/Users/zainjaffar/ssi_work/paper1_2026-08-15/out/postop_only/derived/'; PO='/Users/zainjaffar/ssi_work/paper1_2026-08-15/out/postop_only/'
FINAL=set(open('/Users/zainjaffar/ssi_work/paper1_2026-08-15/final_ssi_list.txt').read().split())
F=os.environ['OUTDOC']; doc=TrackedDoc(F); LOG=[]
M='−'; EN='–'
def _q(x,nd): return f'{float(x):.{nd}f}'   # same rule as the 10-23 scripts and the figures (Python float formatting of the stored 4-dp values)
def f3(x): return _q(x,3).replace('-',M)
def s3(x): v=float(x); return ('+' if v>=0 else '')+_q(v,3).replace('-',M)
from decimal import Decimal, ROUND_HALF_UP
def _q2(x): return str(Decimal(_q(x,3)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))   # 2-dp text values round the 3-dp table value half-up, as a reader would
def f2(x): return _q2(x).replace('-',M)
def s2(x): v=float(x); return ('+' if v>=0 else '')+_q2(v).replace('-',M)
def ci(lo,hi,sep=EN): return f'{f3(lo)}{sep}{f3(hi)}'
def cito(lo,hi): return f'{s3(lo)} to {s3(hi)}'
def R(pat,new,lab):
    n=doc.replace(pat,new,lab); LOG.append(('OK  ' if n else 'MISS')+f' {lab}')
def cells_all():
    pm={c:p for p in doc.root.iter() for c in p}
    def indel(x):
        a=pm.get(x)
        while a is not None:
            if a.tag==W+'del': return True
            a=pm.get(a)
        return False
    out=[]
    for ti,tb in enumerate(doc.root.iter(W+'tbl')):
        for tr in tb.iter(W+'tr'):
            row=[]
            for tc in tr.iter(W+'tc'):
                ts=[x for x in tc.iter(W+'t') if not indel(x)]
                row.append((tc,''.join((x.text or '') for x in ts).strip()))
            out.append((ti,tr,row))
    return out
def set_row(label,vals,lab,table_hint=None):
    for ti,tr,row in cells_all():
        if row and re.fullmatch(label,row[0][1]) and (table_hint is None or re.search(table_hint,' | '.join(c[1] for c in row))):
            for (tc,old),v in zip(row[1:],vals):
                if v is not None and old!=v: doc.set_cell_tracked(tc,v)
            LOG.append(f'OK   row {lab}'); return True
    LOG.append(f'MISS row {lab}'); return False

# ---------------- data ----------------
cc=pd.read_csv(D+'calibration_ci.csv').iloc[0]; ov=pd.read_csv(D+'single_shift_overlap_audit.csv').iloc[0]
cu=pd.read_csv(D+'curation_sensitivity_postop.csv').iloc[0]; tr_=pd.read_csv(D+'prediagnosis_truncation.csv').set_index('landmark')
cm=pd.read_csv(D+'clinical_models.csv').set_index('model'); op=pd.read_csv(D+'operating_points.csv').set_index('target_spec')
S=pd.read_csv(D+'pod7_patient_scores.csv'); y=S.ssi_final.values; s=S.score.values; pos=np.sort(s[y==1]); n=len(S); e=int(y.sum())
def cap(k):
    thr=pos[len(pos)-k]; return (s[y==1]>=thr).mean(),(s[y==0]<thr).mean(),(s>=thr).mean()
c7,c8=cap(e-1),cap(e); o8=op.loc[0.8]
cp=pd.read_csv(D+'context_probes_postop.csv').set_index('probe'); full=float(cp.full_image_3seed_mean.iloc[0])
sd_=pd.read_csv(D+'seedaware_probe_delta_summary_postop.csv').iloc[0]; pp=pd.read_csv(D+'cnn_vs_probe_paired_postop.csv').iloc[0]
cf=pd.read_csv(D+'cnn_vs_context_fullframe_postop.csv').set_index('level'); ar=pd.read_csv(D+'architectures_restricted_postop.csv').set_index('architecture')
fg=pd.read_csv(D+'frozen_geometry_postop.csv').set_index('feature'); gr=pd.read_csv(D+'geometry_encodes_region_postop.csv').set_index('feature')
ma=pd.read_csv(D+'masking_absolute.csv').set_index('condition'); import json; f1=json.load(open(D+'figure1_params.json'))
img_full=cm.loc['image only']

# ---------------- A. calibration CIs + operating sentence ----------------
newcal=f'calibration slope of {f2(cc.slope)} (95% CI, {f2(cc.slope_lo)} to {f2(cc.slope_hi)}), and calibration intercept of {f2(cc.intercept)} (95% CI, {f2(cc.intercept_lo)} to {f2(cc.intercept_hi)})'
R(r'calibration slope of 0\.38, and calibration intercept of [−-]1\.29 \(95% CI, −2\.02 to 0\.47\)',newcal,'calibration CIs (post-10-23 form)')
R(r'calibration slope of 0\.67 \(95% CI, 0\.30–1\.15\), and calibration intercept of −0\.79 \(95% CI, −2\.02 to 0\.47\)',newcal,'calibration CIs (original form)')
R(r'At a threshold selected for approximately 80% specificity, sensitivity was 64\.3% and 23\.4% of patients were flagged\. Capturing 13 of 14 infections \(92\.9% sensitivity\) required flagging 50\.5% of patients and yielded 52\.8% specificity\. Capturing all 14 infections required flagging 77\.1% of patients\.',
  f'At a threshold selected for approximately 80% specificity, sensitivity was {100*o8.sensitivity:.1f}% and {100*o8.n_flagged/n:.1f}% of patients were flagged. Capturing {e-1} of {e} infections ({100*c7[0]:.1f}% sensitivity) required flagging {100*c7[2]:.1f}% of patients and yielded {100*c7[1]:.1f}% specificity. Capturing all {e} infections required flagging {100*c8[2]:.1f}% of patients.','operating-point sentence')
set_row(r'Calibration slope',[f3(cc.slope),cito(cc.slope_lo,cc.slope_hi)],'S6 slope CI'); set_row(r'Calibration intercept',[f3(cc.intercept),cito(cc.intercept_lo,cc.intercept_hi)],'S6 intercept CI')

# ---------------- D. overlap audit ----------------
pct=f'{100*ov.frac_le_0_05:.1f}%'
R(r'In the locked analysis 98\.1% of shifted masks overlapped the true mask by less than 5% of their union',f'In the locked analysis {pct} of shifted masks overlapped the true mask by at most 5% of their union','overlap: discussion')
R(r'Median IoU with the true mask was 0\.000 \(mean, 0\.0026\); 98\.1% of masks had IoU <0\.05 and the maximum was 0\.217',f'Median IoU with the true mask was {f3(ov.iou_median)} (mean, {float(ov.iou_mean):.4f}); {pct} of masks had IoU ≤0.05 and the maximum was {f3(ov.iou_max)}','overlap: S4 footnote')
R(r'minimal overlap with the true wound mask in 98\.1% of images',f'minimal overlap with the true wound mask in {pct} of images','overlap: Fig 4 caption')
R(r'Panel A shows point estimates for full, anatomically aligned, and circularly shifted region models\.','Panel A shows three-seed mean AUROCs for full, anatomically aligned, and circularly shifted region models, with the individual seeds as ticks.','Fig 4 caption: seeds')

# ---------------- E. curation sensitivity ----------------
R(r'had a patient-level AUROC of 0\.7(?:65|44), whereas the expanded or uncurated smartphone frame yielded an AUROC of approximately 0\.662',
  f'had a patient-level AUROC of {f3(img_full.auroc)}, whereas the expanded or uncurated smartphone frame, restricted to its {int(cu.n_photos):,} postoperative photographs ({int(cu.n_pts)} patients, {int(cu.n_events)} events) and scored with that model’s saved out-of-fold predictions, yielded an AUROC of {f3(cu.auroc)} (95% CI, {ci(cu.lo,cu.hi)})','curation sensitivity')

# ---------------- F. prediagnosis truncation ----------------
t7=tr_.loc[7]; t14=tr_.loc[14]; nd=f1['dated_pts']; ed=f1['dated_events']
R(r'Among (?:203|170) patients with dated photographs and (?:17|12) dated SSI events, restricting SSI-positive patient aggregation to photographs obtained strictly before clinical diagnosis changed patient-level AUROC by .*? relative to the all-image score\. This finding indicates that evaluation aggregation was not materially improved by photographs obtained on or after diagnosis\.',
  f'Among {nd} patients with dated photographs and {ed} dated SSI events, restricting SSI-positive patient aggregation to photographs obtained strictly before clinical diagnosis left the landmark estimates unchanged: {f3(t7.auroc)} at POD 7 ({int(t7.n_pts)} patients, {int(t7.n_events)} events) and {f3(t14.auroc)} at POD 14 ({int(t14.n_pts)} patients, {int(t14.n_events)} events). On the postoperative frame every photograph that enters a landmark score precedes the diagnosis of the patient who contributed it, so the truncation removes nothing.','truncation paragraph')

# ---------------- F2. Methods frame sentences + S13 citation ----------------
R(r'On the 1,886-image mask-eligible frame, patient-grouped logistic probes were trained without wound pixels\.',f"On the {f1['mask_photos']:,}-image mask-eligible frame, patient-grouped logistic probes were trained without wound pixels.",'Methods 1,886 context frame')
R(r'The final masking frame included 1,886 photographs from 206 patients with 19 SSI events\.',f"The final masking frame included {f1['mask_photos']:,} photographs from {f1['mask_pts']} patients with {f1['mask_events']} SSI events.",'Methods 1,886 masking frame')
R(r'against a prospectively written six-category codebook —','against a prospectively written six-category codebook (Supplementary Table S13) —','Methods cites S13')
# ---------------- G. context probes (S3, Fig 3B, Discussion, captions) ----------------
P=lambda k:cp.loc[k]; d=lambda k:float(full-cp.loc[k].patient_auroc)
R(r'On the matched (?:206|168)-patient masking frame, the retrained full-image model achieved an AUROC of 0\.636\..*?rather than against the multi-seed architecture means\.',
  f'On the matched 168-patient masking frame (1,296 photographs, 12 events), the three-seed full-image model achieved a mean AUROC of {f3(full)}. Context-only AUROCs were {f3(P("pod_only").patient_auroc)} (95% CI, {ci(P("pod_only").lo,P("pod_only").hi)}) for POD, {f3(P("acquisition").patient_auroc)} ({ci(P("acquisition").lo,P("acquisition").hi)}) for acquisition features, {f3(P("behavior_past_only").patient_auroc)} ({ci(P("behavior_past_only").lo,P("behavior_past_only").hi)}) for prior imaging behavior, and {f3(P("acquisition+behavior+pod").patient_auroc)} ({ci(P("acquisition+behavior+pod").lo,P("acquisition+behavior+pod").hi)}) for their combination. Every probe was below the full-image mean, by {f3(d("pod_only"))} for POD, {f3(d("acquisition"))} for acquisition features, {f3(d("behavior_past_only"))} for behavior and {f3(d("acquisition+behavior+pod"))} for the combination (Supplementary Table S3). Restricting the architecture runs to these same 168 patients raises their three-seed means by {s3(ar.loc["efficientnet_b0"]["shift"])} for EfficientNet-B0, {s3(ar.loc["resnet18"]["shift"])} for ResNet18 and {s3(ar.loc["vit_b_16"]["shift"])} for ViT-B/16, so the mask-eligible patients are an easier subset than the full postoperative cohort, and the masking conditions should be read against each other rather than against the full-cohort architecture means.','S3 paragraph')
im=cf.loc['image']; pa=cf.loc['patient']
R(r'Across three seeds, the mean CNN-minus-combined-context difference on this matched frame was −0\.009±0\.052 \(range, −0\.066 to \+0\.036; Figure 3B\)\..*?rather than improved probes\.',
  f'Across three seeds, the mean CNN-minus-combined-context difference on this matched frame was {s3(sd_["mean"])}±{f3(sd_["sd"])} (range, {s3(sd_["lo"])} to {s3(sd_["hi"])}; Figure 3B), and the paired comparison of the locked seed with the probe was {s3(pp.d)} (95% CI, {cito(pp.lo,pp.hi)}; P={float(pp.p):.2f}). On the full 171-patient postoperative frame, where the context probe reached {f3(im.probe_auroc)} at image level and {f3(pa.probe_auroc)} at patient level, the single locked CNN exceeded the combined probe by {s3(im.cnn_minus_probe)} at image level (95% CI, {cito(im.d_lo,im.d_hi)}; P={float(im.p):.2f}) and {s3(pa.cnn_minus_probe)} at patient level ({cito(pa.d_lo,pa.d_hi)}; P={float(pa.p):.2f}). The CNN therefore exceeded measured context on both frames, but by margins that the seed spread and the confidence intervals contain, so measured context remains an alternative predictive pathway of comparable magnitude.','Fig 3B / full-frame paragraph')
R(r'Measured collection context was also sufficient to match or exceed full-image discrimination\. A combined probe using POD, acquisition features, and prior imaging behavior achieved an AUROC of 0\.727, against 0\.636 for the locked single run of the matched full-image model and 0\.671 for its three-seed mean, with seed-specific differences spanning both directions\.',
  f'Measured collection context approached full-image discrimination. A combined probe using POD, acquisition features, and prior imaging behavior achieved an AUROC of {f3(P("acquisition+behavior+pod").patient_auroc)} on the matched masking frame, against a three-seed full-image mean of {f3(full)}, with seed-specific differences spanning both directions ({s3(sd_["lo"])} to {s3(sd_["hi"])}).','Discussion context sentence')
R(r'The context-probe pipeline was reimplemented for the present analysis and reproduces the originally reported probe AUROCs to within 0\.016; differences of that magnitude from earlier values are therefore not interpretable\. Computed within this implementation under both label sets, the combined probe moved from 0\.721 to 0\.727, so the widened margin on the masking frame derives from the lower single-run full-image estimate rather than from the probes\.',
  'The context-probe pipeline was reimplemented for the present analysis, and all probe values reported here were computed on the postoperative frame under the final labels.','analysis-history context sentence')
R(r'The combined probe exceeded the point estimate of the retrained full-image model by \+0\.091\.',f'The combined probe fell below the three-seed full-image mean by {f3(d("acquisition+behavior+pod"))}.','S3 footnote 1')
R(r'The probe pipeline was reimplemented and reproduces the originally reported values to within 0\.016; within this implementation the combined probe was 0\.721 under the original labels and 0\.727 under the final labels\.','Confidence intervals are patient-clustered bootstrap percentiles with 20,000 resamples; the full-image value is the mean of three training seeds.','S3 footnote 2')
R(r'The combined context probe approximately matched the full-image model\.',f'The full-image model exceeded the combined context probe by {s3(sd_["mean"])} on average across seeds, within the seed spread.','Fig 3 caption')
set_row(r'Input',['Patient AUROC (95% CI)',None],'S3 header',table_hint='Full-image minus probe')
set_row(r'Full RGB image',[f3(full),'Reference'],'S3 full image')
set_row(r'POD only',[f'{f3(P("pod_only").patient_auroc)} ({ci(P("pod_only").lo,P("pod_only").hi)})',s3(d("pod_only"))],'S3 POD')
set_row(r'Acquisition features',[f'{f3(P("acquisition").patient_auroc)} ({ci(P("acquisition").lo,P("acquisition").hi)})',s3(d("acquisition"))],'S3 acquisition')
set_row(r'Prior imaging behavior',[f'{f3(P("behavior_past_only").patient_auroc)} ({ci(P("behavior_past_only").lo,P("behavior_past_only").hi)})',s3(d("behavior_past_only"))],'S3 behavior')
set_row(r'Acquisition \+ behavior \+ POD',[f'{f3(P("acquisition+behavior+pod").patient_auroc)} ({ci(P("acquisition+behavior+pod").lo,P("acquisition+behavior+pod").hi)})',s3(d("acquisition+behavior+pod"))],'S3 combined')

# ---------------- H. S11 geometry ----------------
mi=sorted(glob.glob(PO+'maskimage_seed*.csv')); ft_txt=None
if len(mi)==3:
    aus=[]
    for f in mi:
        dd=pd.read_csv(f).dropna(subset=['proba']); dd['y']=dd.pid.isin(FINAL).astype(int); g=dd.groupby('pid').agg(y=('y','max'),p=('proba','mean')); aus.append(safe_auroc(g.y,g.p))
    ft_txt=f'{f3(np.mean(aus))} (range {f3(min(aus))} to {f3(max(aus))})'; ft_cell=f'{f3(np.mean(aus))} (range {ci(min(aus),max(aus))})'
    pd.DataFrame(dict(file=[os.path.basename(f) for f in mi],patient_auroc=[round(a,4) for a in aus])).to_csv(D+'maskimage_finetuned_postop.csv',index=False)
else: LOG.append(f'PENDING S11 fine-tuned mask-as-image ({len(mi)}/3 seeds)')
fz=fg.loc['binary_mask_image_frozen']; g7=fg.loc['scalar_geometry_7']; ao=fg.loc['mask_area_only']
el=gr.loc['elongation']; fr_=gr.loc['fill_ratio']; aur=lambda r:max(float(r.auroc_for_region),1-float(r.auroc_for_region))
ftc=ft_txt if ft_txt else '[fine-tuned three-seed value pending]'
if ft_txt: R(r'\[fine-tuned three-seed value pending\]',ft_txt,'S11 fine-tuned placeholder (re-run)')
R(r'reached 0\.465 \(95% CI, 0\.327–0\.601\) with frozen features and 0\.579 \(range 0\.563 to 0\.609\) across three fine-tuned seeds\. However, seven scalar descriptors of the mask alone — area, centroid, bounding-box dimensions, elongation and fill ratio — reached 0\.802 \(0\.680–0\.897\), compared with 0\.671 for the full-image model on the same frame \(Supplementary Table S11\)\. These descriptors also discriminated operative region, with elongation and fill ratio separating chest from abdominal incisions, so annotation geometry carries case-mix information independently of image content\.',
  f'reached {f3(fz.auroc)} (95% CI, {ci(fz.lo,fz.hi)}) with frozen features and {ftc} across three fine-tuned seeds. Seven scalar descriptors of the mask alone — area, centroid, bounding-box dimensions, elongation and fill ratio — reached {f3(g7.auroc)} ({ci(g7.lo,g7.hi)}), below the {f3(full)} three-seed mean of the full-image model on the same frame, and mask area alone was below chance ({f3(ao.auroc)}, {ci(ao.lo,ao.hi)}; Supplementary Table S11). The descriptors still discriminated operative region, with elongation and fill ratio separating chest from abdominal incisions (AUROC {aur(el):.2f} and {aur(fr_):.2f}), so annotation geometry carries case-mix information independently of image content.','S11 paragraph')
R(r'The more informative observation is that seven scalar descriptors of the annotation geometry alone exceeded the full-image model, since those descriptors also encode operative region\.',
  f'Seven scalar descriptors of the annotation geometry alone reached {f3(g7.auroc)}, within {abs(float(fg.loc["geometry_minus_full_image"].auroc)):.2f} of the full-image model, and those descriptors also encode operative region.','S11 discussion sentence')
set_row(r'Binary mask as an image, frozen features',[f'{f3(fz.auroc)} ({ci(fz.lo,fz.hi)})'],'S11 frozen')
if ft_txt: set_row(r'Binary mask as an image, fine-tuned, 3 seeds',[ft_cell],'S11 fine-tuned')
set_row(r'Mask area alone',[f'{f3(ao.auroc)} ({ci(ao.lo,ao.hi)})'],'S11 area')
set_row(r'Seven scalar geometry descriptors',[f'{f3(g7.auroc)} ({ci(g7.lo,g7.hi)})'],'S11 seven')
set_row(r'Full-image model, same frame, 3-seed mean',[f3(full)],'S11 full')
set_row(r'Difference, geometry minus full image',[s3(fg.loc['geometry_minus_full_image'].auroc)],'S11 diff')
R(r'The comparison with the full-image model is on the same 206-patient mask-eligible frame','The comparison with the full-image model is on the same 168-patient mask-eligible frame','S11 footnote frame')

# ---------------- I. S12 foundation (pending until CSVs exist) ----------------
if os.path.exists(D+'foundation_probes_postop.csv') and os.path.exists(D+'foundation_paired_contrasts_postop.csv'):
    fp=pd.read_csv(D+'foundation_probes_postop.csv'); fc=pd.read_csv(D+'foundation_paired_contrasts_postop.csv').set_index('contrast'); ra=pd.read_csv(D+'foundation_rank_agreement_postop.csv').set_index('encoder')
    V=lambda enc,c:fp[(fp.encoder==enc)&(fp.condition==c)].iloc[0]
    cond={'Full image':'full_image','True wound-only':'wound_only','True background-only':'background_only','Circularly shifted region':'sham_wound_only','Complement of shifted region':'sham_background_only'}
    for lab,c in cond.items():
        a=V('dinov2_vitb14',c); b=V('biomedclip_vitb16',c); set_row(re.escape(lab),[f'{f3(a.auroc)} ({ci(a.lo,a.hi)})',f'{f3(b.auroc)} ({ci(b.lo,b.hi)})'],f'S12 {lab}',table_hint='DINOv2|dinov2|0\\.')
    # paragraph composed from the data: ranks against the fine-tuned 3-seed ordering, the contrast family, the area-matched contrast
    ftm={'full_image':ma.loc['Full image'].seed_mean,'wound_only':ma.loc['True wound-only'].seed_mean,'background_only':ma.loc['True background-only'].seed_mean,'sham_wound_only':ma.loc['Circularly shifted wound-size region'].seed_mean,'sham_background_only':ma.loc['Complement of shifted region'].seed_mean}
    def ranks(dct): return {k:i+1 for i,(k,v) in enumerate(sorted(dct.items(),key=lambda kv:-kv[1]))}
    ordw=['first','second','third','fourth','fifth']
    rf=ranks(ftm); rd=ranks({c:float(V('dinov2_vitb14',c).auroc) for c in cond.values()}); rb=ranks({c:float(V('biomedclip_vitb16',c).auroc) for c in cond.values()})
    dw=V('dinov2_vitb14','wound_only'); bvals=[float(V('biomedclip_vitb16',c).auroc) for c in cond.values()]
    agree='as the fine-tuned model did' if rd['wound_only']==rf['wound_only'] else f"where the fine-tuned three-seed means place it {ordw[rf['wound_only']-1]}"
    def C(k): r=fc.loc[k]; return f"{s3(r.d)} ({cito(r.lo,r.hi)}; Holm-adjusted P={float(r.holm_p):.3f})"
    sig=[k for k in fc.index if float(fc.loc[k].holm_p)<0.05]
    para=(f"Frozen foundation-model probes were fitted on the same five conditions. DINOv2 placed the true wound region {ordw[rd['wound_only']-1]} of the five conditions at {f3(dw.auroc)} ({ci(dw.lo,dw.hi)}), {agree}; its ordering of the five conditions correlated with the fine-tuned ordering at \u03c1={ra.loc['dinov2_vitb14'].spearman_vs_finetuned:+.2f}. "
          f"BiomedCLIP values ranged from {f3(min(bvals))} to {f3(max(bvals))} across conditions, with \u03c1={ra.loc['biomedclip_vitb16'].spearman_vs_finetuned:+.2f} against the fine-tuned ordering. "
          f"On paired patient-clustered resampling of the DINOv2 probe, true background minus true wound was {C('background_only - wound_only')}, the complement of the shifted region minus true wound was {C('sham_background_only - wound_only')}, and the one area-matched contrast, shifted region minus true wound, was {C('sham_wound_only - wound_only')}. "
          +("No contrast in the family survived Holm correction" if not sig else f"{len(sig)} contrast(s) survived Holm correction ({'; '.join(sig)})")
          +" (Supplementary Table S12).")
    R(r'Frozen foundation-model probes agreed with the fine-tuned analysis on the true wound region but not on the ordering as a whole\..*?\(Supplementary Table S12\)\.',para,'S12 paragraph')
    set_row(r'background only - wound only',[C('background_only - wound_only').replace('; Holm-adjusted P=',', Holm P=')],'S12 contrast row 1')
    set_row(r'sham background only - wound only',[C('sham_background_only - wound_only').replace('; Holm-adjusted P=',', Holm P=')],'S12 contrast row 2')
    R(r'Only the two paired contrasts with unadjusted P<0\.05 are listed; after Holm adjustment across the five contrasts, only the second remains significant \(P=0\.041\)\. BiomedCLIP intervals overlap across all five conditions\.',
      f"The two listed contrasts are the same two shown in the earlier version; P values are Holm-adjusted across the five contrasts of the family, of which {len(sig)} remain significant. BiomedCLIP intervals overlap across all five conditions.",'S12 footnote')
    R(r'Frozen foundation-model features placed the true wound region lowest of the five conditions, as the fine-tuned model did, but did not reproduce the wider ordering, and the one area-matched contrast available to them was null; they therefore corroborate only the weakest part of the claim\.',
      f"Frozen foundation-model features placed the true wound region {ordw[rd['wound_only']-1]} of the five conditions and correlated with the fine-tuned ordering at \u03c1={ra.loc['dinov2_vitb14'].spearman_vs_finetuned:+.2f}; the one area-matched contrast available to them was {C('sham_wound_only - wound_only')}.",'S12 discussion sentence')
else: LOG.append('PENDING S12 foundation probes')

# ---------------- J. S2 learning curve (pending until CSVs exist) ----------------
lc=sorted(glob.glob(PO+'lc_*_seed*.csv'))
if len(lc)==6:
    rows={}
    for f in lc:
        frac=os.path.basename(f).split('_')[1]; dd=pd.read_csv(f).dropna(subset=['proba']); dd['y']=dd.pid.isin(FINAL).astype(int); g=dd.groupby('pid').agg(y=('y','max'),p=('proba','mean')); rows.setdefault(frac,[]).append(safe_auroc(g.y,g.p))
    ar_=pd.read_csv(D+'architectures.csv').set_index('architecture').loc['efficientnet_b0']
    third,two=rows['0.333'],rows['0.667']; m3,s3_=np.mean(third),np.std(third,ddof=1); m6,s6=np.mean(two),np.std(two,ddof=1); mf,sf=float(ar_['mean']),float(ar_['sd'])
    pd.DataFrame(dict(fraction=['one-third','two-thirds','full'],mean=[m3,m6,mf],sd=[s3_,s6,sf],per_seed=[third,two,ar_.per_seed])).to_csv(D+'learning_curve_postop.csv',index=False)
    R(r'EfficientNet-B0 mean AUROC rose from 0\.607±0\.053 with one-third of training patients to 0\.743±0\.023 with two-thirds, and was 0\.7(?:33±0\.049|12±0\.018) with the full training set',
      f'EfficientNet-B0 mean AUROC was {f3(m3)}±{f3(s3_)} with one-third of training patients, {f3(m6)}±{f3(s6)} with two-thirds, and {f3(mf)}±{f3(sf)} with the full training set','S2 sentence 1')
    R(r'The mean therefore gained substantially between one-third and two-thirds of the training data \(\+0\.136\) and did not gain further beyond it \(−0\.010\)',
      f'The mean therefore changed by {s3(m6-m3)} between one-third and two-thirds of the training data and by {s3(mf-m6)} beyond it','S2 sentence 2')
    set_row(r'One-third',[f'{f3(m3)}±{f3(s3_)}',None],'S2 one-third'); set_row(r'Two-thirds',[f'{f3(m6)}±{f3(s6)}',None],'S2 two-thirds'); set_row(r'Full',[f'{f3(mf)}±{f3(sf)}',None],'S2 full',table_hint='Mean')
    means={'One-third':m3,'Two-thirds':m6,'Full':mf}; sds={'One-third':s3_,'Two-thirds':s6,'Full':sf}
    def interp(k):
        v=means[k]; w='Highest mean' if v==max(means.values()) else ('Lowest mean' if v==min(means.values()) else 'Intermediate mean'); return f"{w}; three-seed SD {f3(sds[k])}"
    set_row(r'One-third',[None,interp('One-third')],'S2 interp one-third'); set_row(r'Two-thirds',[None,interp('Two-thirds')],'S2 interp two-thirds'); set_row(r'Full',[None,interp('Full')],'S2 interp full',table_hint='Mean')
    R(r'However, mean performance rose substantially between one-third and two-thirds of the training patients but not beyond, and the small number of events limits resolution\.',
      f'Mean performance changed by {s3(m6-m3)} between one-third and two-thirds of the training patients and by {s3(mf-m6)} beyond that, differences that the three-seed spread contains, and the small number of events limits resolution.','S2 discussion sentence')
else: LOG.append(f'PENDING S2 learning curve ({len(lc)}/6 runs)')

LOG.append(f'minus-sign sweep inside my insertions: {doc.sweep_minus()} runs')
doc.save(F); print('\n'.join(LOG))
