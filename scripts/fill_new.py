# -*- coding: utf-8 -*-
"""Fill Mahdi's 2026-10-26 version (MB-Clean) as tracked changes: 66 [PENDING] placeholders, 12
[Reference] placeholders + mapping fixes from his REFERENCES comment, 7 new verified entries, four
postoperative figures, Table 1/2 corrections, and replies to his comments. Every number is read from
out/postop_only/derived or computed here from the frame files; nothing is typed in."""
import sys, os, re, json, glob, copy, zipfile, shutil, datetime, random
sys.path.insert(0,'.'); sys.path.insert(0,'/Users/zainjaffar/ssi_work/paper1_2026-08-15')
import numpy as np, pandas as pd
from lxml import etree as ET
from tracked_lx import TrackedDocx, W, AUTHOR
from lib import safe_auroc, dedup
P1='/Users/zainjaffar/ssi_work/paper1_2026-08-15/'; D=P1+'out/postop_only/derived/'; PO=P1+'out/postop_only/'
REPO='/Users/zainjaffar/Documents/New project/ssi-paper1-information-audit/'
SRC='new/MB-Clean.docx'; OUT='new/MB-Clean-FILLED.docx'
M='−'; EN='–'
def f3(x): return f'{float(x):.3f}'.replace('-',M)
def s3(x): return f'{float(x):+.3f}'.replace('-',M)
def f2h(x):  # 2-dp text value = half-up of the 3-dp value
    from decimal import Decimal, ROUND_HALF_UP; return str(Decimal(f'{float(x):.3f}').quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)).replace('-',M)
def ci(lo,hi): return f'{f3(lo)}{EN}{f3(hi)}'
def cito(lo,hi): return f'{s3(lo)} to {s3(hi)}'
d=TrackedDocx(SRC); R=d.replace; CH=[]   # change log entries (where, what)
def RR(pat,new,lab):
    n=R(pat,new,lab)
    if n: CH.append((lab,new if len(new)<160 else new[:157]+'...'))
    return n
# ================= data =================
FINAL=set(open(P1+'final_ssi_list.txt').read().split())
cc=pd.read_csv(D+'calibration_ci.csv').iloc[0]; ov=pd.read_csv(D+'single_shift_overlap_audit.csv').iloc[0]; cu=pd.read_csv(D+'curation_sensitivity_postop.csv').iloc[0]
cp=pd.read_csv(D+'context_probes_postop.csv').set_index('probe'); full=float(cp.full_image_3seed_mean.iloc[0]); P_=lambda k:cp.loc[k]; dd=lambda k:float(full-cp.loc[k].patient_auroc)
sd_=pd.read_csv(D+'seedaware_probe_delta_summary_postop.csv').iloc[0]; pp=pd.read_csv(D+'cnn_vs_probe_paired_postop.csv').iloc[0]
cf=pd.read_csv(D+'cnn_vs_context_fullframe_postop.csv').set_index('level'); im=cf.loc['image']; pa=cf.loc['patient']
fg=pd.read_csv(D+'frozen_geometry_postop.csv').set_index('feature'); gr=pd.read_csv(D+'geometry_encodes_region_postop.csv').set_index('feature')
fz=fg.loc['binary_mask_image_frozen']; g7=fg.loc['scalar_geometry_7']; ao=fg.loc['mask_area_only']; aur=lambda r:max(float(r.auroc_for_region),1-float(r.auroc_for_region))
mi=pd.read_csv(D+'maskimage_finetuned_postop.csv'); ft_mean=mi.patient_auroc.mean(); ft_lo=mi.patient_auroc.min(); ft_hi=mi.patient_auroc.max()
fp=pd.read_csv(D+'foundation_probes_postop.csv'); fc=pd.read_csv(D+'foundation_paired_contrasts_postop.csv').set_index('contrast'); ra=pd.read_csv(D+'foundation_rank_agreement_postop.csv').set_index('encoder')
V=lambda enc,c:fp[(fp.encoder==enc)&(fp.condition==c)].iloc[0]
lc=pd.read_csv(D+'learning_curve_postop.csv').set_index('fraction'); m3,s3v=lc.loc['one-third','mean'],lc.loc['one-third','sd']; m6,s6=lc.loc['two-thirds','mean'],lc.loc['two-thirds','sd']; mf,sf=lc.loc['full','mean'],lc.loc['full','sd']
tr_=pd.read_csv(D+'prediagnosis_truncation_fullcohort_postop.csv').iloc[0]
ma=pd.read_csv(D+'masking_absolute.csv').set_index('condition')
# frame facts computed from the frame files
fr=pd.read_csv(P1+'out/recompute/per_photo_frame_1948.csv'); keep=np.load(P1+'out/recompute/postop_row_mask.npy'); master=pd.read_csv(P1+'out/paper1/MASTER_map_1948_seed42.csv'); P=master.loc[keep]; pids=sorted(set(P.pid)); ssi=[p for p in pids if p in FINAL]
b=dedup(pd.read_csv(P1+'base/ssi_smartphone.csv',low_memory=False)); DX=b.assign(dd=pd.to_numeric(b.dx_pod,errors='coerce')).groupby('pid').dd.first()
dx={p:DX.get(p,np.nan) for p in ssi}; n30=sum(1 for v in dx.values() if not pd.isna(v) and v<=30); late=sorted(int(v) for v in dx.values() if not pd.isna(v) and v>30); nund=sum(1 for v in dx.values() if pd.isna(v))
assert len(ssi)==13 and n30+len(late)+nund==13
rp=pd.read_csv(REPO+'results/MASTER_map_1948_seed42.csv'); cnt=P.assign(image_id=rp.image_id.values[keep]).groupby('pid').size(); yy=cnt.index.isin(FINAL)
from decimal import Decimal as _D, ROUND_HALF_UP as _H
hu=lambda v:int(_D(str(float(v))).quantize(_D('1'),rounding=_H))
q=lambda s:(hu(s.median()),hu(s.quantile(.25)),hu(s.quantile(.75)))   # Q3 of the overall count is 9.5 -> 10, as Mahdi computed
qa,qs,qn=q(cnt),q(cnt[yy]),q(cnt[~yy])
B=pd.read_csv(os.environ.get('TRACKER_CSV','<local PHI tracker, not in repo>'),encoding='latin-1',low_memory=False); B.columns=[c.strip() for c in B.columns]
ntrk=len([p for p in pids if p not in set(B['SUBJECT ID'].astype(str).str.strip().str.upper().str.replace(' ','',regex=False))])
cov=b.drop_duplicates('pid').set_index('pid'); cov=cov[cov.index.isin(pids)]; nimpl=0
for c in ('closure_temp_c','nadir_temp_c'):
    v=pd.to_numeric(cov[c],errors='coerce'); nimpl+=int(((v<30)|(v>43)).sum())
rl=pd.read_csv(P1+'out/recompute/region_labels_FOR_REVIEW.csv').set_index('patient_id'); nreg=sum(1 for p in pids if p in rl.index and not pd.isna(rl.loc[p,'region']))
# truncation P (mid-P, ties counted half)
dtr=pd.read_csv(PO+'resnet18_seed20260807.csv').dropna(subset=['proba']); dtr['y']=dtr.pid.isin(FINAL).astype(int); day=fr.set_index('image_id').day_after_surgery; dtr['day']=dtr.image_id.map(day); dtr['dx']=dtr.pid.map(DX)
dated=dtr[(dtr.y==0)|dtr.dx.notna()]; allg=dated.groupby('pid').agg(y=('y','max'),p=('proba','mean')); pre=dated[(dated.y==0)|(dated.day<dated.dx)]; preg=pre.groupby('pid').agg(y=('y','max'),p=('proba','mean'))
common=allg.index.intersection(preg.index); A=allg.loc[common]; Bq=preg.loc[common]; yv=A.y.values; rng=np.random.default_rng(913); reps=[]
for _ in range(20000):
    ii=rng.integers(0,len(common),len(common))
    if len(np.unique(yv[ii]))<2: continue
    reps.append(safe_auroc(yv[ii],Bq.p.values[ii])-safe_auroc(yv[ii],A.p.values[ii]))
reps=np.array(reps); d0=safe_auroc(yv,Bq.p)-safe_auroc(yv,A.p); ptr=min(1.0,2*min((reps<0).mean()+0.5*(reps==0).mean(),(reps>0).mean()+0.5*(reps==0).mean()))
tr_lo,tr_hi=np.percentile(reps,2.5),np.percentile(reps,97.5); nlost=len(allg)-len(common)
pd.DataFrame([dict(n_pts=len(common),n_events=int(yv.sum()),auroc_all=round(safe_auroc(yv,A.p),4),auroc_prediagnosis=round(safe_auroc(yv,Bq.p),4),delta=round(d0,4),lo=round(tr_lo,4),hi=round(tr_hi,4),p_midp=round(ptr,4),ssi_pts_without_prediagnosis_photo=nlost)]).to_csv(D+'prediagnosis_truncation_fullcohort_postop.csv',index=False)
print(f'truncation: delta {d0:+.4f} ({tr_lo:+.4f} to {tr_hi:+.4f}) P={ptr:.2f}; lost {nlost}')
# ================= A. cohort / outcome counts =================
RR(r'\[PENDING\] were diagnosed within 30 days, \[PENDING\] were documented later, and one had no recorded diagnosis date',
   f'{n30} were diagnosed within 30 days, {"one was" if len(late)==1 else str(len(late))+" were"} documented later, and one had no recorded diagnosis date','p7 SSI timing split')
RR(r'\[PENDING\] infections in the postoperative frame were documented later, on \[PENDING\], and are retained',
   ("one infection in the postoperative frame was documented later, on POD "+str(late[0])+", and is retained") if len(late)==1 else (f"{len(late)} infections in the postoperative frame were documented later, on POD {' and POD '.join(map(str,late))}, and are retained"),'p52 late infections')
RR(r'\[PENDING\] of the 171 patients had no record in the source clinical tracker',f'{"Two" if ntrk==2 else ntrk} of the 171 patients had no record in the source clinical tracker','Table 1 footnote: tracker')
RR(r'available for \[PENDING\] of 171 patients and unambiguous',f'available for {nreg} of 171 patients and unambiguous','Table 1 footnote: region availability')
# Table 1 region rows: the two unclassified patients are the "Other" row, not additionally "missing"
d.set_row(r'Incision region \(derived from procedure text\)',[None,None,'0 (0%)'],'Table 1 region missing cell',table_hint='Characteristic'); CH.append(('Table 1 incision-region row','Missing column 2 (1%) -> 0 (0%); the two unclassified patients are counted once, in the Other row'))
# Table 2 photographs per patient (pandas default quantiles, same as the text)
d.set_row(r'Photographs per patient, median \(IQR\)',[f'{qa[0]} ({qa[1]}{EN}{qa[2]})',f'{qs[0]} ({qs[1]}{EN}{qs[2]})',f'{qn[0]} ({qn[1]}{EN}{qn[2]})'],'Table 2 photographs per patient',table_hint='Photographs per patient'); CH.append(('Table 2 photographs per patient',f'all {qa[0]} ({qa[1]}-{qa[2]}), SSI {qs[0]} ({qs[1]}-{qs[2]}), no SSI {qn[0]} ({qn[1]}-{qn[2]}) (pandas default quantiles)'))
RR(r'median of 6 photographs overall \(interquartile range, 4–9;',f'median of {qa[0]} photographs overall (interquartile range, {qa[1]}{EN}{qa[2]};','text photographs per patient IQR')
# ================= B. truncation =================
RR(r'changed patient-level AUROC by \[PENDING\] relative to the all-image score\. \[PENDING: one sentence interpreting the result\]',
   f'changed patient-level AUROC by {s3(d0)} (95% CI, {cito(tr_lo,tr_hi)}; P={ptr:.2f}) relative to the all-image score, from {f3(safe_auroc(yv,A.p))} to {f3(safe_auroc(yv,Bq.p))}. {"Two" if nlost==2 else nlost} SSI-positive patients whose photographs were all obtained on or after diagnosis contribute no prediagnosis score, so the comparison covers {len(common)} patients and {int(yv.sum())} events; photographs obtained on or after diagnosis raised the all-image estimate only within its bootstrap uncertainty.','p11 truncation')
# ================= C. learning curve (S2) =================
shape=f'The mean changed by {s3(m6-m3)} between one-third and two-thirds of the training patients and by {s3(mf-m6)} from two-thirds to the full set, and the three-seed spread at two-thirds ({f3(lc.loc["two-thirds","mean"]-0)}; seeds {", ".join(f3(x) for x in json.loads(lc.loc["two-thirds","per_seed"]))}) was wider than either difference.'
shape=f'The mean changed by {s3(m6-m3)} between one-third and two-thirds of the training patients and by {s3(mf-m6)} from two-thirds to the full set, and the three-seed spread at two-thirds ({", ".join(f3(x) for x in json.loads(lc.loc["two-thirds","per_seed"]))}) was wider than either difference.'
RR(r'EfficientNet-B0 mean AUROC was \[PENDING\] with one-third of training patients, \[PENDING\] with two-thirds and \[PENDING\] with the full training set \(Supplementary Table S2\)\. \[PENDING: one sentence on the shape of the learning curve\]',
   f'EfficientNet-B0 mean AUROC was {f3(m3)}±{f3(s3v)} with one-third of training patients, {f3(m6)}±{f3(s6)} with two-thirds and {f3(mf)}±{f3(sf)} with the full training set (Supplementary Table S2). {shape}','p13 learning curve')
RR(r'\[PENDING: one sentence on the learning curve\]',f'Mean performance did not rise between one-third and two-thirds of the training patients ({s3(m6-m3)}) and rose by {s3(mf-m6)} with the full set, differences that the three-seed spread contains','p36 learning curve (Discussion)')
interp={'One-third':f'Lowest spread across seeds (SD {f3(s3v)})','Two-thirds':f'Lowest mean; widest spread (SD {f3(s6)})','Full':f'Highest mean (SD {f3(sf)})'}
if m3>=m6 and m3>=mf: interp['One-third']='Highest mean; narrowest spread'
d.set_row(r'One-third',[f'{f3(m3)}±{f3(s3v)}',interp['One-third']],'S2 one-third',table_hint='Training fraction'); d.set_row(r'Two-thirds',[f'{f3(m6)}±{f3(s6)}',interp['Two-thirds']],'S2 two-thirds',table_hint='Training fraction'); d.set_row(r'Full',[f'{f3(mf)}±{f3(sf)}',interp['Full']],'S2 full',table_hint='Training fraction')
CH.append(('Supplementary Table S2',f'one-third {f3(m3)}±{f3(s3v)}, two-thirds {f3(m6)}±{f3(s6)}, full {f3(mf)}±{f3(sf)}'))
# ================= D. context probes (S3) =================
RR(r'Context-only AUROCs were \[PENDING\] for POD, \[PENDING\] for acquisition features, \[PENDING\] for prior imaging behavior and \[PENDING\] for their combination, against a three-seed mean of 0\.749 for the full-image model \(Supplementary Table S3; Figure 3B\)\. \[PENDING: one sentence comparing the combined probe with the full-image model\]',
   f'Context-only AUROCs were {f3(P_("pod_only").patient_auroc)} (95% CI, {ci(P_("pod_only").lo,P_("pod_only").hi)}) for POD, {f3(P_("acquisition").patient_auroc)} ({ci(P_("acquisition").lo,P_("acquisition").hi)}) for acquisition features, {f3(P_("behavior_past_only").patient_auroc)} ({ci(P_("behavior_past_only").lo,P_("behavior_past_only").hi)}) for prior imaging behavior and {f3(P_("acquisition+behavior+pod").patient_auroc)} ({ci(P_("acquisition+behavior+pod").lo,P_("acquisition+behavior+pod").hi)}) for their combination, against a three-seed mean of {f3(full)} for the full-image model (Supplementary Table S3; Figure 3B). The combined probe fell {f3(dd("acquisition+behavior+pod"))} below the full-image mean, a gap smaller than the spread across the three full-image seeds ({f3(ma.loc["Full image"][["seed20260807","seed42","seed7"]].min())} to {f3(ma.loc["Full image"][["seed20260807","seed42","seed7"]].max())}); the seed-specific differences were {s3(sd_["mean"])}±{f3(sd_["sd"])} (range, {s3(sd_["lo"])} to {s3(sd_["hi"])}), and the paired comparison of the locked seed with the probe was {s3(pp.d)} (95% CI, {cito(pp.lo,pp.hi)}; P={float(pp.p):.2f}).','p15 context probes')
for lab,key in (('POD only','pod_only'),('Acquisition features','acquisition'),('Prior imaging behavior','behavior_past_only'),(r'Acquisition \+ behavior \+ POD','acquisition+behavior+pod')):
    d.set_row(lab,[f'{f3(P_(key).patient_auroc)} ({ci(P_(key).lo,P_(key).hi)})',s3(dd(key))],f'S3 {key}',table_hint='Full-image minus probe')
CH.append(('Supplementary Table S3','four probe rows filled (AUROC with 95% CI; full-image minus probe)'))
RR(r'whereas in the present cohort \[PENDING: result of the context-probe comparison\]',f'whereas in the present cohort the full-image model exceeded the combined context probe by only {s3(sd_["mean"])} on average across seeds (range, {s3(sd_["lo"])} to {s3(sd_["hi"])}), a margin within the seed spread','p29 SurgWound contrast sentence')
RR(r'\[PENDING: two or three sentences on the context-probe result, to be written after the re-run\]',
   f'Measured collection context approached full-image discrimination. A combined probe using POD, acquisition features and prior imaging behavior reached {f3(P_("acquisition+behavior+pod").patient_auroc)} on the matched masking frame against a three-seed full-image mean of {f3(full)}, with seed-specific differences spanning both directions ({s3(sd_["lo"])} to {s3(sd_["hi"])}); on the full 171-patient frame the locked model exceeded the probe by {s3(im.cnn_minus_probe)} at image level (95% CI, {cito(im.d_lo,im.d_hi)}) and {s3(pa.cnn_minus_probe)} at patient level ({cito(pa.d_lo,pa.d_hi)}). This does not prove that the CNN encoded those variables, but it shows that the dataset contained an alternative predictive pathway, not requiring wound pixels, of comparable magnitude.','p34 Discussion context paragraph')
RR(r'\[PENDING: context-probe result\]',f'a probe built from those fields alone came within {f3(dd("acquisition+behavior+pod"))} of the full-image model on the matched frame','p38 context-probe result')
RR(r'\[PENDING: context-probe sentence\]',f'and a probe built only from measured collection context came within {f3(dd("acquisition+behavior+pod"))} of the full-image model','p42 context-probe sentence')
RR(r'\[PENDING: one sentence on the context-probe result\]',f'The full-image model exceeded the combined context probe by {s3(sd_["mean"])} on average across seeds, within the seed spread.','Figure 3 caption')
# ================= E. mask geometry (S11) =================
RR(r'reached \[PENDING\] with frozen features and \[PENDING\] across three fine-tuned seeds\.',f'reached {f3(fz.auroc)} (95% CI, {ci(fz.lo,fz.hi)}) with frozen features and {f3(ft_mean)} (range, {f3(ft_lo)} to {f3(ft_hi)}) across three fine-tuned seeds.','p16 mask-as-image')
RR(r'reached \[PENDING\], compared with 0\.749 for the full-image model on the same frame \(Supplementary Table S11\)\. \[PENDING: whether the descriptors also separate chest from abdominal incisions\]',
   f'reached {f3(g7.auroc)} ({ci(g7.lo,g7.hi)}), compared with {f3(full)} for the full-image model on the same frame, and mask area alone was below chance ({f3(ao.auroc)}, {ci(ao.lo,ao.hi)}; Supplementary Table S11). The descriptors still separated chest from abdominal incisions (patient-level AUROC {aur(gr.loc["elongation"]):.2f} for elongation and {aur(gr.loc["fill_ratio"]):.2f} for fill ratio), so annotation geometry carries case-mix information even though it no longer out-scores the image model.','p16 geometry descriptors')
d.set_row(r'Binary mask as an image, frozen features',[f'{f3(fz.auroc)} ({ci(fz.lo,fz.hi)})'],'S11 frozen',table_hint='geometry'); d.set_row(r'Binary mask as an image, fine-tuned, 3 seeds',[f'{f3(ft_mean)} (range {ci(ft_lo,ft_hi)})'],'S11 fine-tuned',table_hint='geometry')
d.set_row(r'Mask area alone',[f'{f3(ao.auroc)} ({ci(ao.lo,ao.hi)})'],'S11 area',table_hint='geometry'); d.set_row(r'Seven scalar geometry descriptors',[f'{f3(g7.auroc)} ({ci(g7.lo,g7.hi)})'],'S11 seven',table_hint='geometry'); d.set_row(r'Difference, geometry minus full image',[s3(fg.loc['geometry_minus_full_image'].auroc)],'S11 diff',table_hint='geometry')
CH.append(('Supplementary Table S11','five cells filled'))
# ================= F. foundation probes (S12) =================
cond={'Full image':'full_image','True wound-only':'wound_only','True background-only':'background_only','Circularly shifted region':'sham_wound_only','Complement of shifted region':'sham_background_only'}
def ranks(dct): return {k:i+1 for i,(k,v) in enumerate(sorted(dct.items(),key=lambda kv:-kv[1]))}
ordw=['first','second','third','fourth','fifth']; ftm={'full_image':ma.loc['Full image'].seed_mean,'wound_only':ma.loc['True wound-only'].seed_mean,'background_only':ma.loc['True background-only'].seed_mean,'sham_wound_only':ma.loc['Circularly shifted wound-size region'].seed_mean,'sham_background_only':ma.loc['Complement of shifted region'].seed_mean}
rf=ranks(ftm); rd=ranks({c:float(V('dinov2_vitb14',c).auroc) for c in cond.values()}); dw=V('dinov2_vitb14','wound_only'); bvals=[float(V('biomedclip_vitb16',c).auroc) for c in cond.values()]
C=lambda k:f"{s3(fc.loc[k].d)} ({cito(fc.loc[k].lo,fc.loc[k].hi)}; Holm-adjusted P={float(fc.loc[k].holm_p):.2f})"; sig=[k for k in fc.index if float(fc.loc[k].holm_p)<0.05]
dv=lambda c:V('dinov2_vitb14',c)
para=(f"DINOv2 reached {f3(dv('full_image').auroc)} ({ci(dv('full_image').lo,dv('full_image').hi)}) for the full image, {f3(dv('wound_only').auroc)} ({ci(dv('wound_only').lo,dv('wound_only').hi)}) for the true wound region, {f3(dv('background_only').auroc)} ({ci(dv('background_only').lo,dv('background_only').hi)}) for the true background, {f3(dv('sham_wound_only').auroc)} ({ci(dv('sham_wound_only').lo,dv('sham_wound_only').hi)}) for the shifted region and {f3(dv('sham_background_only').auroc)} ({ci(dv('sham_background_only').lo,dv('sham_background_only').hi)}) for its complement; BiomedCLIP values ranged from {f3(min(bvals))} to {f3(max(bvals))}. "
      f"DINOv2 placed the true wound region {ordw[rd['wound_only']-1]} of the five conditions, {'as the fine-tuned model did' if rd['wound_only']==rf['wound_only'] else 'unlike the fine-tuned model'}, but the rank agreement with the fine-tuned ordering was weak (ρ={ra.loc['dinov2_vitb14'].spearman_vs_finetuned:+.2f} for DINOv2, {ra.loc['biomedclip_vitb16'].spearman_vs_finetuned:+.2f} for BiomedCLIP). "
      f"On paired patient-clustered resampling of the DINOv2 probe, true background minus true wound was {C('background_only - wound_only')}, the complement of the shifted region minus true wound was {C('sham_background_only - wound_only')}, and the area-matched contrast, shifted region minus true wound, was {C('sham_wound_only - wound_only')}; "
      +("no contrast survived Holm correction" if not sig else f"{len(sig)} contrast(s) survived Holm correction")+" (Supplementary Table S12).")
RR(r'\[PENDING: AUROCs for the five conditions, agreement with the fine-tuned ranking, and the paired contrasts with Holm adjustment \(Supplementary Table S12\)\]',para,'p17 S12 paragraph')
for lab,c in cond.items():
    a=V('dinov2_vitb14',c); bb=V('biomedclip_vitb16',c); d.set_row(re.escape(lab),[f'{f3(a.auroc)} ({ci(a.lo,a.hi)})',f'{f3(bb.auroc)} ({ci(bb.lo,bb.hi)})'],f'S12 {lab}',table_hint='DINOv2')
d.set_row(r'True background − true wound',[C('background_only - wound_only').replace('; Holm-adjusted P=',', Holm P=')],'S12 contrast 1',table_hint='DINOv2'); d.set_row(r'Complement of shifted region − true wound',[C('sham_background_only - wound_only').replace('; Holm-adjusted P=',', Holm P=')],'S12 contrast 2',table_hint='DINOv2')
CH.append(('Supplementary Table S12','ten condition cells and two contrast cells filled'))
RR(r'\[PENDING: which paired contrasts are listed and their Holm-adjusted P values\]',f'The two listed contrasts are true background minus true wound and the complement of the shifted region minus true wound; after Holm adjustment across the five contrasts of the family, {"no contrast remained significant" if not sig else str(len(sig))+" remained significant"} (all adjusted P={float(fc.holm_p.min()):.2f} or higher).','S12 footnote')
RR(r'\[PENDING: one sentence on the frozen foundation-model probes and one on the mask-geometry control\]',
   f"Frozen foundation-model probes placed the true wound region {ordw[rd['wound_only']-1]} of the five conditions, as the fine-tuned model did, but no frozen contrast survived Holm correction and their ordering agreed only weakly with the fine-tuned one (ρ={ra.loc['dinov2_vitb14'].spearman_vs_finetuned:+.2f} for DINOv2). Seven scalar descriptors of the annotation mask alone reached {f3(g7.auroc)}, within {abs(float(fg.loc['geometry_minus_full_image'].auroc)):.2f} of the full-image model, and mask area alone was below chance ({f3(ao.auroc)}).",'p32 Discussion S12/S11 sentences')
# ================= G. overlap, calibration, curation, temperatures =================
pct=f'{100*float(ov.frac_le_0_05):.1f}%'
RR(r'In the masked retraining, \[PENDING\] of shifted masks overlapped',f'In the masked retraining, {pct} of shifted masks overlapped','p33 overlap')
RR(r'\[PENDING: median, mean and maximum IoU and the share of masks with IoU <0\.05\]',f'Median IoU with the true mask was {f3(ov.iou_median)} (mean, {float(ov.iou_mean):.4f}); {pct} of masks had IoU ≤0.05 and the maximum was {f3(ov.iou_max)}.','S4 footnote overlap')
RR(r'minimal overlap with the true wound mask in \[PENDING\] of images',f'minimal overlap with the true wound mask in {pct} of images','Figure 4 caption overlap')
RR(r'calibration slope of 0\.38, and calibration intercept of −1\.29 \[Reference\]',f'calibration slope of {f2h(cc.slope)} (95% CI, {f2h(cc.slope_lo)} to {f2h(cc.slope_hi)}), and calibration intercept of {f2h(cc.intercept)} (95% CI, {f2h(cc.intercept_lo)} to {f2h(cc.intercept_hi)}) [39]','p23 calibration CIs + citation')
d.set_row(r'Calibration slope',[f3(cc.slope),cito(cc.slope_lo,cc.slope_hi)],'S6 slope',table_hint='Brier'); d.set_row(r'Calibration intercept',[f3(cc.intercept),cito(cc.intercept_lo,cc.intercept_hi)],'S6 intercept',table_hint='Brier'); CH.append(('Supplementary Table S6','slope and intercept with 95% CIs'))
RR(r'uncurated postoperative smartphone photographs yielded an AUROC of \[PENDING\]\.',f'uncurated postoperative smartphone photographs yielded an AUROC of {f3(cu.auroc)} (95% CI, {ci(cu.lo,cu.hi)}; {int(cu.n_photos):,} postoperative photographs from {int(cu.n_pts)} patients with {int(cu.n_events)} events, scored with the expanded-frame model’s saved out-of-fold predictions because the image files of that frame are no longer available for retraining).','p911 curation')
RR(r'\[PENDING: postoperative-frame values, if this note is kept\]',f'Among the 171 patients of the postoperative frame, {nimpl} implausible temperature values were set to missing before the comparison.','p907 temperatures')
# ================= H. references: 12 placeholders + mapping fixes + 7 new entries =================
NEWREF=[('Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. Epidemiology. 2010;21:128–138. doi:10.1097/EDE.0b013e3181c30fb2'),
        ('Dafni U. Landmark analysis at the 25-year landmark point. Circ Cardiovasc Qual Outcomes. 2011;4:363–371. doi:10.1161/CIRCOUTCOMES.110.957951'),
        ('Phipson B, Smyth GK. Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. Stat Appl Genet Mol Biol. 2010;9:Article39. doi:10.2202/1544-6115.1585'),
        ('Wolpert DH. Stacked generalization. Neural Netw. 1992;5:241–259. doi:10.1016/S0893-6080(05)80023-1'),
        ('Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS One. 2015;10:e0118432. doi:10.1371/journal.pone.0118432'),
        ('Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26:565–574. doi:10.1177/0272989X06295361'),
        ('Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall/CRC; 1993.')]
# numbering 39..45 in order of first appearance: p23 (39), p58 (40), p68 (41), p76 (42), p78 (43), p78 (44), p82 (45)
RR(r'capture less clinical context than the electronic health record \[Reference\]','capture less clinical context than the electronic health record [7]','ref: EHR context -> Bucher [7]')
RR(r'placed away from the wound\) \[Reference\]','placed away from the wound) [16]','ref: sham masks -> Hooker [16]')
RR(r'DINOv2 and BiomedCLIP \[Reference\]','DINOv2 and BiomedCLIP [30,31]','ref: DINOv2/BiomedCLIP [30,31]')
RR(r'On the SurgWound \[Reference\] collection','On the SurgWound [24] collection','ref: SurgWound [24]')
RR(r'rather than mere nuisance variation \[Reference\]','rather than mere nuisance variation [8,20]','ref: submission behaviour -> McLean multimodal [8], Bluebelle [20]')
RR(r'among patients still at risk at that landmark \[Reference\]','among patients still at risk at that landmark [40]','ref: landmark -> Dafni [40]')
RR(r'\(1 \+ the number of relocations scoring at least as high\) / \(draws \+ 1\) \[Reference\]','(1 + the number of relocations scoring at least as high) / (draws + 1) [41]','ref: add-one P -> Phipson & Smyth [41]')
RR(r'post hoc stacked model \[Reference\]','post hoc stacked model [42]','ref: stacking -> Wolpert [42]')
RR(r'AUPRC was emphasized because SSI prevalence was low \[Reference\]','AUPRC was emphasized because SSI prevalence was low [43]','ref: AUPRC -> Saito & Rehmsmeier [43]')
RR(r'Decision-curve analyses were treated as secondary \[Reference\]','Decision-curve analyses were treated as secondary [44]','ref: DCA -> Vickers & Elkin [44]')
RR(r'20,000 patient-level bootstrap resamples \[Reference\]','20,000 patient-level bootstrap resamples [45]','ref: bootstrap -> Efron & Tibshirani [45]')
# mapping fixes from Mahdi's REFERENCES comment
RR(r'well recognized in medical imaging \[10\]','well recognized in medical imaging [34]','ref fix: shortcut learning -> Geirhos [34]')
RR(r'(chest radiograph[^.\[]{0,160})\[11,12\]',r'\1[10,11]','ref fix: chest radiography -> DeGrave, Zech [10,11]')
RR(r'(fracture detection[^.\[]{0,200})\[13\]',r'\1[12]','ref fix: hip fracture -> Badgeley [12]')
RR(r'signature of shortcut learning reported in other imaging domains \[10,11\]','signature of shortcut learning reported in other imaging domains [34,10]','ref fix: Discussion signature -> [34,10]')
RR(r'apparent discrimination in other imaging models \[13\]','apparent discrimination in other imaging models [12]','ref fix: Limitations -> Badgeley [12]')
RR(r'within a single cohort provide a complementary check \[7\]','within a single cohort provide a complementary check [13]','ref fix: acquisition-bias method -> Ong Ly [13]')
RR(r'\(CDC/NHSN\) criteria \[24\]','(CDC/NHSN) criteria [38]','ref fix: NHSN -> [38]')
RR(r'(SurgWound[^\[]{0,220})\[34\]',r'\1[24]','ref fix: SurgWound [34] -> [24]')
RR(r' \[17\]','','ref fix: Norman [17] removed, site 1'); RR(r'more likely to be infected \[17\]','more likely to be infected','ref fix: Norman [17] removed, site 2')
# append the new entries after the last reference paragraph (entries are non-empty paragraphs after the REFERENCES heading)
body=d.root.find(W+'body'); kids=list(body); ptext=lambda p:''.join(t.text or '' for t in p.iter(W+'t'))
i0=next(i for i,k in enumerate(kids) if k.tag==W+'p' and ptext(k).strip().lower()=='references'); last=None
for i in range(i0+1,len(kids)):
    s=ptext(kids[i]).strip()
    if re.match(r'Table 1\b',s) or s.lower().startswith('supplementary') or s.startswith('Figure'): break
    if s: last=kids[i]
anchor=last
for j,e in enumerate(NEWREF,start=39): anchor=d.insert_paragraph_after(anchor,e)
CH.append(('REFERENCES',f'7 entries appended as [39]-[45] (all DOIs resolved at the publisher on 2026-10-26); 12 placeholders cited; 8 mis-mapped citations corrected; Norman [17] removed from the sentence it did not support'))
d.log.append('appended 7 reference entries 39-45')
# ================= I. figures: replace the four images (media) and fix Figure 1's extent =================
d.sweep_minus(); d.save(OUT)
from PIL import Image
figs={'rId12':REPO+'figures/figure1_flow_postop.png','rId13':D+'figures/figure2_pod7_postop.png','rId14':D+'figures/figure3_architecture_context_postop.png','rId15':D+'figures/figure4_masking_postop.png'}
z=zipfile.ZipFile(OUT); rels=z.read('word/_rels/document.xml.rels').decode(); target={rid:re.search(r'Id="'+rid+r'"[^>]*Target="([^"]+)"',rels).group(1) for rid in figs}
docxml=z.read('word/document.xml').decode()
EMU_W=6629400
sizes={}
for rid,png in figs.items():
    w,h=Image.open(png).size; cx=EMU_W; cy=int(round(EMU_W*h/w))
    if cy>6858000: cy=6858000; cx=int(round(cy*w/h))
    sizes[rid]=(cx,cy)
def fixblk(m):
    blk=m.group(0); rid=re.search(r'r:embed="(rId\d+)"',blk)
    if not rid or rid.group(1) not in sizes: return blk
    cx,cy=sizes[rid.group(1)]; blk=re.sub(r'<wp:extent cx="\d+" cy="\d+"/>',f'<wp:extent cx="{cx}" cy="{cy}"/>',blk); return re.sub(r'<a:ext cx="\d+" cy="\d+"/>',f'<a:ext cx="{cx}" cy="{cy}"/>',blk)
docxml=re.sub(r'<w:drawing>(?:(?!</w:drawing>).)*?</w:drawing>',fixblk,docxml,flags=re.S)
tmp=OUT+'.tmp'; zo=zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED)
for item in z.infolist():
    data=z.read(item.filename)
    if item.filename=='word/document.xml': data=docxml.encode()
    for rid,png in figs.items():
        if item.filename=='word/'+target[rid]: data=open(png,'rb').read()
    zo.writestr(item,data)
zo.close(); z.close(); shutil.move(tmp,OUT); CH.append(('Figures 1-4','images replaced with the postoperative versions (Figure 1 rescaled to its portrait aspect; Figures 2-4 keep the slot size)'))
# ================= J. comment replies =================
REPLY={
 0:f'{n30} within 30 days, 1 later (POD {late[0]}) and 1 with no diagnosis date. The POD 47 patient is not in this frame, none of their photos passed the gate. I put the same numbers in Outcome Ascertainment.',
 1:f'You are right, I had used the lower quantile. With pandas default it is {qa[0]} ({qa[1]}-{qa[2]}) overall, {qs[0]} ({qs[1]}-{qs[2]}) SSI, {qn[0]} ({qn[1]}-{qn[2]}) no SSI. Changed the text and Table 2 to match. Range 1-39 is right.',
 3:'Yes, 0.085-0.742 is the patient-clustered bootstrap on the POD 7 risk set (20,000 resamples, seed 913, same as the AUROC CI). I reran it today to be sure and added it to the repo as postop_primary_auprc_ci.csv.',
 5:f'Added them: slope {f2h(cc.slope)} ({f2h(cc.slope_lo)} to {f2h(cc.slope_hi)}), intercept {f2h(cc.intercept)} ({f2h(cc.intercept_lo)} to {f2h(cc.intercept_hi)}). Same bootstrap, 20,000 resamples. Also in S6 now (postop_calibration_ci.csv in the repo).',
 6:'Yes. It is the mean over all 1,341 postop photos per patient, so photos taken on or after diagnosis are in it.',
 8:'Confirmed. Every photo was linked to the participant and the date, POD from the surgery date, then study IDs assigned and identifiers stripped before anything else. The 9 photos we could not date were dropped.',
 13:'Went through all of them against the sentences. Fixed: shortcut learning [34], chest radiography [10,11], hip fracture [12], the Discussion one [34,10], Limitations [12], acquisition-bias method [13], NHSN [38], SurgWound [24], and DINOv2/BiomedCLIP [30,31] at first mention. Norman 2022 does not support the sentence it was on (it is the Cochrane NPWT review), so I took it off rather than leave a wrong citation, which means [17] is now uncited. The 12 [Reference] spots are filled; I added 7 new entries ([39]-[45]) for calibration, the landmark method, the add-one P value, stacking, AUPRC, decision curves and the bootstrap, and checked each DOI resolves to the paper.',
 14:f'{ntrk}. Filled in.',
 15:f'From the keyword rule: chest 120, abdomen 49, and 2 that the rule could not classify (the two lower-extremity ones), so {nreg} of 171. In Table 1 those 2 are the Other row, so the missing cell for region should be 0, I changed it. Rows now add to 171.',
 18:'Partly. 277 of the 468 rated images are postop photos from the locked frame, the other 191 came from the uncurated export and I cannot tell from the rating file whether any of those are intraoperative. I left the section alone for now. I can rerun it on the 277 or we drop it, whichever you prefer.',
 19:'Done, all four swapped for the postop versions (Figure 1 from figure1_flow_postop.png, 2-4 from make_figures_postop.py).',
 20:'I can add the permutation null as Figure 4C if Dr. Kewalramani wants it, it is a quick one from permutation_null.csv. Did not add it in this pass so the figures still match the captions you reviewed.'}
W14='http://schemas.microsoft.com/office/word/2010/wordml'; W15='http://schemas.microsoft.com/office/word/2012/wordml'; W16CID='http://schemas.microsoft.com/office/word/2016/wordml/cid'; W16CEX='http://schemas.microsoft.com/office/word/2018/wordml/cex'
z=zipfile.ZipFile(OUT); parts={n:z.read(n) for n in z.namelist()}; z.close()
com=ET.fromstring(parts['word/comments.xml']); cex=ET.fromstring(parts['word/commentsExtended.xml']); cid=ET.fromstring(parts['word/commentsIds.xml']); cexx=ET.fromstring(parts['word/commentsExtensible.xml']); ppl=ET.fromstring(parts['word/people.xml']); docx_=ET.fromstring(parts['word/document.xml'])
existing=list(com.iter(W+'comment')); nextid=max(int(c.get(W+'id')) for c in existing)+1
paraid_of={c.get(W+'id'):list(c.iter(W+'p'))[-1].get('{%s}paraId'%W14) for c in existing}
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'); rnd=random.Random(2026)
def hexid(): return '%08X'%rnd.randint(0x10000000,0x7FFFFFFF)
tmpl=existing[0]
for pid_,text in REPLY.items():
    nid=str(nextid); nextid+=1; pid8=hexid(); dur=hexid()
    c=copy.deepcopy(tmpl); c.set(W+'id',nid); c.set(W+'author',AUTHOR); c.set(W+'date',now); c.set(W+'initials','ZJ')
    p=list(c.iter(W+'p'))[0]; p.set('{%s}paraId'%W14,pid8); p.set('{%s}textId'%W14,'77777777')
    runs=[r for r in p.findall(W+'r') if r.find(W+'annotationRef') is None]
    for r in runs[1:]: p.remove(r)
    runs[0].find(W+'t').text=text
    for extra in list(c.iter(W+'p'))[1:]: extra.getparent().remove(extra)
    com.append(c)
    e=ET.SubElement(cex,'{%s}commentEx'%W15); e.set('{%s}paraId'%W15,pid8); e.set('{%s}paraIdParent'%W15,paraid_of[str(pid_)]); e.set('{%s}done'%W15,'0')
    e=ET.SubElement(cid,'{%s}commentId'%W16CID); e.set('{%s}paraId'%W16CID,pid8); e.set('{%s}durableId'%W16CID,dur)
    e=ET.SubElement(cexx,'{%s}commentExtensible'%W16CEX); e.set('{%s}durableId'%W16CEX,dur); e.set('{%s}dateUtc'%W16CEX,now)
    # anchor: same range as the parent
    rs=docx_.find('.//'+W+'commentRangeStart[@'+W+'id="'+str(pid_)+'"]'); re_=docx_.find('.//'+W+'commentRangeEnd[@'+W+'id="'+str(pid_)+'"]'); ref=docx_.find('.//'+W+'commentReference[@'+W+'id="'+str(pid_)+'"]')
    if rs is not None and re_ is not None and ref is not None:
        s2=ET.Element(W+'commentRangeStart'); s2.set(W+'id',nid); rs.addnext(s2)
        e2=ET.Element(W+'commentRangeEnd'); e2.set(W+'id',nid); re_.addnext(e2)
        refrun=ref.getparent(); r2=copy.deepcopy(refrun); r2.find(W+'commentReference').set(W+'id',nid); refrun.addnext(r2)
    else:  # parent without a range (Dr K's comment 8): put the reply reference right after the parent's reference run
        refrun=ref.getparent(); r2=copy.deepcopy(refrun); r2.find(W+'commentReference').set(W+'id',nid); refrun.addnext(r2)
    d.log.append(f'reply added to comment {pid_}')
if not any(p.get('{%s}author'%W15)==AUTHOR for p in ppl.iter('{%s}person'%W15)):
    pe=ET.SubElement(ppl,'{%s}person'%W15); pe.set('{%s}author'%W15,AUTHOR); pi=ET.SubElement(pe,'{%s}presenceInfo'%W15); pi.set('{%s}providerId'%W15,'None'); pi.set('{%s}userId'%W15,AUTHOR)
for n,el in (('word/comments.xml',com),('word/commentsExtended.xml',cex),('word/commentsIds.xml',cid),('word/commentsExtensible.xml',cexx),('word/people.xml',ppl),('word/document.xml',docx_)):
    parts[n]=ET.tostring(el,xml_declaration=True,encoding='UTF-8',standalone=True)
zo=zipfile.ZipFile(OUT+'.tmp','w',zipfile.ZIP_DEFLATED)
for n,data in parts.items(): zo.writestr(n,data)
zo.close(); shutil.move(OUT+'.tmp',OUT)
CH.append(('Comments',f'replies added to comments {sorted(REPLY)}'))
json.dump(CH,open('new/change_log.json','w'),indent=1,ensure_ascii=False)
print('\n'.join(d.log)); print('WROTE',OUT)
