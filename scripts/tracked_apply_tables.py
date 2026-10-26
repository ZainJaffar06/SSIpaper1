# -*- coding: utf-8 -*-
import sys, os, re, os
sys.path.insert(0,'.'); sys.path.insert(0,'/Users/zainjaffar/ssi_work/paper1_2026-08-15')
import pandas as pd, numpy as np, xml.etree.ElementTree as ET
from dxml_tracked import TrackedDoc as Doc, W
from lib import safe_auroc, safe_auprc, cluster_boot_ci
P1='/Users/zainjaffar/ssi_work/paper1_2026-08-15/'; D=P1+'out/postop_only/derived/'
F=os.environ['OUTDOC']
doc=Doc(F); LOG=[]
parents={c:p for p in doc.root.iter() for c in p}
def indel(el):
    c=parents.get(el)
    while c is not None:
        if c.tag==W+'del': return True
        c=parents.get(c)
    return False
tbls=list(doc.root.iter(W+'tbl'))
def cells(tb):
    out=[]
    for tr in tb.iter(W+'tr'):
        row=[]
        for tc in tr.iter(W+'tc'):
            ts=[x for x in tc.iter(W+'t') if not indel(x)]
            row.append((tc,ts,''.join((x.text or '') for x in ts).strip()))
        out.append((tr,row))
    return out
def setcell(tc_ts,new):
    doc.set_cell_tracked(tc_ts[0],new)
def rep_cell(ti,old,new,lab):
    n=0
    for tr,row in cells(tbls[ti]):
        for c in row:
            if c[2]==old: setcell(c,new); n+=1
    LOG.append(('OK  ' if n else 'MISS')+f' [{n}] T{ti} {lab}: {old!r} -> {new!r}')
def set_row(ti,label,vals,lab):
    for tr,row in cells(tbls[ti]):
        if row and row[0][2]==label:
            for c,v in zip(row[1:],vals): setcell(c,v)
            LOG.append(f'OK   T{ti} row {label!r}'); return
    LOG.append(f'MISS T{ti} row {label!r}')
def drop_row(ti,label):
    for tr,row in cells(tbls[ti]):
        if row and row[0][2]==label:
            doc.delete_row_tracked(tr); LOG.append(f'OK   T{ti} dropped row {label!r}'); return
    LOG.append(f'MISS T{ti} drop row {label!r}')
f3=lambda x:f'{x:.3f}'; s3=lambda x:f'{x:+.3f}'; ci=lambda lo,hi:f'{f3(lo)}–{f3(hi)}'
FINAL=set(open(P1+'final_ssi_list.txt').read().split())

# ============ data ============
pl=pd.read_csv(D+'primary_and_landmarks.csv'); p7,lt,cl,p14,p30=[pl.iloc[i] for i in range(5)]
a=pd.read_csv(D+'architectures.csv').set_index('architecture')
ma=pd.read_csv(D+'masking_absolute.csv').set_index('condition')
mc=pd.read_csv(D+'masking_contrasts.csv').set_index('contrast')
pm=dict(zip(*pd.read_csv(D+'permutation_summary.csv').values.T))
cm=pd.read_csv(D+'clinical_models.csv').set_index('model'); cc=pd.read_csv(D+'clinical_contrasts.csv').set_index('contrast')
ca=pd.read_csv(D+'calibration.csv').set_index('metric').value
op=pd.read_csv(D+'operating_points.csv').set_index('target_spec')
pf=pd.read_csv(P1+'out/recompute/per_photo_frame_1948.csv'); po=pf[pf.photo_type=='postoperative']
S=pd.read_csv(D+'pod7_patient_scores.csv')
t1=pd.read_csv(P1+'out/recompute/baseline_table1_postop.csv',dtype=str).fillna('')

# ============ T0: Table 1 ============
hdr=cells(tbls[0])[0][1]
setcell(hdr[1],'SSI (n=13)'); setcell(hdr[2],'No SSI (n=158)'); LOG.append('OK   T0 header')
src={r.iloc[0].strip():(r.iloc[1],r.iloc[2],r.iloc[3]) for _,r in t1.iterrows() if r.iloc[0].strip()}
nrow=0
for tr,row in cells(tbls[0])[1:]:
    lab=row[0][2]
    if lab in src:
        for c,v in zip(row[1:4],src[lab]): setcell(c,v if v!='' else ('—' if c[2]=='—' else ''))
        nrow+=1
LOG.append(f'OK   T0 {nrow} rows refreshed from baseline_table1_postop.csv')
drop_row(0,'Remote from photographed incision')

# ============ T1: Table 2 cohort frames ============
pp=po.groupby('pid').size(); ssi=po.drop_duplicates('pid').set_index('pid').ssi_final
iq=lambda s:f'{int(s.median())} ({int(s.quantile(.25))}–{int(s.quantile(.75))})'
by7=po[po.day_after_surgery<=7].pid.unique(); n7=len(by7); n7p=int(ssi.loc[by7].sum())
set_row(1,'Locked quality-gated cohort, patients',['171','13','158'],'')
set_row(1,'Photographs',['1,341',f'{int(po[po.ssi_final==1].shape[0]):,}',f'{int(po[po.ssi_final==0].shape[0]):,}'],'')
set_row(1,'Photographs per patient, median (IQR)',[iq(pp),iq(pp[ssi[pp.index]==1]),iq(pp[ssi[pp.index]==0])],'')
set_row(1,'Imaged by POD 7, n (%)',[f'{n7} ({100*n7/171:.1f})',f'{n7p} ({100*n7p/13:.0f})',f'{n7-n7p} ({100*(n7-n7p)/158:.1f})'],'')
set_row(1,'POD 7 risk set, patients',['132','8 subsequent SSI','124'],'')
set_row(1,'Diagnosis-timing sensitivity, patients',['170','12 dated SSI','158'],'')
set_row(1,'Mask-eligible cohort, patients / images',['168 / 1,296','12 / not separately tabulated','156 / not separately tabulated'],'')
inc1=src.get('Incision length, cm',('','',''))
set_row(1,'Incision length, cm, mean±SD (n recorded)',['As Table 1',inc1[0],inc1[1]],'')
rep_cell(1,'Locked quality-gated cohort, patients','Postoperative analytic frame, patients','relabel')

# ============ T2: Table 3 POD7 ============
_,aplo,aphi=cluster_boot_ci(S.pid.values,S.ssi_final.values,S.score.values,safe_auprc,20000,913)
set_row(2,'Mean of all photographs through POD 7',['132 / 8',f'{f3(p7.auroc)} ({ci(p7.lo,p7.hi)})',f3(p7.auprc),'—','—','—'],'')
set_row(2,'Latest photograph through POD 7',['132 / 8',f'{f3(lt.auroc)} ({ci(lt.lo,lt.hi)})',f3(lt.auprc),'—','—','—'],'')
set_row(2,'Photograph closest to POD 7',['132 / 8',f'{f3(cl.auroc)} ({ci(cl.lo,cl.hi)})',f3(cl.auprc),'—','—','—'],'')
o8,o9,o95=op.loc[0.8],op.loc[0.9],op.loc[0.95]
set_row(2,'Threshold selected for 80% specificity',['132 / 8','—','—',f3(o8.sensitivity),f3(o8.specificity),f3(o8.n_flagged/132)],'')
# rebuild the two capture rows for 8 events
y=S.ssi_final.values; s=S.score.values; pos=np.sort(s[y==1])
def cap(k):
    thr=pos[len(pos)-k]; fl=(s>=thr); sens=(s[y==1]>=thr).mean(); spec=(s[y==0]<thr).mean(); return sens,spec,fl.mean()
c7=cap(7); c8=cap(8)
for tr,row in cells(tbls[2]):
    if row[0][2]=='Threshold capturing 13 of 14 events':
        setcell(row[0],'Threshold capturing 7 of 8 events'); 
        for c,v in zip(row[1:],['132 / 8','—','—',f3(c7[0]),f3(c7[1]),f3(c7[2])]): setcell(c,v)
    if row[0][2]=='Threshold capturing all 14 events':
        setcell(row[0],'Threshold capturing all 8 events')
        for c,v in zip(row[1:],['132 / 8','—','—',f3(c8[0]),f3(c8[1]),f3(c8[2])]): setcell(c,v)
LOG.append('OK   T2 capture rows rebuilt for 8 events')

# ============ T3: Table 4 clinical (full cohort rows now; POD7 rows patched later) ============
c8v,imv,stv=cm.loc['clinical 8-variable'],cm.loc['image only'],cm.loc['stacked, 8-variable + image']
rows3=cells(tbls[3])
for tr,row in rows3:
    if row[0][2]=='Full range-validated cohort' and row[1][2]=='Clinical':
        for c,v in zip(row[2:],['171 / 13',f'{f3(c8v.auroc)} ({ci(c8v.lo,c8v.hi)})',f3(c8v.auprc)]): setcell(c,v)
    if row[0][2]=='Full range-validated cohort' and row[1][2]=='Image':
        for c,v in zip(row[2:],['171 / 13',f'{f3(imv.auroc)} ({ci(imv.lo,imv.hi)})',f3(imv.auprc)]): setcell(c,v)
    if row[0][2]=='Full range-validated cohort' and row[1][2]=='Stacked':
        for c,v in zip(row[2:],['171 / 13',f'{f3(stv.auroc)} ({ci(stv.lo,stv.hi)})',f3(stv.auprc)]): setcell(c,v)
    if row[0][2]=='POD 7 risk set' and row[1][2]=='Image':
        for c,v in zip(row[2:],['132 / 8',f'{f3(p7.auroc)} ({ci(p7.lo,p7.hi)})',f3(p7.auprc)]): setcell(c,v)
LOG.append('OK   T3 full-cohort rows + POD7 image row (clinical/stacked POD7 patched after background job)')
rep_cell(3,'Full range-validated cohort','Full postoperative cohort','relabel')

# ============ T4: S1 architectures ============
for key,lab,interp in [('resnet18','ResNet18','Rank changed by seed'),('efficientnet_b0','EfficientNet-B0','Lowest mean; narrowest seed spread'),('vit_b_16','ViT-B/16','Highest mean; was lowest with intraoperative photographs')]:
    r=a.loc[key]; set_row(4,lab,[f'{f3(r["mean"])}±{f3(r.sd)}',ci(r.lo,r.hi),interp],'')

# ============ T7: S4 -> seed 20260807 single run on the postop frame, with per-condition CIs ============
# recompute per-condition patient scores for CIs
def pat(c,seed):
    d=pd.read_csv(P1+f'out/postop_only/mask_{c}_seed{seed}.csv').dropna(subset=['proba'])
    d['y']=d.pid.isin(FINAL).astype(int); return d.groupby('pid').agg(y=('y','max'),p=('proba','mean'))
CONDS=[('full_image','Full image'),('wound_only','True wound-only'),('background_only','True background-only'),('sham_wound_only','Circularly shifted wound-size region'),('sham_background_only','Complement of shifted region')]
PS={(c,s):pat(c,s) for c,_ in CONDS for s in (20260807,42,7)}
pids=sorted(set.intersection(*[set(v.index) for v in PS.values()]))
yv=PS[('full_image',20260807)].loc[pids,'y'].values
for c,lab in CONDS:
    g=PS[(c,20260807)].loc[pids]; _,lo,hi=cluster_boot_ci(np.array(pids),g.y.values,g.p.values,safe_auroc,20000,913)
    set_row(7,lab,[f3(safe_auroc(g.y,g.p)),ci(lo,hi),'—'],'')
# S4 contrasts for seed 20260807
def paired(a_,b_):
    rng=np.random.default_rng(913); uq=np.array(pids); reps=[]
    A=PS[(a_,20260807)].loc[pids,'p'].values; B=PS[(b_,20260807)].loc[pids,'p'].values
    for _ in range(20000):
        ii=rng.integers(0,len(uq),len(uq))
        if len(np.unique(yv[ii]))<2: continue
        reps.append(safe_auroc(yv[ii],A[ii])-safe_auroc(yv[ii],B[ii]))
    reps=np.array(reps); d=safe_auroc(yv,A)-safe_auroc(yv,B); p=2*min((reps<=0).mean(),(reps>=0).mean())
    return d,np.percentile(reps,2.5),np.percentile(reps,97.5),p
for lab,(x,yy) in {'Shifted region − true wound':('sham_wound_only','wound_only'),'Shifted region − true background':('sham_wound_only','background_only'),
                   'True background − full':('background_only','full_image'),'True wound − full':('wound_only','full_image'),
                   'Complement of shifted region − full':('sham_background_only','full_image')}.items():
    d,lo,hi,p=paired(x,yy); set_row(7,lab,[s3(d),f'{s3(lo)} to {s3(hi)}',f'{p:.2f}' if p>=0.01 else f'{p:.3f}'],'')

# ============ T8: S5 landmarks ============
set_row(8,'POD 7 / mean',['132 / 8',f'{f3(p7.auroc)} ({ci(p7.lo,p7.hi)})',f3(p7.auprc),'Primary'],'')
set_row(8,'POD 14 / mean',['157 / 5',f'{f3(p14.auroc)} ({ci(p14.lo,p14.hi)})',f3(p14.auprc),'Secondary; sparse events'],'')
set_row(8,'POD 30 / mean',['159 / 1','not estimable (single event)','—','Not reported'],'')

# ============ T9: S6 calibration ============
set_row(9,'Brier score after cross-fold Platt scaling',[f3(ca['brier']),f'no-information {f3(ca["no_information_brier"])}; skill {ca["brier_skill"]:+.3f}'],'')
set_row(9,'Calibration slope',[f3(ca['calibration_slope']),'—'],'')
set_row(9,'Calibration intercept',[f3(ca['calibration_intercept']),'—'],'')
set_row(9,'Approximately 90% specificity',[f'Sensitivity {f3(o9.sensitivity)}; flagged {f3(o9.n_flagged/132)}',f'Specificity {f3(o9.specificity)}'],'')
for tr,row in cells(tbls[9]):
    if row[0][2]=='Capture 13/14 events':
        setcell(row[0],'Capture 7/8 events'); setcell(row[1],f'Sensitivity {f3(c7[0])}; flagged {f3(c7[2])}'); setcell(row[2],f'Specificity {f3(c7[1])}')
    if row[0][2]=='Capture 14/14 events':
        setcell(row[0],'Capture 8/8 events'); setcell(row[1],f'Sensitivity {f3(c8[0])}; flagged {f3(c8[2])}'); setcell(row[2],f'Specificity {f3(c8[1])}')
LOG.append('OK   T9 capture rows rebuilt')

# ============ T12: S9 three-seed ============
hdr12=cells(tbls[12])[0][1]; setcell(hdr12[1],'Seed 20260807')
for c,lab in CONDS:
    r=ma.loc[lab]; mean_scores=np.column_stack([PS[(c,s)].loc[pids,'p'].values for s in (20260807,42,7)])
    rng=np.random.default_rng(913); reps=[]
    for _ in range(20000):
        ii=rng.integers(0,len(pids),len(pids))
        if len(np.unique(yv[ii]))<2: continue
        reps.append(np.mean([safe_auroc(yv[ii],mean_scores[ii,k]) for k in range(3)]))
    set_row(12,lab,[f3(r.seed20260807),f3(r.seed42),f3(r.seed7),f'{f3(r.seed_mean)} ({ci(np.percentile(reps,2.5),np.percentile(reps,97.5))})','—'],'')
CMAP={'Shifted region − true wound':'shifted - true wound','True background − true wound':'true background - true wound','Shifted region − true background':'shifted - true background',
      'True background − full':'true background - full','True wound − full':'true wound - full','Complement of shifted region − full':'shifted complement - full'}
PAIR={'shifted - true wound':('sham_wound_only','wound_only'),'true background - true wound':('background_only','wound_only'),'shifted - true background':('sham_wound_only','background_only'),
      'true background - full':('background_only','full_image'),'true wound - full':('wound_only','full_image'),'shifted complement - full':('sham_background_only','full_image')}
for lab,key in CMAP.items():
    r=mc.loc[key]; x,yy=PAIR[key]
    per=[f3(safe_auroc(yv,PS[(x,s)].loc[pids,'p'].values)-safe_auroc(yv,PS[(yy,s)].loc[pids,'p'].values)) for s in (20260807,42,7)]
    per=[('+' if not v.startswith('-') else '')+v for v in per]
    set_row(12,lab,[per[0],per[1],per[2],f'{s3(r.d)} ({s3(r.lo)} to {s3(r.hi)})',f'{r.p:.3f}'],'')

# ============ T13: S10 permutation ============
rep_cell(13,'46 of 1,886 (median minimum achievable overlap 0.145); these retained their minimum-overlap position','1 of 1,296 (minimum achievable overlap 0.140); it retained its minimum-overlap position','')
rep_cell(13,'median 0.000, mean 0.008, maximum 0.545; 97.6% at or below 0.05',f'median {pm["iou_median"]:.3f}, mean {pm["iou_mean"]:.3f}, maximum {pm["iou_max"]:.3f}; {100*pm["iou_frac_le_0.05"]:.1f}% at or below 0.05','')
rep_cell(13,'0.585 (0.057)',f'{pm["null_mean"]:.3f} ({pm["null_sd"]:.3f})','')
rep_cell(13,'0.397–0.778',f'{pm["null_min"]:.3f}–{pm["null_max"]:.3f}','')
null=pd.read_csv(D+'permutation_null.csv').auroc.values
rep_cell(13,'0.479–0.692',f'{np.percentile(null,2.5):.3f}–{np.percentile(null,97.5):.3f}','')
rep_cell(13,'0.538',f'{pm["true_wound_auroc"]:.3f}','')
rep_cell(13,'20',f'{pm["true_wound_percentile"]:.0f}','')
rep_cell(13,'240 of 300',f'{int(pm["n_ge_true_wound"])} of 300','')
rep_cell(13,'0.80',f'{pm["one_sided_p_wound_better"]:.2f}','')
rep_cell(13,'0.20',f'{1-pm["one_sided_p_wound_better"]+1/301:.2f}','')
rep_cell(13,'0.641, at the 84th percentile of this distribution',f'{pm["locked_shift_auroc_rescored"]:.3f}, at the {pm["locked_shift_percentile"]:.0f}th percentile of this distribution','')

doc.save(F)
print('\n'.join(LOG)); print(f'\n{sum(1 for l in LOG if l.startswith("OK"))} ok, {sum(1 for l in LOG if l.startswith("MISS"))} missed')
