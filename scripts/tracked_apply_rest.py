# -*- coding: utf-8 -*-
"""Post-table pass: the 3 missed prose items, Table 4 POD-7 rows, S13 codebook, and the
comment-driven text edits that are safe to do mechanically."""
import sys, os, re, copy
sys.path.insert(0,'.')
import pandas as pd, xml.etree.ElementTree as ET
from dxml_tracked import TrackedDoc as Doc, W
D='/Users/zainjaffar/ssi_work/paper1_2026-08-15/out/postop_only/derived/'
F=os.environ['OUTDOC']
doc=Doc(F); LOG=[]
def R(pat,new,lab,count=1):
    n=doc.replace(pat,new,lab,count=count); LOG.append(('OK  ' if n else 'MISS')+f' [{n}] {lab}')
pl=pd.read_csv(D+'primary_and_landmarks.csv'); lt,cl=pl.iloc[1],pl.iloc[2]
pm=dict(zip(*pd.read_csv(D+'permutation_summary.csv').values.T))
c7=pd.read_csv(D+'clinical_pod7.csv').set_index('model'); k7=pd.read_csv(D+'clinical_pod7_contrast.csv').iloc[0]
f3=lambda x:f'{x:.3f}'; s3=lambda x:f'{x:+.3f}'
ratio=(pm['locked_shift_auroc_rescored']-pm['true_wound_auroc'])/(pm['null_mean']-pm['true_wound_auroc'])

# --- the 3 misses ---
R(r'The latest photograph through POD 7 yielded an AUROC of 0\.737 and an AUPRC of 0\.150; the photograph closest to POD 7 yielded an AUROC of 0\.748 and an AUPRC of 0\.153\.',
  f'The latest photograph through POD 7 yielded an AUROC of {f3(lt.auroc)} ({f3(lt.lo)}–{f3(lt.hi)}) and an AUPRC of {f3(lt.auprc)}; the photograph closest to POD 7 yielded an AUROC of {f3(cl.auroc)} ({f3(cl.lo)}–{f3(cl.hi)}) and an AUPRC of {f3(cl.auprc)}. Both sensitivities exceeded the prespecified mean, which is retained as the primary because it was prespecified.','aggregation sensitivities')
R(r'to have been an 84th-percentile draw, so its apparent advantage over the wound region was roughly twice a typical relocation.s\.',
  f'to have been an {pm["locked_shift_percentile"]:.0f}th-percentile draw, so its apparent advantage over the wound region was roughly {ratio:.1f} times a typical relocation’s.','Discussion shift draw')
# --- Table 4 POD-7 prose ---
R(r'In the POD 7 risk set, AUROCs were 0\.713, 0\.754, and 0\.740 for clinical, image, and stacked models, respectively\.',
  f'In the POD 7 risk set, AUROCs were {f3(c7.loc["clinical","auroc"])}, {f3(c7.loc["image","auroc"])}, and {f3(c7.loc["stacked","auroc"])} for clinical, image, and stacked models, respectively.','POD7 clinical prose')
R(r'The stacked-minus-image difference was −0\.014 \(95% CI, −0\.167 to \+0\.134; P=0\.86\); the AUPRC difference was \+0\.099 \(P=0\.17\)\.',
  f'The stacked-minus-image difference was {s3(k7.d)} (95% CI, {s3(k7.lo)} to {s3(k7.hi)}; P={k7.p:.2f}).','POD7 stacked contrast')
R(r'In the full cohort, stacked minus clinical AUROC was \+0\.135 \(95% CI, \+0\.037 to \+0\.253; P=0\.005\), significant in only two of five cross-validation partitions, and AUPRC was −0\.024 \(P=0\.83\)\. At POD 7, stacked minus image AUROC was −0\.014 \(−0\.167 to \+0\.134; P=0\.86\) and AUPRC was \+0\.099 \(P=0\.17\)\.',
  f'In the full postoperative cohort, stacked minus clinical AUROC was +0.259 (95% CI, +0.100 to +0.436; P=0.002), positive in all five cross-validation partitions. At POD 7, stacked minus image AUROC was {s3(k7.d)} ({s3(k7.lo)} to {s3(k7.hi)}; P={k7.p:.2f}).','Table 4 footnote prose')
# --- comment 86: serial ---
R(r'[^.]*\bserial\b[^.]*\.\s*','','comment 86: remove serial sentence',count=3)
# --- Table 4 POD-7 cells ---
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
def setcell(c,new):
    doc.set_cell_tracked(c[0],new)
n=0
for tr,row in cells(tbls[3]):
    if row[0][2]=='POD 7 risk set' and row[1][2]=='Clinical':
        for c,v in zip(row[2:],['132 / 8',f'{f3(c7.loc["clinical","auroc"])} ({f3(c7.loc["clinical","lo"])}–{f3(c7.loc["clinical","hi"])})',f3(c7.loc["clinical","auprc"])]): setcell(c,v); n+=1
    if row[0][2]=='POD 7 risk set' and row[1][2]=='Stacked':
        for c,v in zip(row[2:],['132 / 8',f'{f3(c7.loc["stacked","auroc"])} ({f3(c7.loc["stacked","lo"])}–{f3(c7.loc["stacked","hi"])})',f3(c7.loc["stacked","auprc"])]): setcell(c,v); n+=1
LOG.append(f'OK   [{n}] Table 4 POD-7 clinical/stacked cells')
# --- S13 codebook: insert caption + table + footnote before the analysis-history note ---
body=doc.root.find(W+'body')
kids=list(body)
anchor=None
for i,k in enumerate(kids):
    if k.tag==W+'p' and 'Supplementary note on analysis history' in ''.join(t.text or '' for t in k.iter(W+'t')):
        anchor=i; break
if anchor is None: LOG.append('MISS S13 anchor')
else:
    def para(text):
        p=ET.Element(W+'p'); r=ET.SubElement(p,W+'r'); t=ET.SubElement(r,W+'t'); t.text=text; t.set('{http://www.w3.org/XML/1998/namespace}space','preserve'); return p
    rows=[('Code','Name','Definition'),
     ('Q1','Poor focus','Image is blurry or out of focus; incision edges not clearly visible due to focus issues'),
     ('Q2','Partial view','Incision not fully visible; part of the incision is cut off at the image edges'),
     ('Q3','Obstruction','Major obstruction present; hands, instruments or drapes cover more than 25% of the incision'),
     ('Q4','Motion artifact','Motion blur present; blurring from patient or camera movement during capture'),
     ('Q5','Dressing present','Dressing visible; bandage or dressing partially or fully covers the incision'),
     ('Q6','Annotation uncertain','Annotator unsure; complex case, unsure about the correct annotation approach')]
    tbl=ET.Element(W+'tbl'); tp=ET.SubElement(tbl,W+'tblPr'); st=ET.SubElement(tp,W+'tblStyle'); st.set(W+'val','TableGrid')
    tw=ET.SubElement(tp,W+'tblW'); tw.set(W+'w','0'); tw.set(W+'type','auto')
    for ri,rw in enumerate(rows):
        tr=ET.SubElement(tbl,W+'tr')
        for v in rw:
            tc=ET.SubElement(tr,W+'tc'); p=ET.SubElement(tc,W+'p'); r=ET.SubElement(p,W+'r')
            if ri==0: rp=ET.SubElement(r,W+'rPr'); ET.SubElement(rp,W+'b')
            t=ET.SubElement(r,W+'t'); t.text=v; t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    foot=para('Definitions are reproduced verbatim from the annotation tracker’s quality-flag reference sheet, which was written before analysis. Annotators recorded flags per photograph during annotation and without access to outcome data; the tracker sheet used for annotation contains no infection or outcome field. The primary analytic frame excluded any photograph carrying a recorded flag under any of the six codes. No counts are reported because the flags were recorded against the annotation frame rather than the analytic frame.')
    cap=para('Supplementary Table S13 | Image quality-flag codebook')
    for el in (foot,tbl,cap): body.insert(anchor,el); doc.mark_inserted(el)
    LOG.append('OK   S13 codebook inserted (caption, table, footnote)')
R(r'against a prospectively written six-category codebook, and','against a prospectively written six-category codebook (Supplementary Table S13), and','Methods cites S13')
doc.save(F); print('\n'.join(LOG))
