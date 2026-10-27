# -*- coding: utf-8 -*-
"""Round 2 (Mahdi's Oct 26 file): tracked edits for his eight 'Zain:' comments."""
import sys,os,re,copy,zipfile,shutil,datetime,random,json
sys.path.insert(0,'/Users/zainjaffar/ssi_work/mb_review'); sys.path.insert(0,'/Users/zainjaffar/ssi_work/paper1_2026-08-15')
import numpy as np,pandas as pd
from lxml import etree as ET
from tracked_lx import TrackedDocx, W, AUTHOR
D='/Users/zainjaffar/ssi_work/paper1_2026-08-15/out/postop_only/derived/'
SRC='/Users/zainjaffar/ssi_work/mb_review/r2/MB-Oct26.docx'; OUT='/Users/zainjaffar/ssi_work/mb_review/r2/MB-Oct26-ZJ.docx'
M='−'; EN='–'
f3=lambda x:f'{float(x):.3f}'.replace('-',M); s3=lambda x:f'{float(x):+.3f}'.replace('-',M); cito=lambda lo,hi:f'{s3(lo)} to {s3(hi)}'
d=TrackedDocx(SRC); LOG=d.log
# ---------- helpers: tracked row insertion, tracked paragraph deletion, span-based rewrite ----------
def insert_row_after(tr,texts):
    new=copy.deepcopy(tr)
    for x in new.iter(W+'ins'): x.getparent().remove(x) if x.getparent().tag!=W+'rPr' else None
    for x in list(new.iter(W+'del')):
        if x.getparent().tag!=W+'rPr': x.getparent().remove(x)
    tcs=list(new.iter(W+'tc'))
    for tc,txt in zip(tcs,texts):
        ps=tc.findall(W+'p'); p=ps[0]
        for extra in ps[1:]: tc.remove(extra)
        rpr=None
        for r in p.findall(W+'r'):
            if rpr is None and r.find(W+'rPr') is not None: rpr=copy.deepcopy(r.find(W+'rPr'))
            p.remove(r)
        for x in list(p):
            if x.tag in (W+'ins',W+'del'): p.remove(x)
        if txt: p.append(d._wrap('ins',d._mk_run(rpr,txt)))
    trpr=new.find(W+'trPr')
    if trpr is None: trpr=ET.Element(W+'trPr'); new.insert(0,trpr)
    for x in trpr.findall(W+'ins')+trpr.findall(W+'del'): trpr.remove(x)
    m=ET.SubElement(trpr,W+'ins'); m.set(W+'id',d._nid()); m.set(W+'author',AUTHOR); m.set(W+'date',d._date)
    tr.addnext(new); return new
def delete_paragraph(p):
    for r in list(p.iter(W+'r')):
        if r.find(W+'t') is None: continue
        par=r.getparent(); i=par.index(r); par.remove(r)
        for x in r.findall(W+'t'): x.tag=W+'delText'
        par.insert(i,d._wrap('del',r))
    ppr=p.find(W+'pPr')
    if ppr is None: ppr=ET.Element(W+'pPr'); p.insert(0,ppr)
    rpr=ppr.find(W+'rPr')
    if rpr is None: rpr=ET.SubElement(ppr,W+'rPr')
    m=ET.SubElement(rpr,W+'del'); m.set(W+'id',d._nid()); m.set(W+'author',AUTHOR); m.set(W+'date',d._date)
def ptext(p): return ''.join(t.text or '' for t in p.iter(W+'t'))
# ---------- 1. truncation ----------
tr=pd.read_csv(D+'prediagnosis_truncation_fullcohort_postop.csv').iloc[0]
d.replace(r'Among 170 patients with dated photographs and 12 dated SSI events, restricting SSI-positive patient aggregation.*?within its bootstrap uncertainty\.',
  f'Among 170 patients with dated photographs and 12 dated SSI events, two SSI-positive patients had every photograph obtained on or after diagnosis and therefore have no prediagnosis score, so the comparison was made among the remaining {int(tr.n_pts)} patients and {int(tr.n_events)} events. Restricting SSI-positive patient aggregation to photographs obtained strictly before clinical diagnosis changed patient-level AUROC from {f3(tr.auroc_all)} to {f3(tr.auroc_prediagnosis)}, a difference of {s3(tr.delta)} (95% CI, {cito(tr.lo,tr.hi)}; P={float(tr.p_sign):.2f}, two-sided sign test on the same patient-clustered bootstrap); photographs obtained on or after diagnosis raised the all-image estimate only within its bootstrap uncertainty.','truncation sentences')
# ---------- 2. S12 rows + footnote ----------
fc=pd.read_csv(D+'foundation_paired_contrasts_postop.csv').set_index('contrast')
C=lambda k:f"{s3(fc.loc[k].d)} ({cito(fc.loc[k].lo,fc.loc[k].hi)}, Holm P={float(fc.loc[k].holm_p):.2f})"
anchor=None
for tb in d.root.iter(W+'tbl'):
    if 'DINOv2' not in ''.join(x.text or '' for x in tb.iter(W+'t')): continue
    for trr in tb.iter(W+'tr'):
        if ptext(list(trr.iter(W+'tc'))[0]).strip().startswith('Complement of shifted region − true wound'): anchor=trr
if anchor is not None:
    r1=insert_row_after(anchor,['Circularly shifted region − true wound',C('sham_wound_only - wound_only'),''])
    r2=insert_row_after(r1,['Full image − true wound',C('full_image - wound_only'),''])
    r3=insert_row_after(r2,['True background − full image',C('background_only - full_image'),'']); LOG.append('OK   S12: three contrast rows inserted')
else: LOG.append('MISS S12 anchor row')
d.replace(r'The two listed contrasts are true background minus true wound and the complement of the shifted region minus true wound; after Holm adjustment across the five contrasts of the family, no contrast remained significant \(all adjusted P=1\.00 or higher\)\.',
  'All five contrasts of the Holm family are listed; none remained significant after adjustment (all adjusted P=1.00).','S12 footnote')
# ---------- 3. citations in the Introduction and the [8,20] sentence ----------
d.replace(r'expanded beyond models based on electronic health record data to include models using patient-generated incision photographs \[7\]','expanded beyond models based on electronic health record data [7] to include models using patient-generated incision photographs [9,18,19]','Intro: EHR [7] / image models [9,18,19]')
d.replace(r'capture less clinical context than the electronic health record \[7\]','capture less clinical context than the electronic health record [8]','Intro: EHR context -> McLean multimodal [8]')
d.replace(r'rather than mere nuisance variation \[8,20\]','rather than mere nuisance variation [8]','[8,20] -> [8]')
# ---------- 4. SurgWound entry + delete Norman ----------
d.replace(r'Xu X, et al\. SurgWound-Bench: a benchmark for surgical wound assessment\..*?(?=\n)','Xu J, Yin C, Chatzipanagiotou OP, et al. SurgWound-Bench: a benchmark for surgical wound diagnosis. npj Digit Med. 2026;9:632. doi:10.1038/s41746-026-02791-3. Dataset: https://huggingface.co/datasets/xuxuxuxuxu/SurgWound','SurgWound full citation')
body=d.root.find(W+'body'); kids=list(body)
i0=next(i for i,k in enumerate(kids) if k.tag==W+'p' and ptext(k).strip().lower()=='references'); refs=[]
for i in range(i0+1,len(kids)):
    s=ptext(kids[i]).strip()
    if re.match(r'Table 1\b',s) or s.lower().startswith('supplementary') or s.startswith('Figure'): break
    if s: refs.append(kids[i])
assert len(refs)==45, len(refs)
norman=[p for p in refs if ptext(p).startswith('Norman G')][0]; delete_paragraph(norman); LOG.append('OK   Norman entry deleted (tracked)')
old_entries={i+1:ptext(p) for i,p in enumerate(refs)}   # old number -> entry text
# ---------- 5. renumber by first appearance (text and list together) ----------
t=d.build_index(); refpos=t.lower().find('\nreferences\n'); body_text=t
order=[]
for m in re.finditer(r'\[(\d{1,2}(?:,\s*\d{1,2})*)\]',body_text):
    if refpos<m.start()<refpos+60000 and False: pass
    for n in re.split(r',\s*',m.group(1)):
        n=int(n)
        if n in old_entries and n!=17 and n not in order: order.append(n)
missing=[n for n in old_entries if n!=17 and n not in order]; order+=missing   # uncited entries (none expected) go last
remap={old:new for new,old in enumerate(order,start=1)}
json.dump({'order':order,'remap':remap},open('/Users/zainjaffar/ssi_work/mb_review/r2/renumber_map.json','w'),indent=1)
# rewrite every numeric bracket in one pass over a fixed index, last match first
d.build_index(); matches=list(re.finditer(r'\[(\d{1,2}(?:,\s*\d{1,2})*)\]',d.flat)); n_re=0
for m in reversed(matches):
    nums=[int(x) for x in re.split(r',\s*',m.group(1))]
    if any(n not in remap for n in nums): continue
    new='['+','.join(str(x) for x in sorted(remap[n] for n in nums))+']'
    if new==m.group(0): continue
    i,j=m.span(); touched=[(el,s,e) for el,s,e in d.spans if el is not None and e>i and s<j]
    for k,(el,s,e) in enumerate(touched):
        run=d._run_of(el); par=run.getparent(); rpr=run.find(W+'rPr'); txt=el.text or ''; a=max(i-s,0); b=min(j-s,len(txt)); pre,mid,post=txt[:a],txt[a:b],txt[b:]
        if d._own_ins(run) is not None: el.text=pre+(new if k==0 else '')+post; continue
        pos=par.index(run); el.text=pre; seq=[]
        if mid: seq.append(d._wrap('del',d._mk_run(rpr,mid,deleted=True)))
        if k==0: seq.append(d._wrap('ins',d._mk_run(rpr,new)))
        if post: seq.append(d._mk_run(rpr,post))
        for q,x in enumerate(seq): par.insert(pos+1+q,x)
        if not pre: par.remove(run)
    n_re+=1
LOG.append(f'OK   renumbered {n_re} citation brackets ({len(order)} entries in new order)')
# reorder the list: delete the remaining 44 entries (tracked) and insert them in the new order after the last one
d.build_index(); new_text={}
for old,p in enumerate(refs,start=1):
    if old==17: continue
    txt=ptext(p)  # accepted view of the entry (SurgWound already replaced in place)
    new_text[remap[old]]=txt
for p in refs:
    if p is norman: continue
    delete_paragraph(p)
anchor=refs[-1]
for k in range(1,len(order)+1): anchor=d.insert_paragraph_after(anchor,new_text[k])
LOG.append('OK   reference list reordered (tracked delete + insert, auto-numbered)')
d.sweep_minus(); d.save(OUT)
# ---------- 6. Figure 3 image swap ----------
z=zipfile.ZipFile(OUT); parts={n:z.read(n) for n in z.namelist()}; z.close()
rels=parts['word/_rels/document.xml.rels'].decode(); tgt=re.search(r'Id="rId14"[^>]*Target="([^"]+)"',rels).group(1)
parts['word/'+tgt]=open(D+'figures/figure3_architecture_context_postop.png','rb').read(); LOG.append(f'OK   Figure 3 image replaced ({tgt})')
# ---------- 7. comment replies ----------
REPLY={
 9005:'You are right that they look inconsistent. The 0.63 was a mid-P with ties counted half. From the same bootstrap the two-sided sign test gives P=0.73: 63% of resamples are negative, 10% are exactly zero (resamples that draw none of the four SSI patients whose score changes) and 27% are positive but all tiny, so the 97.5th percentile is +0.006 while a third of the mass sits at or above zero. I put P=0.73 and say which test it is. Also rewrote the sentence so it is clear that 0.818 and 0.779 are on the 168/10 set and 170/12 is the dated frame before the two patients drop out. CSV updated.',
 9006:'Correct, the patient-resampled interval is the right one given the Methods, and its P is 0.35. The 0.09 in my email was the image-resampled version. Nothing else to change.',
 9007:'Yes. The CSV gives the AUROC for abdomen (0.134 and 0.173); for chest it is one minus that, 0.866 and 0.827. Your wording is right.',
 9008:'Yes, 9.5 is the exact value, 4-10 was me rounding. 4-9.5 is fine.',
 9009:'Added the three rows from the CSV (shifted minus wound +0.131, full minus wound +0.090, background minus full +0.036, all Holm P=1.00) and changed the footnote to say all five are listed.',
 9010:'Fixed, panel B now says Prior imaging behavior and the new image is swapped in.',
 9011:'Done. Norman deleted and everything renumbered by first appearance, text and list together, 44 entries now. Because the list is auto-numbered, the reorder shows as a deleted block plus an inserted block in All Markup; in No Markup view the numbering reads straight through. In the Introduction, Bucher now sits on "models based on electronic health record data" where it belongs, the image-based models cite Tabja Bortesi, Muaddi and Bontekoning, and the sentence on images capturing less clinical context than the record cites McLean’s multimodal paper. SurgWound is now the published paper: Xu et al., npj Digit Med 2026;9:632, DOI checked, dataset link kept. On [8,20]: you are right about Bluebelle, I removed it and left McLean (images plus patient-reported and record data), which is the closest support for that sentence.'}
W14='http://schemas.microsoft.com/office/word/2010/wordml'; W15='http://schemas.microsoft.com/office/word/2012/wordml'; W16CID='http://schemas.microsoft.com/office/word/2016/wordml/cid'; W16CEX='http://schemas.microsoft.com/office/word/2018/wordml/cex'
com=ET.fromstring(parts['word/comments.xml']); cex=ET.fromstring(parts['word/commentsExtended.xml']); cid=ET.fromstring(parts['word/commentsIds.xml']); cexx=ET.fromstring(parts['word/commentsExtensible.xml']); ppl=ET.fromstring(parts['word/people.xml']); docx_=ET.fromstring(parts['word/document.xml'])
existing=list(com.iter(W+'comment')); nextid=max(int(c.get(W+'id')) for c in existing)+1
paraid_of={c.get(W+'id'):list(c.iter(W+'p'))[-1].get('{%s}paraId'%W14) for c in existing}
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'); rnd=random.Random(20261026); hexid=lambda:'%08X'%rnd.randint(0x10000000,0x7FFFFFFF)
tmpl=[c for c in existing if c.get(W+'author')=='Mahdi Babaei'][0]
for pid_,text in REPLY.items():
    nid=str(nextid); nextid+=1; pid8=hexid(); dur=hexid()
    c=copy.deepcopy(tmpl); c.set(W+'id',nid); c.set(W+'author',AUTHOR); c.set(W+'date',now); c.set(W+'initials','ZJ')
    ps=list(c.iter(W+'p')); p=ps[0]; p.set('{%s}paraId'%W14,pid8); p.set('{%s}textId'%W14,'77777777')
    runs=[r for r in p.findall(W+'r') if r.find(W+'annotationRef') is None]
    for r in runs[1:]: p.remove(r)
    runs[0].find(W+'t').text=text
    for extra in ps[1:]: extra.getparent().remove(extra)
    com.append(c)
    e=ET.SubElement(cex,'{%s}commentEx'%W15); e.set('{%s}paraId'%W15,pid8); e.set('{%s}paraIdParent'%W15,paraid_of[str(pid_)]); e.set('{%s}done'%W15,'0')
    e=ET.SubElement(cid,'{%s}commentId'%W16CID); e.set('{%s}paraId'%W16CID,pid8); e.set('{%s}durableId'%W16CID,dur)
    e=ET.SubElement(cexx,'{%s}commentExtensible'%W16CEX); e.set('{%s}durableId'%W16CEX,dur); e.set('{%s}dateUtc'%W16CEX,now)
    rs=docx_.find('.//'+W+'commentRangeStart[@'+W+'id="'+str(pid_)+'"]'); re_=docx_.find('.//'+W+'commentRangeEnd[@'+W+'id="'+str(pid_)+'"]'); ref=docx_.find('.//'+W+'commentReference[@'+W+'id="'+str(pid_)+'"]')
    if rs is not None and re_ is not None: s2=ET.Element(W+'commentRangeStart'); s2.set(W+'id',nid); rs.addnext(s2); e2=ET.Element(W+'commentRangeEnd'); e2.set(W+'id',nid); re_.addnext(e2)
    refrun=ref.getparent(); r2=copy.deepcopy(refrun); r2.find(W+'commentReference').set(W+'id',nid); refrun.addnext(r2)
    LOG.append(f'reply added to comment {pid_}')
if not any(p.get('{%s}author'%W15)==AUTHOR for p in ppl.iter('{%s}person'%W15)):
    pe=ET.SubElement(ppl,'{%s}person'%W15); pe.set('{%s}author'%W15,AUTHOR); pi=ET.SubElement(pe,'{%s}presenceInfo'%W15); pi.set('{%s}providerId'%W15,'None'); pi.set('{%s}userId'%W15,AUTHOR)
for n,el in (('word/comments.xml',com),('word/commentsExtended.xml',cex),('word/commentsIds.xml',cid),('word/commentsExtensible.xml',cexx),('word/people.xml',ppl),('word/document.xml',docx_)):
    parts[n]=ET.tostring(el,xml_declaration=True,encoding='UTF-8',standalone=True)
zo=zipfile.ZipFile(OUT+'.tmp','w',zipfile.ZIP_DEFLATED)
for n,data in parts.items(): zo.writestr(n,data)
zo.close(); shutil.move(OUT+'.tmp',OUT)
print('\n'.join(LOG)); print('WROTE',OUT)
