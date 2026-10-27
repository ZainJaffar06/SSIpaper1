# -*- coding: utf-8 -*-
"""Word-safe tracked-change emitter: edits the python-docx Document's lxml tree in place and saves
through python-docx, so namespaces, relationships, media, comments and settings are preserved
byte-for-byte except for the edited parts. Replacements become <w:del>/<w:ins> revisions by AUTHOR."""
import re, copy, datetime
from lxml import etree as ET
import docx
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
XML_SPACE='{http://www.w3.org/XML/1998/namespace}space'
AUTHOR='Zain Jaffar'
class TrackedDocx:
    def __init__(self, path):
        self.docx=docx.Document(path); self.root=self.docx.element
        ids=[int(e.get(W+'id')) for e in self.root.iter() if e.tag in (W+'ins',W+'del') and (e.get(W+'id') or '').isdigit()]
        self._id=max(ids+[5000])+1; self._date=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'); self.log=[]
    def _nid(self): self._id+=1; return str(self._id)
    def save(self,out): self.docx.save(out)
    # ---- text index (accepted view: w:t everywhere except inside w:del) ----
    def build_index(self):
        self.spans=[]; pos=0; parts=[]
        for p in self.root.iter(W+'p'):
            for t in p.iter(W+'t'):
                a=t.getparent(); dead=False
                while a is not None:
                    if a.tag==W+'del': dead=True; break
                    a=a.getparent()
                if dead: continue
                s=t.text or ''; self.spans.append((t,pos,pos+len(s))); parts.append(s); pos+=len(s)
            parts.append('\n'); self.spans.append((None,pos,pos+1)); pos+=1
        self.flat=''.join(parts); return self.flat
    def _own_ins(self,el):
        a=el.getparent()
        while a is not None:
            if a.tag==W+'ins' and a.get(W+'author')==AUTHOR: return a
            a=a.getparent()
        return None
    def _mk_run(self,rpr,text,deleted=False):
        r=ET.Element(W+'r')
        if rpr is not None: r.append(copy.deepcopy(rpr))
        t=ET.SubElement(r,W+('delText' if deleted else 't')); t.text=text; t.set(XML_SPACE,'preserve'); return r
    def _wrap(self,tag,run):
        e=ET.Element(W+tag); e.set(W+'id',self._nid()); e.set(W+'author',AUTHOR); e.set(W+'date',self._date); e.append(run); return e
    @staticmethod
    def _run_of(t):
        r=t.getparent()
        while r is not None and r.tag!=W+'r': r=r.getparent()
        return r
    def replace(self,pattern,new,label='',flags=re.S,count=1):
        n=0
        for _ in range(max(1,count)):
            if not self._replace_once(pattern,new,flags): break
            n+=1
        self.log.append(('OK  ' if n else 'MISS')+f' [{n}] {label}'); return n
    def _replace_once(self,pattern,new,flags):
        self.build_index(); m=re.search(pattern,self.flat,flags)
        if not m: return 0
        i,j=m.span(); new=m.expand(new) if '\\' in new else new
        touched=[(el,s,e) for el,s,e in self.spans if el is not None and e>i and s<j]
        if not touched: return 0
        for n,(el,s,e) in enumerate(touched):
            run=self._run_of(el); par=run.getparent(); rpr=run.find(W+'rPr'); txt=el.text or ''
            a=max(i-s,0); b=min(j-s,len(txt)); pre,mid,post=txt[:a],txt[a:b],txt[b:]
            if self._own_ins(run) is not None:
                el.text=pre+(new if n==0 else '')+post
                if not el.text:
                    ins=run.getparent(); ins.remove(run)
                    if ins.tag==W+'ins' and len(ins)==0: ins.getparent().remove(ins)
                continue
            pos=par.index(run); el.text=pre; seq=[]
            if mid: seq.append(self._wrap('del',self._mk_run(rpr,mid,deleted=True)))
            if n==0 and new: seq.append(self._wrap('ins',self._mk_run(rpr,new)))
            if post: seq.append(self._mk_run(rpr,post))
            for k,x in enumerate(seq): par.insert(pos+1+k,x)
            if not pre: par.remove(run)
        return 1
    def set_cell(self,tc,new):
        live=[]
        for r in tc.iter(W+'r'):
            if r.find(W+'t') is None: continue
            a=r.getparent(); dead=False
            while a is not None and a is not tc:
                if a.tag==W+'del': dead=True; break
                a=a.getparent()
            if not dead: live.append(r)
        old=''.join((r.find(W+'t').text or '') for r in live).strip()
        if old==new: return 0
        rpr=live[0].find(W+'rPr') if live else None
        for r in live:
            if self._own_ins(r) is not None:
                ins=r.getparent(); ins.remove(r)
                if ins.tag==W+'ins' and len(ins)==0: ins.getparent().remove(ins)
                continue
            par=r.getparent(); i=par.index(r); par.remove(r)
            for x in r.findall(W+'t'): x.tag=W+'delText'
            par.insert(i,self._wrap('del',r))
        if new: tc.find(W+'p').append(self._wrap('ins',self._mk_run(rpr,new)))
        return 1
    def set_row(self,label,vals,lab,table_hint=None):
        for tb in self.root.iter(W+'tbl'):
            ttext=' | '.join(''.join(t.text or '' for t in tb.iter(W+'t')) for _ in [0])
            if table_hint and not re.search(table_hint,ttext): continue
            for tr in tb.iter(W+'tr'):
                tcs=list(tr.iter(W+'tc'))
                if not tcs: continue
                first=''.join(t.text or '' for t in tcs[0].iter(W+'t')).strip()
                if re.fullmatch(label,first):
                    for tc,v in zip(tcs[1:],vals):
                        if v is not None: self.set_cell(tc,v)
                    self.log.append(f'OK   row {lab}'); return True
        self.log.append(f'MISS row {lab}'); return False
    def insert_paragraph_after(self,anchor_p,text,copy_ppr=True):
        p=ET.Element(W+'p'); src=anchor_p.find(W+'pPr')
        if copy_ppr and src is not None:
            ppr=copy.deepcopy(src)
            for x in ppr.findall(W+'rPr'): ppr.remove(x)
            p.append(ppr)
        else: ppr=ET.SubElement(p,W+'pPr')
        rpr=ET.SubElement(ppr,W+'rPr'); mark=ET.SubElement(rpr,W+'ins'); mark.set(W+'id',self._nid()); mark.set(W+'author',AUTHOR); mark.set(W+'date',self._date)
        p.append(self._wrap('ins',self._mk_run(None,text))); anchor_p.addnext(p); return p
    def sweep_minus(self):
        n=0
        for ins in self.root.iter(W+'ins'):
            if ins.get(W+'author')!=AUTHOR: continue
            for x in ins.iter(W+'t'):
                if x.text and re.search(r'(?<![\w–—-])-(?=\d)',x.text): x.text=re.sub(r'(?<![\w–—-])-(?=\d)','−',x.text); n+=1
        return n
