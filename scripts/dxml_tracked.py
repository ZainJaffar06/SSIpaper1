# -*- coding: utf-8 -*-
"""Tracked-change emitter on top of dxml.Doc. Replacements become real Word revisions:
   <w:del w:id w:author w:date><w:r><w:rPr/><w:delText>old</w:delText></w:r></w:del>
   <w:ins ...><w:r><w:rPr/><w:t>new</w:t></w:r></w:ins>
so Mahdi sees them under Review > Track Changes and can accept/reject. Run formatting
(rPr) is copied from the run being edited. IDs continue from the document's current max."""
import re, copy, datetime
import xml.etree.ElementTree as ET
from dxml import Doc, W
XML_SPACE='{http://www.w3.org/XML/1998/namespace}space'
AUTHOR='Zain Jaffar'

class TrackedDoc(Doc):
    def __init__(self, path):
        super().__init__(path)
        ids=[int(e.get(W+'id')) for e in self.root.iter() if e.tag in (W+'ins',W+'del') and e.get(W+'id','').isdigit()]
        self._id=max(ids+[1000])+1
        self._date=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        self._parents=None
    def _nid(self): self._id+=1; return str(self._id)
    def _parentmap(self):
        self._parents={c:p for p in self.root.iter() for c in p}; return self._parents
    def _run_of(self, t_el):
        pm=self._parents or self._parentmap(); r=pm.get(t_el)
        while r is not None and r.tag!=W+'r': r=pm.get(r)
        return r
    def _mk_run(self, rpr, text, deleted=False):
        r=ET.Element(W+'r')
        if rpr is not None: r.append(copy.deepcopy(rpr))
        t=ET.SubElement(r, W+('delText' if deleted else 't')); t.text=text; t.set(XML_SPACE,'preserve'); return r
    def _wrap(self, tag, run):
        e=ET.Element(W+tag); e.set(W+'id',self._nid()); e.set(W+'author',AUTHOR); e.set(W+'date',self._date); e.append(run); return e
    def _own_ins(self, el):
        """Return the enclosing w:ins authored by me, if any."""
        pm=self._parents or self._parentmap(); a=pm.get(el)
        while a is not None:
            if a.tag==W+'ins' and a.get(W+'author')==AUTHOR: return a
            a=pm.get(a)
        return None
    def _drop_run(self, run):
        """Remove a run that sits inside my own insertion; drop the insertion if it becomes empty."""
        pm=self._parents; par=pm[run]; par.remove(run)
        if par.tag==W+'ins' and len(par)==0: pm[par].remove(par)
    def mark_inserted(self, el):
        """Mark a freshly built element (paragraph or table) as a tracked insertion: every run wrapped
        in w:ins, every row flagged w:trPr/w:ins, every paragraph mark flagged pPr/rPr/ins."""
        for p in ([el] if el.tag==W+'p' else list(el.iter(W+'p'))):
            ppr=p.find(W+'pPr')
            if ppr is None: ppr=ET.Element(W+'pPr'); p.insert(0,ppr)
            rpr=ppr.find(W+'rPr')
            if rpr is None: rpr=ET.SubElement(ppr,W+'rPr')
            m=ET.SubElement(rpr,W+'ins'); m.set(W+'id',self._nid()); m.set(W+'author',AUTHOR); m.set(W+'date',self._date)
            for r in list(p.findall(W+'r')):
                i=list(p).index(r); p.remove(r); p.insert(i,self._wrap('ins',r))
        if el.tag==W+'tbl':
            for tr in el.iter(W+'tr'):
                trpr=tr.find(W+'trPr')
                if trpr is None: trpr=ET.Element(W+'trPr'); tr.insert(0,trpr)
                m=ET.SubElement(trpr,W+'ins'); m.set(W+'id',self._nid()); m.set(W+'author',AUTHOR); m.set(W+'date',self._date)
        self._parents=None; return el
    def replace_tracked(self, pattern, new, label='', flags=re.S):
        """One tracked replacement of the first regex match. Handles matches that span runs."""
        self.build_index(); self._parentmap()
        m=re.search(pattern,self.flat,flags)
        if not m: return 0
        i,j=m.span(); new=m.expand(new) if '\\' in new else new
        touched=[(el,s,e) for el,s,e in self.spans if el is not None and e>i and s<j]
        if not touched: return 0
        pm=self._parents
        for n,(el,s,e) in enumerate(touched):
            run=self._run_of(el); par=pm[run]; rpr=run.find(W+'rPr')
            txt=el.text or ''; a=max(i-s,0); b=min(j-s,len(txt))
            pre,mid,post=txt[:a],txt[a:b],txt[b:]
            pos=list(par).index(run)
            if self._own_ins(run) is not None:
                # text I inserted earlier: just rewrite it in place (an insertion that is then deleted would only clutter Mahdi's view)
                el.text=pre+(new if n==0 else '')+post
                if not el.text: self._drop_run(run)
                continue
            # original run keeps only the untouched prefix
            el.text=pre
            seq=[]
            if mid: seq.append(self._wrap('del',self._mk_run(rpr,mid,deleted=True)))
            if n==0 and new: seq.append(self._wrap('ins',self._mk_run(rpr,new)))
            if post: seq.append(self._mk_run(rpr,post))
            for k,x in enumerate(seq): par.insert(pos+1+k,x)
            if not pre and not (run.find(W+'t') is not None and (run.find(W+'t').text or '')): 
                # empty shell: drop it so Word does not show an empty run
                try: par.remove(run)
                except ValueError: pass
        self._parents=None
        return 1
    def set_cell_tracked(self, tc, new):
        """Replace a table cell's visible text as a tracked change. Every live run in the cell
        (any paragraph, inside another author's w:ins or not, excluding runs already deleted) is
        marked deleted in place; the new text is one inserted run in the first paragraph."""
        self._parentmap(); pm=self._parents
        live=[]
        for r in tc.iter(W+'r'):
            if r.find(W+'t') is None: continue
            a=pm.get(r); dead=False
            while a is not None and a is not tc:
                if a.tag==W+'del': dead=True; break
                a=pm.get(a)
            if not dead: live.append(r)
        old=''.join((r.find(W+'t').text or '') for r in live).strip()
        if old==new: return 0
        rpr=live[0].find(W+'rPr') if live else None
        for r in live:
            if self._own_ins(r) is not None: self._drop_run(r); continue
            par=pm[r]; i=list(par).index(r); par.remove(r)
            for x in r.findall(W+'t'): x.tag=W+'delText'
            par.insert(i,self._wrap('del',r))
        if new:
            p=tc.find(W+'p'); p.append(self._wrap('ins',self._mk_run(rpr,new)))
        self._parents=None; return 1
    def insert_paragraph_tracked(self, anchor_p, text, after=False, copy_ppr=True):
        """Insert a new paragraph as a tracked insertion (runs AND paragraph mark). With copy_ppr the
        anchor's paragraph properties (style, list numbering) are copied so the paragraph joins its list."""
        self._parentmap(); body=self._parents[anchor_p]
        p=ET.Element(W+'p'); src=anchor_p.find(W+'pPr')
        if copy_ppr and src is not None:
            ppr=copy.deepcopy(src)
            for x in ppr.findall(W+'rPr'): ppr.remove(x)
            p.append(ppr)
        else: ppr=ET.SubElement(p,W+'pPr')
        rpr=ET.SubElement(ppr,W+'rPr')
        mark=ET.SubElement(rpr,W+'ins'); mark.set(W+'id',self._nid()); mark.set(W+'author',AUTHOR); mark.set(W+'date',self._date)
        p.append(self._wrap('ins',self._mk_run(None,text)))
        idx=list(body).index(anchor_p)+(1 if after else 0); body.insert(idx,p); self._parents=None; return p

    # ---- drop-in compatibility with the untracked apply scripts ----
    def replace(self, pattern, repl, label="", flags=re.S, count=1):
        n=0
        for _ in range(max(1,count)):
            if not self.replace_tracked(pattern, repl, label, flags): break
            n+=1
        return n
    def delete_row_tracked(self, tr):
        """Mark a table row deleted: runs wrapped in w:del, row mark flagged with w:trPr/w:del."""
        self._parentmap()
        for tc in tr.iter(W+'tc'):
            for p in tc.findall(W+'p'):
                for r in list(p.findall(W+'r')):
                    t=r.find(W+'t')
                    if t is None: continue
                    t.tag=W+'delText'; i=list(p).index(r); p.remove(r); p.insert(i,self._wrap('del',r))
        trpr=tr.find(W+'trPr')
        if trpr is None: trpr=ET.Element(W+'trPr'); tr.insert(0,trpr)
        d=ET.SubElement(trpr,W+'del'); d.set(W+'id',self._nid()); d.set(W+'author',AUTHOR); d.set(W+'date',self._date)
        self._parents=None; return 1

    def sweep_minus(self):
        """Inside MY insertions only: ASCII hyphen before a digit -> U+2212 minus. Returns count."""
        n=0
        for ins in self.root.iter(W+'ins'):
            if ins.get(W+'author')!=AUTHOR: continue
            for x in ins.iter(W+'t'):
                if x.text and re.search(r'(?<![\w\u2013\u2014-])-(?=\d)',x.text):
                    new=re.sub(r'(?<![\w\u2013\u2014-])-(?=\d)','\u2212',x.text); n+=new!=x.text; x.text=new
        return n
