# -*- coding: utf-8 -*-
"""List every in-text citation [n] with its sentence and the n-th reference entry, so each
pairing can be checked by eye. Reference entries = non-empty paragraphs between the paragraph
whose text is exactly 'References' and the first paragraph starting with 'Table 1'.
Usage: python3 -I check_citations.py <docx> [--accepted]"""
import sys, re, zipfile
import xml.etree.ElementTree as ET
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
path=sys.argv[1]
root=ET.fromstring(zipfile.ZipFile(path).read('word/document.xml'))
def ptext(p): return ''.join((t.text or '') for t in p.iter() if t.tag==W+'t')   # accepted view: skips delText
paras=[ptext(p) for p in root.iter(W+'p')]
try: i0=next(i for i,t in enumerate(paras) if t.strip().lower()=='references')
except StopIteration: sys.exit('no References heading paragraph')
refs=[]
for t in paras[i0+1:]:
    if re.match(r'\s*Table 1\b',t) or t.strip().lower().startswith('supplementary'): break
    if t.strip(): refs.append(t.strip())
print(f'{len(refs)} reference entries\n')
for k,r in enumerate(refs,1): print(f'[{k}] {r[:110]}')
body='\n'.join(paras[:i0])
print('\n=== in-text citations, in order of appearance ===')
seen=set()
for m in re.finditer(r'\[(\d+(?:[,–\-–]\s*\d+)*)\]',body):
    nums=set()
    for part in re.split(r',\s*',m.group(1)):
        if re.search(r'[–\-–]',part):
            a,b=re.split(r'[–\-–]',part); nums.update(range(int(a),int(b)+1))
        else: nums.add(int(part))
    s0=body.rfind('.',0,m.start()-1)+1; s1=body.find('.',m.end()); sent=body[s0:s1+1].strip().replace('\n',' ')
    for n in sorted(nums):
        key=(n,sent[:60]); 
        if key in seen: continue
        seen.add(key); ref=refs[n-1][:90] if 0<n<=len(refs) else '!! NO SUCH ENTRY'
        print(f'\n[{n}] -> {ref}\n     ...{sent[-230:]}')
