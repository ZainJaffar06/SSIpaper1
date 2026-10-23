import sys, re, xml.etree.ElementTree as ET
sys.path.insert(0,'.')
from dxml import Doc, W
F='Smartphone Images Sham paper - POSTOP (rev 2026-10-23).docx'
doc=Doc(F); t=doc.build_index()
body=doc.root.find(W+'body'); kids=list(body)
def ptext(p): return ''.join(x.text or '' for x in p.iter(W+'t'))
# reference paragraphs
ref_idx=[i for i,k in enumerate(kids) if k.tag==W+'p' and re.search(r'doi:|arXiv|Available at|Scand J Stat|MIT Press', ptext(k))]
refs={}
for i in ref_idx:
    s=ptext(kids[i]); m=re.match(r'\s*(\d{1,2})[.)]\s',s)
    refs[i]=int(m.group(1)) if m else None
if any(v is None for v in refs.values()):
    # Word auto-numbering: assign by document order
    for n,i in enumerate(ref_idx, start=1): refs[i]=n
    literal=False
else: literal=True
print(f'reference paragraphs: {len(ref_idx)} (literal numbering: {literal})')
# body text before the first reference paragraph
head=''.join(ptext(k)+'\n' for k in kids[:ref_idx[0]])
order=[]
for m in re.finditer(r'\[(\d{1,2}(?:[,–-]\d{1,2})*)\]', head):
    for n in re.split(r'[,–-]', m.group(1)):
        n=int(n)
        if n not in order: order.append(n)
print('first-appearance order:', order)
cited=set(order); allnums=set(refs.values())
print('cited but no entry:', sorted(cited-allnums), '| entries never cited:', sorted(allnums-cited))
remap={old:new for new,old in enumerate(order, start=1)}
# uncited entries go to the end, in their existing order
for old in sorted(allnums-cited): remap[old]=len(remap)+1
# rewrite every in-text bracket (body only, before references)
def rew(m):
    parts=re.split(r'([,–-])', m.group(1)); out=[]
    for p in parts:
        out.append(str(remap[int(p)]) if p.isdigit() else p)
    return '['+''.join(out)+']'
for k in kids[:ref_idx[0]]:
    for x in k.iter(W+'t'):
        if x.text and '[' in x.text: x.text=re.sub(r'\[(\d{1,2}(?:[,–-]\d{1,2})*)\]', rew, x.text)
# reorder reference paragraphs and rewrite their prefixes
paras={refs[i]:kids[i] for i in ref_idx}
for i in ref_idx: body.remove(kids[i])
anchor=ref_idx[0]
for new in range(1,len(paras)+1):
    old=[o for o,n in remap.items() if n==new][0]
    p=paras[old]
    if literal:
        ts=[x for x in p.iter(W+'t') if x.text]
        if ts: ts[0].text=re.sub(r'^\s*\d{1,2}[.)]\s', f'{new}. ', ts[0].text, count=1)
    body.insert(anchor+new-1, p)
doc.save(F)
print('renumbered', len(remap), 'references; in-text brackets rewritten')
