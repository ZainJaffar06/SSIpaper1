import sys, re
sys.path.insert(0,'.')
from dxml import Doc, W
F='Smartphone Images Sham paper - POSTOP (rev 2026-10-23).docx'
doc=Doc(F); body=doc.root.find(W+'body'); kids=list(body)
def ptext(p): return ''.join(x.text or '' for x in p.iter(W+'t')).strip()
TOK=['GlobalSurg','Woelber','Gunter RL','McLean KA, Mountain','Sanger PC','Kummerow Broman','McLean KA, Sgr','Tabja Bortesi','Muaddi','Bontekoning',
     'Adebayo','Arun N','Collins GS','Mongan J','Xu X, et al. SurgWound','Geirhos','DeGrave','Zech JR, Badgeley','Badgeley MA, Zech','Ong Ly',
     'Bucher BT','Hooker S','Norman G','Bluebelle Study Group','von Elm','He K, Zhang X','Tan M, Le','Dosovitskiy','Deng J, Dong','Loshchilov',
     'Oquab','Zhang S, Xu Y','Holm S','Platt J','Zauner','Fitzpatrick TB','Fleiss JL']   # index+1 = ORIGINAL number
orig_of={}
for i,k in enumerate(kids):
    if k.tag!=W+'p': continue
    s=ptext(k)
    for n,t in enumerate(TOK, start=1):
        if s.startswith(t): orig_of[i]=n; break
print('identified:',len(orig_of),'| missing:',sorted(set(range(1,38))-set(orig_of.values())))
assert len(orig_of)==37
first_ref=min(orig_of); last_ref=max(orig_of)
ORDER_MAIN=[1,2,3,4,5,6,21,7,8,17,18,19,20,11,12,22,23,9,10,24,25,13,14,15,26,27,28,29,30,31,32,33,34,16,35]
remap={old:new for new,old in enumerate(ORDER_MAIN,start=1)}
supp=''.join(ptext(k)+'\n' for k in kids[last_ref+1:])
for m in re.finditer(r'\[(\d{1,2}(?:[,–-]\d{1,2})*)\]',supp):
    for n in re.split(r'[,–-]',m.group(1)):
        n=int(n)
        if n not in remap: remap[n]=len(remap)+1
print('remap size:',len(remap),'| supplementary-only originals:',[o for o in remap if o not in ORDER_MAIN])
def rew(m):
    parts=re.split(r'([,–-])',m.group(1))
    return '['+''.join(str(remap[int(p)]) if p.isdigit() else p for p in parts)+']'
n=0
for k in kids[last_ref+1:]:
    for x in k.iter(W+'t'):
        if x.text and '[' in x.text:
            new=re.sub(r'\[(\d{1,2}(?:[,–-]\d{1,2})*)\]',rew,x.text)
            if new!=x.text: x.text=new; n+=1
print('supplementary bracket runs rewritten:',n)
paras={orig_of[i]:kids[i] for i in orig_of}
for i in sorted(orig_of,reverse=True): body.remove(kids[i])
inv={new:old for old,new in remap.items()}
for new in range(1,38): body.insert(first_ref+new-1, paras[inv[new]])
doc.save(F); print('reordered 1..37 and saved')
