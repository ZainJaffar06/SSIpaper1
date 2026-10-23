import sys,re; sys.path.insert(0,'.')
import pandas as pd, xml.etree.ElementTree as ET
from dxml import Doc, W
D='/Users/zainjaffar/ssi_work/paper1_2026-08-15/out/postop_only/derived/'
F='Smartphone Images Sham paper - POSTOP (rev 2026-10-23).docx'
mc=pd.read_csv(D+'masking_contrasts.csv').set_index('contrast'); s3=lambda x:f'{x:+.3f}'
print('20k contrasts:'); print(mc.to_string())
doc=Doc(F); L=[]
NUM=r'[+−-]?\d\.\d{3}'
sw=mc.loc['shifted - true wound']
n=doc.replace(rf'the mean within-seed difference was {NUM} \(95% CI, {NUM} to {NUM}; P=\d\.\d+; Holm-adjusted P=\d\.\d+ across the six contrasts in this family\)',
  f'the mean within-seed difference was {s3(sw.d)} (95% CI, {s3(sw.lo)} to {s3(sw.hi)}; P={sw.p:.2f}; Holm-adjusted P={sw.holm_p:.2f} across the six contrasts in this family)','a'); L.append(f'headline prose {n}')
tw,tb,bf,sb=mc.loc['true wound - full'],mc.loc['true background - true wound'],mc.loc['true background - full'],mc.loc['shifted - true background']
n=doc.replace(rf'The true wound region scored below the full image \({NUM}; {NUM} to {NUM}; P=\d\.\d+\), true background exceeded the true wound region by {NUM} \({NUM} to {NUM}; P=\d\.\d+\) and matched the full image \({NUM}; {NUM} to {NUM}; P=\d\.\d+\), and the shifted region did not differ from true background \({NUM}; {NUM} to {NUM}; P=\d\.\d+\)',
  f'The true wound region scored below the full image ({s3(tw.d)}; {s3(tw.lo)} to {s3(tw.hi)}; P={tw.p:.2f}), true background exceeded the true wound region by {s3(tb.d)} ({s3(tb.lo)} to {s3(tb.hi)}; P={tb.p:.2f}) and matched the full image ({s3(bf.d)}; {s3(bf.lo)} to {s3(bf.hi)}; P={bf.p:.2f}), and the shifted region did not differ from true background ({s3(sb.d)}; {s3(sb.lo)} to {s3(sb.hi)}; P={sb.p:.2f})','b'); L.append(f'remaining prose {n}')
n=doc.replace(rf'single-shift contrast itself was null \({NUM}; P=\d\.\d+\)',f'single-shift contrast itself was null ({s3(sw.d)}; P={sw.p:.2f})','c'); L.append(f'discussion {n}')
# Table S9 contrast cells (T12): mean-within-seed (CI) and P columns
parents={c:p for p in doc.root.iter() for c in p}
def indel(el):
    c=parents.get(el)
    while c is not None:
        if c.tag==W+'del': return True
        c=parents.get(c)
    return False
tb12=list(doc.root.iter(W+'tbl'))[12]
CMAP={'Shifted region − true wound':'shifted - true wound','True background − true wound':'true background - true wound','Shifted region − true background':'shifted - true background',
      'True background − full':'true background - full','True wound − full':'true wound - full','Complement of shifted region − full':'shifted complement - full'}
k=0
for tr in tb12.iter(W+'tr'):
    cells=[]
    for tc in tr.iter(W+'tc'):
        ts=[x for x in tc.iter(W+'t') if not indel(x)]; cells.append((ts,''.join((x.text or '') for x in ts).strip()))
    if cells and cells[0][1] in CMAP:
        r=mc.loc[CMAP[cells[0][1]]]
        for (ts,_),v in zip(cells[4:6],[f'{s3(r.d)} ({s3(r.lo)} to {s3(r.hi)})',f'{r.p:.3f}']):
            if ts: ts[0].text=v; [setattr(x,'text','') for x in ts[1:]]; k+=1
L.append(f'S9 cells {k}')
# stray 1,886
t=doc.build_index()
for m in re.finditer(r'[^.\n]*1,886[^.\n]*\.',t): print('STRAY 1,886 ->',m.group(0).strip()[:200])
doc.save(F); print(L)
