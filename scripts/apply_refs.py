# -*- coding: utf-8 -*-
import sys, re, xml.etree.ElementTree as ET
sys.path.insert(0,'.')
from dxml import Doc, W
F='Smartphone Images Sham paper - POSTOP (rev 2026-10-23).docx'
NEW=[  # 21..37, every one verified against its source this session
 'Bucher BT, Shi J, Ferraro JP, et al. Portable automated surveillance of surgical site infections using natural language processing: development and validation. Ann Surg. 2020;272:629–636. doi:10.1097/SLA.0000000000004133',
 'Hooker S, Erhan D, Kindermans P-J, Kim B. A benchmark for interpretability methods in deep neural networks. Adv Neural Inf Process Syst. 2019;32. arXiv:1806.10758',
 'Norman G, Shi C, Goh EL, et al. Negative pressure wound therapy for surgical wounds healing by primary closure. Cochrane Database Syst Rev. 2022;4:CD009261. doi:10.1002/14651858.CD009261.pub7',
 'Bluebelle Study Group. Validation of the Bluebelle Wound Healing Questionnaire for assessment of surgical-site infection in closed primary wounds after hospital discharge. Br J Surg. 2019;106:226–235. doi:10.1002/bjs.11008',
 'von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. Lancet. 2007;370:1453–1457. doi:10.1016/S0140-6736(07)61602-X',
 'He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016:770–778. doi:10.1109/CVPR.2016.90',
 'Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional neural networks. In: Proceedings of the 36th International Conference on Machine Learning (ICML). 2019. arXiv:1905.11946',
 'Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16×16 words: transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR). 2021. arXiv:2010.11929',
 'Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L. ImageNet: a large-scale hierarchical image database. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2009:248–255. doi:10.1109/CVPR.2009.5206848',
 'Loshchilov I, Hutter F. Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR). 2019. arXiv:1711.05101',
 'Oquab M, Darcet T, Moutakanni T, et al. DINOv2: learning robust visual features without supervision. arXiv:2304.07193. 2023.',
 'Zhang S, Xu Y, Usuyama N, et al. BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv:2303.00915. 2023.',
 'Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6:65–70.',
 'Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: Smola AJ, Bartlett P, Schölkopf B, Schuurmans D, eds. Advances in Large Margin Classifiers. MIT Press; 1999:61–74.',
 'Zauner C. Implementation and benchmarking of perceptual image hash functions. Master’s thesis. Upper Austria University of Applied Sciences, Hagenberg; 2010.',
 'Fitzpatrick TB. The validity and practicality of sun-reactive skin types I through VI. Arch Dermatol. 1988;124:869–871. doi:10.1001/archderm.124.6.869',
 'Fleiss JL. Measuring nominal scale agreement among many raters. Psychol Bull. 1971;76:378–382. doi:10.1037/h0031619',
]
# (anchor regex, citation to append right after the anchor)
PLACE=[
 (r'to include models using patient-generated incision photographs','[21]'),
 (r'have documented or modeled some of these factors, particularly image quality','[3,5,6]'),
 (r'without establishing that the model is using infection-specific features','[11,12]'),
 (r'rather than removing relevant anatomy','[22]'),
 (r'is both harder to photograph and more likely to be infected','[23]'),
 (r'are both less photographable and more likely to be infected','[23]'),
 (r'as evidence that a model has learned clinically meaningful pathology','[9,10]'),
 (r'remain plausible sources of information not consistently visible in one RGB photograph','[7,24]'),
 (r'Observational Studies in Epidemiology \(STROBE\)','[25]'),
 (r'ResNet18, EfficientNet-B0, and ViT-B/16 were trained end to end','[26,27,28]'),
 (r'Experiments used ImageNet initialization','[29]'),
 (r'AdamW optimization','[30]'),
 (r'DINOv2 ViT-B/14, a general-purpose self-supervised encoder','[31]'),
 (r'BiomedCLIP ViT-B/16, pretrained on biomedical image-text pairs','[32]'),
 (r'with Holm adjustment within the family of five contrasts','[33]'),
 (r'used cross-fold Platt recalibration','[34]'),
 (r'identified by perceptual hashing','[35]'),
 (r'Fitzpatrick skin type was assessed from standard','[36]'),
 (r'Fleiss κ=0\.105','[37]'),
]
doc=Doc(F); L=[]
for pat,cit in PLACE:
    n=doc.replace(r'('+pat+r')', r'\1 '+cit, 'p')
    L.append(('OK  ' if n else 'MISS')+f' {cit:12s} {pat[:60]}')
# the [15] marker sits on the context-probe RESULTS line, which is old-frame; say so rather than cite
doc.replace(r'On the matched 206-patient masking frame, the retrained full-image model achieved',
            'On the matched 206-patient masking frame of the all-photograph analysis (these probes were not repeated on the postoperative frame), the retrained full-image model achieved','q')
# ---- append reference entries after the last existing reference paragraph ----
body=doc.root.find(W+'body'); kids=list(body)
def ptext(p): return ''.join(t.text or '' for t in p.iter(W+'t'))
ref_idx=[i for i,k in enumerate(kids) if k.tag==W+'p' and re.search(r'doi:|arXiv|Available at|Scand J Stat|MIT Press', ptext(k))]
last=max(ref_idx); tmpl=kids[last]
numbered_literal=bool(re.match(r'\s*\d{1,2}[.)]\s', ptext(kids[ref_idx[0]])))
import copy
for j,entry in enumerate(NEW, start=21):
    p=copy.deepcopy(tmpl)
    for r in list(p.findall(W+'r'))[1:]: p.remove(r)
    ts=list(p.iter(W+'t'))
    if ts: ts[0].text=(f'{j}. ' if numbered_literal else '')+entry; [setattr(x,'text','') for x in ts[1:]]
    body.insert(last+1+(j-21), p)
L.append(f'appended 17 entries after reference paragraph {last} (literal numbering: {numbered_literal})')
doc.save(F); print('\n'.join(L))
