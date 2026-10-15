"""Derive incision region (chest / abdomen) from the free-text procedure description.
Stems use \\b(stem)\\w* -- a trailing \\b after a truncated stem never matches."""
import re
CHEST=[r"cabg",r"\bavr\b",r"\bmvr\b",r"\btvr\b",r"tavr",r"savr",r"aortic valve",r"mitral",r"tricuspid",
  r"sternotom\w*",r"sternal",r"sternum",r"rib plating",r"ribs?\b",r"lobectom\w*",r"\bvats\b",
  r"thoracotom\w*",r"thoracoscop\w*",r"thoracic",r"\blung\w*",r"pericard\w*",r"ascending aort\w*",
  r"arch repair",r"bentall",r"\blvad\b",r"heart transplant",r"myectom\w*",r"epicardial",r"chest wall",
  r"\bchest\b",r"coronary",r"valve replacement",r"aortic root",r"mediastin\w*"]
ABD=[r"exlap",r"ex lap",r"\bdx lap\b",r"laparotom\w*",r"laparoscop\w*",r"\blap\b",r"hartmann",
  r"colectom\w*",r"colostom\w*",r"ileostom\w*",r"\bsbr\b",r"small bowel",r"appendect\w*",r"\bappy\b",
  r"\bappe\b",r"cholecystect\w*",r"cholectom\w*",r"\bchole\b",r"hernia",r"whipple",r"gastrect\w*",
  r"gastrostom\w*",r"g-?tube",r"duoden\w*",r"pancreat\w*",r"splenect\w*",r"adhesi\w*",r"diverticul\w*",
  r"anastomos\w*",r"ostomy",r"abdom\w*",r"peritone\w*",r"\begs\b",r"bowel",r"rect\w*",r"sigmoid\w*",
  r"ileocec\w*",r"\bsbo\b",r"omental",r"\bileum\b",r"jejun\w*",r"\bmeckel",r"\btah-?bso\b",r"\bhysterect\w*",
  r"inguinal",r"umbilical",r"ventral",r"incisional hernia",r"\bperf\w*",r"\baaa\b",r"abdominal aort\w*"]
_C=re.compile("|".join(CHEST)); _A=re.compile("|".join(ABD))
def region(x):
    t=str(x).strip().lower()
    if t in ("","nan","none"): return None
    c,a=bool(_C.search(t)),bool(_A.search(t))
    if c and not a: return "Chest"
    if a and not c: return "Abdomen"
    if c and a:     return "Both regions documented"
    return "Other or unclassifiable"
