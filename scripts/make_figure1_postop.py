#!/usr/bin/env python
"""Figure 1 | Analytic flow, postoperative-only frame.

Drawn deterministically with matplotlib: every number comes from the PARAMS block
below, which is populated from computed outputs, and every box and label position
is set explicitly so nothing overlaps. No generative image model is involved, so
the figure carries no AI-generated image content.

Fixes requested 2026-10-21:
  - counts corrected (the previous figure showed 204/18 and 193/15 against text
    values of 203/17 and 192/14; all of these are superseded here anyway)
  - a box added for the 6 of 26 chart-confirmed SSI patients outside the cohort
  - the two overlapping labels separated by explicit vertical placement
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os, json

HERE=os.path.dirname(os.path.abspath(__file__))
P=json.load(open(os.path.join(HERE,"out","postop_only","derived","figure1_params.json")))

FS=9.0
def box(ax,x,y,w,h,lines,face="white",edge="#333333",lw=1.1,fs=FS,style="round,pad=0.012"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=style,linewidth=lw,
                                edgecolor=edge,facecolor=face,mutation_aspect=1))
    ax.text(x+w/2,y+h/2,"\n".join(lines),ha="center",va="center",fontsize=fs,linespacing=1.45)

def arrow(ax,x0,y0,x1,y1,lw=1.1):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=11,
                                 linewidth=lw,color="#333333",shrinkA=0,shrinkB=0))

fig,ax=plt.subplots(figsize=(7.2,9.0))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

CX=0.50          # main column centre
MW=0.46          # main box width
SX=0.735         # side column x (exclusions)
SW=0.255

# 1. chart-confirmed SSI
box(ax,CX-MW/2,0.905,MW,0.072,
    [f"Chart-confirmed SSI across the enrolled cohort",f"{P['chart_ssi']} patients"],face="#eef2f7")
# exclusion: 6 of 26
box(ax,SX,0.800,SW,0.082,
    [f"{P['ssi_no_gate_photo']} of {P['chart_ssi']} contributed no",
     "photograph passing the","quality gate (excluded)"],face="#f7eeee",edge="#8c5050")
arrow(ax,CX,0.905,CX,0.862)
arrow(ax,CX+0.075,0.841,SX,0.841)

# 2. locked cohort
box(ax,CX-MW/2,0.762,MW,0.079,
    [f"Locked quality-gated cohort",
     f"{P['locked_pts']} patients · {P['locked_photos']:,} photographs · {P['locked_ssi']} SSI"],face="#eef2f7")
arrow(ax,CX,0.762,CX,0.700)

# exclusion: intraoperative + undated
box(ax,SX,0.628,SW,0.096,
    [f"Excluded {P['intraop_photos']} intraoperative",
     f"({P['pre_closure']} pre-closure,",
     f"{P['post_closure']} post-closure) and",
     f"{P['undated']} undated photographs"],face="#f7eeee",edge="#8c5050")
arrow(ax,CX+0.075,0.676,SX,0.676)

# 3. postoperative analytic frame
box(ax,CX-MW/2,0.600,MW,0.079,
    [f"Postoperative analytic frame",
     f"{P['postop_pts']} patients · {P['postop_photos']:,} photographs · {P['postop_ssi']} SSI"],face="#e8f0e8")
arrow(ax,CX,0.600,CX,0.545)

# fan-out rail
ax.plot([0.175,0.825],[0.545,0.545],color="#333333",lw=1.1)
for x in (0.175,0.50,0.825): arrow(ax,x,0.545,x,0.500)

# 4. three downstream frames
box(ax,0.175-0.155,0.392,0.310,0.108,
    [f"POD 7 landmark risk set","(primary analysis)",
     f"{P['pod7_pts']} patients","" ,f"{P['pod7_events']} subsequent SSI events"],face="white")
box(ax,0.50-0.155,0.392,0.310,0.108,
    [f"Mask-eligible frame","(spatial controls)",
     f"{P['mask_pts']} patients",f"{P['mask_photos']:,} photographs",f"{P['mask_events']} SSI events"],face="white")
box(ax,0.825-0.155,0.392,0.310,0.108,
    [f"Diagnosis-timing frame","(dated outcomes required)",
     f"{P['dated_pts']} patients","",f"{P['dated_events']} dated SSI events"],face="white")

# note block
ax.text(0.5,0.330,
    "Event counts differ by analysis because each frame imposes a different requirement:\n"
    "the POD 7 landmark excludes patients already diagnosed at the landmark and those\n"
    "without a recorded diagnosis day; the mask-eligible frame requires a usable incision\n"
    "annotation; the diagnosis-timing frame requires a dated outcome.",
    ha="center",va="top",fontsize=FS-0.7,linespacing=1.5,color="#222222")

ax.text(0.5,0.245,
    "Attrition at the quality gate is outcome-dependent: a wound that is packed, dressed or\n"
    "managed with a vacuum dressing is both harder to photograph and more likely to be\n"
    "infected, so every estimate here is conditioned on surviving that filter.",
    ha="center",va="top",fontsize=FS-0.7,linespacing=1.5,style="italic",color="#444444")

out=os.path.join(HERE,"out","postop_only","derived")
fig.savefig(os.path.join(out,"figure1_flow.png"),dpi=400,bbox_inches="tight",facecolor="white")
fig.savefig(os.path.join(out,"figure1_flow.pdf"),bbox_inches="tight",facecolor="white")
print("figure1_flow.png / .pdf written")
