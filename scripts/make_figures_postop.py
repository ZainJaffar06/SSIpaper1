# -*- coding: utf-8 -*-
"""Figures 2-4 on the postoperative-only frame. Drawn deterministically in matplotlib from
out/postop_only/derived/*.csv; no generative tooling. Layout mirrors the submitted figures.
Palette validated with the dataviz validator (light surface): blue #2a78d6, orange #eb6834,
green #1baf7a; ink #0b0b0b / #52514e. Every value on the figure is read from a CSV."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); DER=os.path.join(HERE,"out","postop_only","derived")
FIG=os.path.join(DER,"figures"); os.makedirs(FIG,exist_ok=True)
BLUE,ORANGE,GREEN="#2a78d6","#eb6834","#1baf7a"; INK,INK2,GRID="#0b0b0b","#52514e","#dedcd8"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
  "axes.edgecolor":INK2,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,"text.color":INK,
  "axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white","figure.facecolor":"white"})
def frame(title, subtitle):
    f=plt.figure(figsize=(12,6.6),dpi=200)
    f.text(0.045,0.935,title,fontsize=20,fontweight="bold",color=INK,va="center")
    f.text(0.045,0.885,subtitle,fontsize=11.5,color=INK2,va="center"); return f
def panel_title(ax,letter,text,x=0.045):
    f=ax.figure; yy=ax.get_position().y1+0.075
    f.text(x,yy,letter,fontsize=14,fontweight="bold",color=INK,va="center")
    f.text(x+0.022,yy,text,fontsize=13.5,fontweight="bold",color=INK,va="center")
def save(f,name):
    f.savefig(os.path.join(FIG,name+".png"),dpi=200); f.savefig(os.path.join(FIG,name+".pdf")); plt.close(f); print("wrote",name,flush=True)
def fmt(x): return f"{x:+.3f}".replace("-","−")

# ---------------- Figure 2 ----------------
def figure2():
    pl=pd.read_csv(os.path.join(DER,"primary_and_landmarks.csv"))
    rows=[("Mean through POD 7","PRIMARY"),("Latest through POD 7","latest"),("Closest to POD 7","closest")]
    sc=pd.read_csv(os.path.join(DER,"pod7_patient_scores.csv")); y=sc.ssi_final.values.astype(int); s=sc.score.values
    n,e=len(sc),int(y.sum()); assert (n,e)==(132,8)
    op=pd.read_csv(os.path.join(DER,"operating_points.csv")); r80=op[op.target_spec==0.8].iloc[0]
    # capture all-but-one events: threshold = lowest positive score (flag rule is score > threshold)
    # table convention: highest threshold that still captures e-1 events, flag rule score >= threshold
    thr=np.sort(s[y==1])[1]; cap=int((s[y==1]>=thr).sum()); assert cap==e-1
    sens2=cap/e; flag2=float((s>=thr).mean()); spec2=float((s[y==0]<thr).mean())
    sens1=float(r80.sensitivity); flag1=float(r80.n_flagged)/n; spec1=float(r80.specificity)
    print(f"Fig2B: ~80% spec -> sens {sens1:.3f}, flagged {flag1:.3f} ({int(r80.n_flagged)}/{n}), spec {spec1:.3f}; "
          f"capture {cap}/{e} -> sens {sens2:.3f}, flagged {flag2:.3f}, spec {spec2:.3f}",flush=True)
    f=frame("Prospective POD 7 performance and alert burden",f"{n} patients, {e} subsequent SSI events; prevalence {100*e/n:.1f}%")
    ax=f.add_axes([0.20,0.25,0.24,0.50]); panel_title(ax,"A","Discrimination across aggregation rules")
    for i,(lab,key) in enumerate(rows):
        r=pl[pl.analysis.str.contains(key,case=False)].iloc[0]; yy=2-i
        ax.plot([r.lo,r.hi],[yy,yy],color=BLUE,lw=2,solid_capstyle="butt"); ax.plot([r.lo,r.lo],[yy-.08,yy+.08],color=BLUE,lw=2); ax.plot([r.hi,r.hi],[yy-.08,yy+.08],color=BLUE,lw=2)
        ax.plot(r.auroc,yy,"o",color=BLUE,ms=9,zorder=3)
        ax.text(1.03,yy+0.09,f"{r.auroc:.3f} ({r.lo:.3f}–{r.hi:.3f})",fontsize=10.5,fontweight="bold",color=INK,va="center")
        ax.text(1.03,yy-0.13,f"AUPRC {r.auprc:.3f}",fontsize=10,color=INK2,va="center")
    ax.set_yticks([2,1,0]); ax.set_yticklabels([r[0] for r in rows],fontsize=11); ax.tick_params(axis="y",length=0)
    ax.set_xlim(0.40,1.00); ax.set_xticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]); ax.set_ylim(-0.6,2.6)
    ax.axvline(0.5,color=INK2,lw=1.2); ax.grid(axis="x",color=GRID,lw=0.8); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Patient-level AUROC (95% CI)",fontsize=11.5,fontweight="bold",labelpad=8)
    ax2=f.add_axes([0.645,0.25,0.31,0.50]); panel_title(ax2,"B","Illustrative operating burden",x=0.565)
    ax2.text(0.0,1.05,"Sensitivity and proportion of patients flagged",transform=ax2.transAxes,fontsize=10.5,color=INK2)
    xs=np.array([0,1]); w=0.26
    b1=ax2.bar(xs-w/2-0.02,[sens1,sens2],w,color=BLUE,label="Sensitivity"); b2=ax2.bar(xs+w/2+0.02,[flag1,flag2],w,color=ORANGE,label="Patients flagged")
    for b in list(b1)+list(b2): ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{100*b.get_height():.0f}%",ha="center",fontsize=10.5,fontweight="bold")
    ax2.set_ylim(0,1.0); ax2.set_yticks([0,.25,.5,.75,1]); ax2.set_yticklabels(["0%","25%","50%","75%","100%"]); ax2.grid(axis="y",color=GRID,lw=0.8); ax2.set_axisbelow(True)
    ax2.set_xticks(xs); ax2.set_xticklabels([f"{100*spec1:.0f}% specificity\n{int(round(sens1*e))}/{e} infections",f"{cap}/{e} captured\n{100*spec2:.0f}% specificity"],fontsize=10.5)
    ax2.tick_params(axis="x",length=0); ax2.spines["left"].set_visible(False); ax2.tick_params(axis="y",length=0)
    ax2.legend(loc="upper center",bbox_to_anchor=(0.5,-0.22),ncol=2,frameon=False,fontsize=10.5)
    f.text(0.045,0.045,"Thresholds were selected in the same out-of-fold cohort and are descriptive, not externally validated rule-out thresholds.",fontsize=9.5,color=INK2)
    save(f,"figure2_pod7_postop")

# ---------------- Figure 3 ----------------
def figure3():
    ar=pd.read_csv(os.path.join(DER,"architectures.csv"))
    cp_p=os.path.join(DER,"context_probes_postop.csv"); dl_p=os.path.join(DER,"seedaware_probe_deltas_postop.csv")
    if not (os.path.exists(cp_p) and os.path.exists(dl_p)): print("figure3: context outputs not ready; skipped"); return
    cp=pd.read_csv(cp_p); dl=pd.read_csv(dl_p)
    seeds=[7,42,20260807]; col={7:BLUE,42:ORANGE,20260807:GREEN}; mk={7:"o",42:"s",20260807:"^"}
    archs=[("efficientnet_b0","EfficientNet-B0"),("resnet18","ResNet18"),("vit_b_16","ViT-B/16")]
    per={a:{int(k):float(v) for k,v in (t.split(":") for t in ar[ar.architecture==a].per_seed.iloc[0].split(" / "))} for a,_ in archs}
    f=frame("Architecture and measured-context robustness","Patient-level AUROC; postoperative cohort (171 patients, 13 SSI events)")
    ax=f.add_axes([0.11,0.22,0.36,0.53]); panel_title(ax,"A","Architecture performance across three training seeds")
    for sd in seeds: ax.plot(range(3),[per[a][sd] for a,_ in archs],marker=mk[sd],color=col[sd],ms=8,lw=2,label=str(sd))
    for i,(a,lab) in enumerate(archs):
        m=float(ar[ar.architecture==a]["mean"].iloc[0]); ax.plot(i,m,"D",color=INK,ms=8,zorder=4); ax.text(i,m+0.009,f"{m:.3f}",ha="center",fontsize=9.5,fontweight="bold")
    ax.plot([],[],"D",color=INK,ms=8,label="mean")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _,l in archs],fontsize=11); ax.set_xlim(-0.35,2.35)
    lo=min(min(v.values()) for v in per.values()); hi=max(max(v.values()) for v in per.values()); ax.set_ylim(np.floor((lo-0.02)*20)/20,np.ceil((hi+0.02)*20)/20)
    ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True); ax.set_ylabel("AUROC",fontsize=11.5,rotation=0,labelpad=24,va="center")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=4,frameon=False,fontsize=10.5)
    ax2=f.add_axes([0.59,0.22,0.28,0.53]); panel_title(ax2,"B","CNN minus context-probe AUROC by seed",x=0.50)
    probes=[("Postoperative day","pod_only"),("Acquisition features","acquisition"),("Prior acquisition behavior","behavior_past_only"),("All context features","acquisition+behavior+pod")]
    cnn={int(r.seed):float(r.cnn_auroc) for r in dl.itertuples()}
    allx=[]
    for i,(lab,key) in enumerate(probes):
        pr=cp[cp.probe==key]; assert len(pr)==1,key
        pa=float(pr.patient_auroc.iloc[0]); yy=3-i; ds=[]
        for sd in seeds:
            d=cnn[sd]-pa; ds.append(d); ax2.plot(d,yy+{7:0.12,42:0.0,20260807:-0.12}[sd],mk[sd],color=col[sd],ms=7,zorder=3); allx.append(d)
        m=float(np.mean(ds)); ax2.plot(m,yy,"D",color=INK,ms=8,zorder=4); ax2.text(1.02,yy,f"mean {fmt(m)}",transform=ax2.get_yaxis_transform(),fontsize=10,color=INK2,va="center")
        print(f"Fig3B {lab}: probe {pa:.3f}; deltas {[round(d,3) for d in ds]}; mean {m:+.3f}",flush=True)
    ax2.axvline(0,color=INK,lw=1.2,ls="--"); ax2.set_yticks([3,2,1,0]); ax2.set_yticklabels([p[0] for p in probes],fontsize=11); ax2.tick_params(axis="y",length=0)
    r=max(abs(min(allx)),abs(max(allx)))+0.03; ax2.set_xlim(-r,r); ax2.set_ylim(-0.6,3.6); ax2.grid(axis="x",color=GRID,lw=0.8); ax2.set_axisbelow(True); ax2.spines["left"].set_visible(False)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: "0" if abs(v)<1e-9 else fmt(v)[:-1] if False else f"{v:+.2f}".replace("-","−")))
    ax2.text(0.0,1.04,"probe higher",transform=ax2.transAxes,fontsize=10,color=INK2); ax2.text(1.0,1.04,"CNN higher",transform=ax2.transAxes,fontsize=10,color=INK2,ha="right")
    ax2.set_xlabel("Difference in patient AUROC",fontsize=11.5,labelpad=8)
    f.text(0.045,0.045,"Diamonds show means across seeds. Panel B uses the matched masking frame (168 patients, 12 SSI events); each probe is one fit, each CNN value one seed.",fontsize=9.5,color=INK2)
    save(f,"figure3_architecture_context_postop")

# ---------------- Figure 4 ----------------
def figure4():
    ma=pd.read_csv(os.path.join(DER,"masking_absolute.csv")); mc=pd.read_csv(os.path.join(DER,"masking_contrasts.csv"))
    ov=pd.read_csv(os.path.join(DER,"single_shift_overlap_audit.csv")).iloc[0]
    conds=[("Full image","Full image",BLUE,True),("True wound-only","True wound-only",ORANGE,True),("True background-only","True background-only",ORANGE,True),
           ("Circularly shifted wound-size region","Shifted wound-size region",BLUE,False),("Complement of shifted region","Complement of shifted region",BLUE,False)]
    f=frame("Spatial masking and sham controls","Patient-level AUROC; mask-eligible postoperative cohort (168 patients, 12 SSI events; 1,296 images)")
    ax=f.add_axes([0.21,0.22,0.26,0.53]); panel_title(ax,"A","Absolute performance by retained region")
    for i,(key,lab,c,filled) in enumerate(conds):
        r=ma[ma.condition==key].iloc[0]; yy=4-i
        for sd in ("seed20260807","seed42","seed7"): ax.plot(float(r[sd]),yy,"|",color=c,ms=14,mew=1.6,alpha=0.75,zorder=2)
        ax.plot(r.seed_mean,yy,"o",color=c if filled else "white",mec=c,mew=2.2,ms=10,zorder=3)
        ax.text(max(float(r[sd]) for sd in ("seed20260807","seed42","seed7"))+0.012,yy,f"{r.seed_mean:.3f}",fontsize=10.5,fontweight="bold",va="center")
    ax.set_yticks([4,3,2,1,0]); ax.set_yticklabels([c[1] for c in conds],fontsize=11); ax.tick_params(axis="y",length=0)
    ax.set_xlim(0.55,0.90); ax.set_xticks([0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]); ax.set_ylim(-0.6,4.6); ax.grid(axis="x",color=GRID,lw=0.8); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Patient AUROC (3-seed mean; ticks are seeds)",fontsize=11.5,labelpad=8)
    ax2=f.add_axes([0.66,0.22,0.27,0.53]); panel_title(ax2,"B","Paired AUROC differences",x=0.50)
    names={"shifted - true wound":"Shifted region − true wound","shifted - true background":"Shifted region − true background","shifted complement - full":"Sham background − full",
           "true background - full":"True background − full","true wound - full":"True wound − full","true background - true wound":"True background − true wound"}
    order=["shifted - true wound","shifted - true background","shifted complement - full","true background - full","true wound - full","true background - true wound"]
    for i,k in enumerate(order):
        r=mc[mc.contrast==k].iloc[0]; yy=len(order)-1-i
        ax2.plot([r.lo,r.hi],[yy,yy],color=BLUE,lw=2); ax2.plot([r.lo,r.lo],[yy-.08,yy+.08],color=BLUE,lw=2); ax2.plot([r.hi,r.hi],[yy-.08,yy+.08],color=BLUE,lw=2)
        ax2.plot(r.d,yy,"o",color=BLUE,ms=9,zorder=3); ax2.text(1.03,yy,fmt(r.d),transform=ax2.get_yaxis_transform(),fontsize=10.5,va="center",color=INK)
    ax2.axvline(0,color=INK,lw=1.2,ls="--"); ax2.set_yticks(range(len(order))); ax2.set_yticklabels([names[k] for k in order][::-1],fontsize=10.5); ax2.tick_params(axis="y",length=0)
    ax2.set_xlim(-0.35,0.35); ax2.set_xticks([-0.3,-0.2,-0.1,0,0.1,0.2,0.3]); ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: "0" if abs(v)<1e-9 else f"{v:+.2f}".replace("-","−")))
    ax2.set_ylim(-0.6,len(order)-0.4); ax2.grid(axis="x",color=GRID,lw=0.8); ax2.set_axisbelow(True); ax2.spines["left"].set_visible(False)
    ax2.text(0.0,1.04,"first condition lower",transform=ax2.transAxes,fontsize=10,color=INK2); ax2.text(1.0,1.04,"first condition higher",transform=ax2.transAxes,fontsize=10,color=INK2,ha="right")
    ax2.set_xlabel("Paired difference in patient AUROC (95% CI)",fontsize=11.5,labelpad=8)
    f.text(0.045,0.075,f"Open circles denote circularly shifted sham regions; {100*ov.frac_le_0_05:.1f}% of images had intersection-over-union ≤0.05 with the true wound mask (maximum {ov.iou_max:.3f}).",fontsize=9.5,color=INK2)
    f.text(0.045,0.045,"Each condition is a mean over three training seeds; intervals in B are paired patient-bootstrap estimates (20,000 resamples). No contrast is significant after Holm correction.",fontsize=9.5,color=INK2)
    save(f,"figure4_masking_postop")

if __name__=="__main__":
    which=sys.argv[1:] or ["2","3","4"]
    for w in which: {"2":figure2,"3":figure3,"4":figure4}[w]()
