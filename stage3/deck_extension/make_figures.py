"""Figures for the extended (16-slide) deck: equations, model diagrams, hyperparameter and MLP-limit plots.
Run from stage3/deck_extension/ ; numbers come from results/stage3/summary.json."""
import json, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
R=str(Path(__file__).resolve().parents[2])+"/"
S=json.load(open(R+"results/stage3/summary.json"))
t=lambda k:S[k]["test"]
NAVY="#142F4E";BLUE="#2F7FD1";GREEN="#1E8E5A";ORANGE="#C96F1E";GREY="#5F6B7A";LB="#CFE6F8"
def eq(path,tex,size=30,w=6,h=0.9):
    f=plt.figure(figsize=(w,h),dpi=220); f.text(0.5,0.5,tex,ha="center",va="center",fontsize=size,color=NAVY); f.savefig(path,transparent=False,facecolor="white"); plt.close(f)
eq("eq_darcy.png",r"$-\nabla\cdot\left(\kappa\,\nabla u\right)=f$",size=34)
eq("eq_loss.png",r"$\mathcal{L}=\frac{1}{B}\sum_{i=1}^{B}\frac{\|\hat{u}_i-u_i\|_2}{\|u_i\|_2}$",size=26,w=6,h=1.1)

# --- model diagrams
def box(ax,x,y,w,h,txt,fc,ec,fs=10,tc=NAVY,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold" if bold else "normal")
def arrow(ax,x0,x1,y):
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",mutation_scale=12,color=GREY,lw=1.3))
fig,axs=plt.subplots(2,1,figsize=(12.1,4.6),dpi=200)
ax=axs[0]; ax.set_xlim(0,12); ax.set_ylim(0,1.4); ax.axis("off")
ax.text(0,1.28,"MLP baseline  (100.7 M parameters)",fontsize=13,fontweight="bold",color=ORANGE,va="center")
xs=[0.0,2.3,4.6,6.9,9.2]; ws=2.0
labs=["Input\n3 × 64 × 64\n= 12,288 numbers","Hidden layer 1\n4,096 units + GELU","Hidden layer 2\n4,096 units + GELU","Hidden layer 3\n4,096 units + GELU","Output\n4,096 numbers\n= 64 × 64 pressure"]
fcs=["#F3F8FD","#FFF8F0","#FFF8F0","#FFF8F0","#F3F8FD"]
for i,(x,l,fc) in enumerate(zip(xs,labs,fcs)):
    box(ax,x,0.12,ws,0.85,l,fc,ORANGE if 0<i<4 else BLUE,fs=12)
    if i<4: arrow(ax,x+ws+0.02,xs[i+1]-0.02,0.545)
ax=axs[1]; ax.set_xlim(0,12); ax.set_ylim(0,1.4); ax.axis("off")
ax.text(0,1.28,"FNO  (9.5 M parameters)",fontsize=13,fontweight="bold",color=GREEN,va="center")
box(ax,0.0,0.12,1.9,0.85,"Input\npermeability + x, y\n3 channels","#F3F8FD",BLUE,fs=11.5)
box(ax,2.3,0.12,1.9,0.85,"Lift\n3 → 64 channels","#EAF6EF",GREEN,fs=11.5)
box(ax,4.6,0.12,4.2,0.85,"8 × Fourier layer\nFFT, keep 12 × 12 modes, learned weights,\ninverse FFT, 1 × 1 skip, GELU","#EAF6EF",GREEN,fs=11)
box(ax,9.2,0.12,1.4,0.85,"Project\n64 → 128 → 1","#EAF6EF",GREEN,fs=11.5)
box(ax,10.9,0.12,1.1,0.85,"Output\n64 × 64","#F3F8FD",BLUE,fs=11.5)
for a,b in [(1.92,2.28),(4.22,4.58),(8.82,9.18),(10.62,10.88)]: arrow(ax,a,b,0.545)
fig.tight_layout(); fig.savefig("fig_deck_models.png"); plt.close(fig)

# --- hyperparameter panels
fig,axs=plt.subplots(1,4,figsize=(12.1,3.0),dpi=200)
def style(ax,title,xl):
    ax.set_title(title,fontsize=11,color=NAVY,fontweight="bold"); ax.set_xlabel(xl,fontsize=9.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y",alpha=.25); ax.set_ylim(0.015,0.032); ax.tick_params(labelsize=9)
modes=[4,6,8,12,16,20,24,32]; ax=axs[0]
ax.plot(range(len(modes)),[t(f"e2_modes{m}") for m in modes],"-o",color=GREEN,ms=5); ax.set_xticks(range(len(modes))); ax.set_xticklabels(modes)
ax.scatter([3],[t("e2_modes12")],s=130,facecolors="none",edgecolors=ORANGE,lw=2,zorder=5); style(ax,"Fourier modes","modes kept"); ax.set_ylabel("test error",fontsize=10)
ax=axs[1]; ax.bar([0,1,2],[t("e2_batch8"),t("e2_modes12"),t("e2_batch32")],color=[LB,BLUE,LB],edgecolor=BLUE); ax.set_xticks([0,1,2]); ax.set_xticklabels([8,16,32]); style(ax,"Batch size","samples per step")
for i,v in enumerate([t("e2_batch8"),t("e2_modes12"),t("e2_batch32")]): ax.text(i,v+0.0006,f"{v:.4f}",ha="center",fontsize=8.5)
lrs=["0.0005","0.001","0.002"]; ax=axs[2]; ax.bar([0,1,2],[t(f"e2_lr{l}") for l in lrs],color=[LB,BLUE,LB],edgecolor=BLUE); ax.set_xticks([0,1,2]); ax.set_xticklabels(["5e-4","1e-3","2e-3"]); style(ax,"Learning rate","peak learning rate")
for i,l in enumerate(lrs): ax.text(i,t(f"e2_lr{l}")+0.0006,f"{t(f'e2_lr{l}'):.4f}",ha="center",fontsize=8.5)
steps=[5700,10000,20000,40000,80000]; ax=axs[3]
ax.plot(range(5),[t(f"e2_steps{s}") for s in steps],"-o",color=GREEN,ms=5); ax.set_xticks(range(5)); ax.set_xticklabels(["5.7k","10k","20k","40k","80k"]); style(ax,"Training steps","optimiser steps")
fig.tight_layout(); fig.savefig("fig_deck_hparams.png"); plt.close(fig)

# --- MLP saturation
fig,ax=plt.subplots(figsize=(6.4,4.2),dpi=200)
cfg=[("e2_mlp_h1024x2","1024×2"),("e2_mlp_h1024x3","1024×3"),("e2_mlp_h2048x3","2048×3"),("e2_mlp_h2048x5","2048×5"),("e2_mlp_h4096x3","4096×3")]
px=[S[k]["params"]/1e6 for k,_ in cfg]; py=[t(k) for k,_ in cfg]
ax.plot(px,py,"o",color=ORANGE,ms=8,label="MLP (different sizes)")
for (k,l),x,y in zip(cfg,px,py): ax.annotate(l,(x,y),textcoords="offset points",xytext=(0,10),ha="center",fontsize=8.5,color=GREY)
fno=np.mean([t(f"final_fno_s{i}") for i in range(5)])
ax.axhline(fno,color=GREEN,ls="--",lw=2); ax.text(max(px)*0.98,fno+0.0015,f"FNO, 9.5 M params: {fno:.3f}",ha="right",fontsize=10,color=GREEN,fontweight="bold")
ax.set_xlabel("MLP parameters (millions)",fontsize=11); ax.set_ylabel("test error (lower is better)",fontsize=11); ax.set_ylim(0,0.058); ax.set_xlim(10,108)
ax.legend(frameon=False,fontsize=10,loc="lower right")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(alpha=.25); fig.tight_layout(); fig.savefig("fig_deck_mlp_limit.png"); plt.close(fig)
print("ok", round(float(fno),4))

# --- simple solver-vs-FNO figure (slide 12): one solver series, FNO on GPU
P=json.load(open(R+"results/stage3/pareto.json"))
sol=[p for p in P if p["label"].startswith("FV solver, direct")]
fno=[p for p in P if p["label"]=="FNO, GPU"]
fig,ax=plt.subplots(figsize=(7.2,4.25),dpi=200)
ax.plot([p["ms"] for p in sol],[p["err"] for p in sol],"-o",color=GREY,ms=9,lw=1.8,label="FV solver")
for p in sol:
    ax.annotate(p["note"].replace("N=","")+" x "+p["note"].replace("N=",""),(p["ms"],p["err"]),textcoords="offset points",xytext=(8,8),fontsize=11,color=GREY)
ax.plot([p["ms"] for p in fno],[p["err"] for p in fno],"s",color=GREEN,ms=11,label="FNO (on a GPU)")
for p in fno:
    ax.annotate("one map at a time" if p["note"]=="batch 1" else "many maps at once",(p["ms"],p["err"]),textcoords="offset points",xytext=(0,-22),ha="center",fontsize=11,color=GREEN)
ax.set_xscale("log"); ax.set_xlim(0.05,100); ax.set_ylim(0,0.11)
ax.set_xlabel("Time per solution (milliseconds, log scale)",fontsize=12); ax.set_ylabel("Error (lower is better)",fontsize=12)
ax.tick_params(labelsize=11); ax.grid(alpha=.25); ax.legend(fontsize=12,loc="upper right",frameon=False)
for s_ in ("top","right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("fig_deck_solver.png"); plt.close(fig)

# --- simple physics-slide figure (slide 13): how errors are spread, and where the bad ones are
D=np.load(R+"results/stage3/diagnostics_arrays.npz"); rel=D["rel"]; area=D["area_hi"]
fig,axs=plt.subplots(1,2,figsize=(12.1,2.85),dpi=200)
ax=axs[0]
ax.hist(rel,bins=np.linspace(0,0.4,21),color=GREEN,edgecolor="white")
ax.axvline(np.median(rel),color=NAVY,ls="--",lw=1.5); ax.text(0.03,ax.get_ylim()[1]*0.55,f"typical map\n{np.median(rel):.3f}",fontsize=10,color=NAVY)
ax.set_xlabel("Error of one test map",fontsize=11); ax.set_ylabel("Number of maps",fontsize=11)
ax.set_title(f"Most maps are almost exact ({(rel<0.05).mean()*100:.0f}% are under 5% error)",fontsize=11,color=NAVY)
ax.tick_params(labelsize=10)
ax=axs[1]
ax.scatter(area,rel,s=14,color=BLUE,alpha=.8)
ax.set_xlabel("Share of the map that is high permeability",fontsize=11); ax.set_ylabel("Error",fontsize=11)
ax.set_title("The worst maps are almost entirely low permeability",fontsize=11,color=NAVY)
ax.tick_params(labelsize=10)
for a_ in axs:
    for s_ in ("top","right"): a_.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("fig_deck_physics.png"); plt.close(fig)
print("mean/median/under5",rel.mean(),np.median(rel),(rel<0.05).mean())
