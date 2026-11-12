"""Extend the 12-slide base deck to the final 16 slides (objectives, Darcy problem, data/training, models,
hyperparameters, MLP limit, limitations) and renumber. Run make_figures.py first, from this folder.
Output: stage3/PINNacles_Stage3_FinalPresentation.pptx (export the PDF from Keynote/PowerPoint)."""
import re, copy
from pathlib import Path
R=Path(__file__).resolve().parents[2]
src=(R/"stage3/build_stage3_presentation.py").read_text()
head=src.split("# ================================================================ slides")[0]
head=head.replace("HERE = Path(__file__).resolve().parent","HERE = Path('/Users/aryamannsrivastava/Desktop/sem7/ae646/stage3')")
g={"__name__":"hdr","__file__":str(R/"stage3/build_stage3_presentation.py")}
exec(head,g)
from pptx import Presentation
from pptx.util import Inches, Pt
Pk=Presentation(str(Path(__file__).resolve().parent/"base_deck_12.pptx")); g["prs"]=Pk; g["BLANK"]=Pk.slide_layouts[6]
textbox,rounded,base_slide,stat_card,callout,picture,bullets=[g[k] for k in "textbox rounded base_slide stat_card callout picture bullets".split()]
NAVY,BLUE,GREEN,ORANGE,GREY,PALE,LINE,WHITE=[g[k] for k in "NAVY BLUE GREEN ORANGE GREY PALE LINE WHITE".split()]
from pptx.dml.color import RGBColor
CREAM=RGBColor(0xFF,0xF8,0xF0)
FIG=Path(__file__).resolve().parent; SF=R/"stage3/figures"
N=g["N"]
n0=len(Pk.slides)  # 12
# ---------- new slides (appended; reordered later)
# A. Objectives and phases
s=base_slide("Objectives and Project Phases","What we set out to do, and what each stage delivered",0); A=s
cols=[("Stage 1","Problem and plan",BLUE,["Darcy flow as an operator-learning task","Real PDEBench data, FNO vs MLP","Evaluation plan and metrics"]),
      ("Stage 2","First real results",GREEN,["Trained FNO and MLP on real data","FNO error 0.052, MLP 0.08","FNO about 1.6x more accurate"]),
      ("Stage 3","Make it better, and check it",ORANGE,["Training, tuning, data, grid and noise studies","Fair comparison with a classical solver","Physics checks and hard-map weighting"])]
for i,(h,sub,c,pts) in enumerate(cols):
    x=0.6+i*4.1
    rounded(s,x,1.6,3.9,3.1,fill=WHITE,line=LINE)
    bar=s.shapes.add_shape(1,Inches(x),Inches(1.6),Inches(3.9),Inches(0.08)); g["rgb_fill"](bar,c); bar.line.fill.background()
    textbox(s,x+0.2,1.78,3.5,0.45,h,size=22,bold=True,color=c)
    textbox(s,x+0.2,2.25,3.5,0.35,sub,size=16,bold=True,color=NAVY)
    textbox(s,x+0.2,2.75,3.5,1.9,["• "+p for p in pts],size=15,color=NAVY)
rounded(s,0.6,4.95,12.1,0.95)
textbox(s,0.8,5.0,11.7,0.4,"Objectives",size=15,bold=True,color=NAVY)
textbox(s,0.8,5.38,11.7,0.5,"1. Compare FNO and MLP fairly      2. Find what really drives accuracy      3. Test against a classical solver and the physics",size=15,color=GREY)
callout(s,"Question for Stage 3: is the remaining error about the model, or about how we trained it?",y=6.1,h=0.7)

# B. Darcy problem
s=base_slide("The Problem: Darcy Flow","Predict how pressure spreads through a porous material",0); B=s
rounded(s,0.6,1.55,5.9,4.35,fill=WHITE,line=LINE)
textbox(s,0.8,1.65,5.5,0.4,"Darcy's equation",size=16,bold=True,color=BLUE)
s.shapes.add_picture(str(FIG/"eq_darcy.png"),Inches(0.85),Inches(2.05),width=Inches(5.4))
textbox(s,0.8,3.2,5.5,2.7,["• Input: permeability map κ (two values, 0.1 and 1.0)",
   "• Output: pressure field u",
   "• Source: constant, f = 1",
   "• Boundary condition: u = 0 on all four edges",
   "• Goal: learn the map κ → u from 900 examples"],size=16,color=NAVY)
picture(s,SF/"fig_sample_row.png",6.75,1.55,5.95,1.75)
textbox(s,6.75,3.32,5.95,0.3,"Permeability, true pressure, FNO prediction, error",size=11,color=GREY,italic=True,align=g["PP_ALIGN"].CENTER)
rounded(s,6.75,3.7,5.95,2.2,fill=CREAM,line=ORANGE)
textbox(s,6.95,3.76,5.5,0.4,"How physics is involved",size=16,bold=True,color=ORANGE)
textbox(s,6.95,4.18,5.55,1.7,["• The PDE generated every training sample",
   "• Its symmetry (flips, rotations) gives free extra data",
   "• We check the predictions against the equation afterwards"],size=14,color=NAVY)
callout(s,"A trained model answers a new permeability map almost instantly",y=6.1,h=0.7)

# C. Data and training setup
s=base_slide("Data and Training Setup","What the models see and how they are trained",0); C=s
stat_card(s,0.6,1.55,3.9,1.15,"10,000","real PDEBench simulations, 128 x 128",vsize=28)
stat_card(s,4.7,1.55,3.9,1.15,"900 / 100 / 200","train / validation / test, at 64 x 64",vsize=28)
stat_card(s,8.8,1.55,3.9,1.15,"5 seeds","for the final models, split fixed (seed 42)",vsize=28)
rounded(s,0.6,2.95,5.95,2.95,fill=WHITE,line=LINE)
textbox(s,0.8,3.02,5.5,0.4,"The data",size=16,bold=True,color=BLUE)
textbox(s,0.8,3.45,5.6,2.4,["• Input: permeability map plus x and y coordinates",
  "• Output: pressure field, same 64 x 64 grid",
  "• Boundary condition is fixed by the PDE: u = 0 at the edges",
  "• Inputs and outputs standardised with training-set statistics",
  "• Error is always reported in physical units"],size=15,color=NAVY)
rounded(s,6.75,2.95,5.95,2.95,fill=WHITE,line=LINE)
textbox(s,6.95,3.02,5.5,0.4,"The training",size=16,bold=True,color=GREEN)
textbox(s,6.95,3.45,5.6,2.4,["• Loss: relative L2 error per sample (slide 6)",
  "• Optimiser AdamW, learning rate 1e-3, weight decay 1e-4",
  "• Batch size 16, one smooth cosine learning-rate decay",
  "• Flip and rotate augmentation (symmetries of the PDE)",
  "• Studies: 5,700 to 20,000 steps; final models: 60,000"],size=15,color=NAVY)
callout(s,"Same data, same loss and same budget for both models: only the network differs",y=6.1,h=0.7)

# D. Models
s=base_slide("The Two Models","Same inputs and outputs, different way of looking at space",0); D=s
picture(s,FIG/"fig_deck_models.png",0.6,1.5,12.1,4.5)
callout(s,"The MLP treats the map as a list of numbers; the FNO shares one operator over the whole grid",y=6.15,h=0.65)

# E. Hyperparameters
s=base_slide("Why These Hyperparameters?","Each setting tested one at a time (one seed each, noise about 0.001); most barely matter",0); E=s
picture(s,FIG/"fig_deck_hparams.png",0.6,1.5,12.1,3.05)
items=[("12 modes","Error is flat from 8 to 32 modes. 32 modes cost 7x the parameters and run 3.6x slower, for no gain.",BLUE),
       ("Batch size 16","8, 16 and 32 give 0.0222, 0.0219, 0.0221: no real change. Kept from Stage 2.",BLUE),
       ("Learning rate 1e-3","0.0005 to 0.002 stay within 0.001 of each other. A standard, safe middle value.",BLUE),
       ("60,000 steps","More steps help slowly (0.0247 at 5.7k, 0.0205 at 80k). 60k fits our compute budget.",BLUE)]
for i,(h,b,c) in enumerate(items):
    x=0.6+i*3.05
    rounded(s,x,4.7,2.95,1.3,fill=WHITE,line=LINE)
    textbox(s,x+0.1,4.74,2.75,0.32,h,size=14,bold=True,color=c)
    textbox(s,x+0.1,5.06,2.75,0.95,b,size=11,color=GREY)
callout(s,"Flat curves mean a robust result: it does not hinge on a lucky setting",y=6.2,h=0.65)

# F. MLP saturation
s=base_slide("Why the MLP Cannot Do Better","It levels off at about twice the FNO error, whatever we change",0); F=s
picture(s,FIG/"fig_deck_mlp_limit.png",0.6,1.5,6.3,4.4)
bullets(s,7.2,1.65,5.5,[("10x more parameters","18 M to 100 M: error stays between 0.040 and 0.046"),
   ("More training","20,000 to 60,000 steps: no gain (0.040 to 0.042)"),
   ("More data","Helps up to about 5,000 samples, then flat near 0.035")],size=15,gap=0.95,num_color=ORANGE)
rounded(s,7.2,4.55,5.5,1.35,fill=CREAM,line=ORANGE)
textbox(s,7.35,4.6,5.2,0.35,"Likely reason (our interpretation)",size=13,bold=True,color=ORANGE)
textbox(s,7.35,4.93,5.25,0.95,"The MLP flattens the grid, so it has no notion of neighbours and must relearn the same local physics at every pixel. The FNO reuses one operator everywhere.",size=12,color=NAVY)
callout(s,"The FNO with only 400 training samples (0.026) already beats an MLP trained on 9,700 (0.036)",y=6.1,h=0.7)

# G. Summary and limitations
s=base_slide("Summary and Limitations","What we learned, and what we do not claim",0); G=s
rounded(s,0.6,1.55,4.5,4.4)
textbox(s,0.8,1.62,4.1,0.4,"What we learned",size=18,bold=True,color=BLUE)
bullets(s,0.8,2.2,4.1,[("Training mattered most","Error halved with a better recipe"),
   ("FNO beats a tuned MLP","2x more accurate, 10x fewer parameters"),
   ("Strong against a solver","More accurate at the same grid size"),
   ("Robust to label noise","Still ahead of the MLP with 10% noise")],size=15,gap=0.98)
rounded(s,5.3,1.55,7.4,4.4,fill=CREAM,line=ORANGE)
textbox(s,5.5,1.62,7.0,0.4,"Limitations",size=18,bold=True,color=ORANGE)
bullets(s,5.5,2.2,7.0,[("One PDE, one dataset","Darcy with f = 1 and two permeability values; trends may not transfer"),
   ("Agreement with the data, not the exact PDE","The FNO learns this generator's quirks, e.g. its boundary offset"),
   ("Tied to the training grid","Other grid sizes are 2 to 3x worse; not resolution-free here"),
   ("Speed depends on hardware","The gain is on a GPU in batches; the Stage 2 speed-up claim was withdrawn"),
   ("Hard cases remain","Nearly uniform maps: wrong height (36% error); adaptive weights did not fix it")],size=15,gap=0.74,num_color=ORANGE)
callout(s,"Better training beat a bigger model, and the FNO stayed accurate and fast",y=6.15,h=0.65)

# ---------- edit existing slides
slides=list(Pk.slides)
old={i+1:sl for i,sl in enumerate(slides[:n0])}
# recipe slide (old 4): add loss formula card
r=old[4]
rounded(r,8.3,4.55,4.4,1.35,fill=WHITE,line=LINE)
textbox(r,8.4,4.58,4.2,0.32,"Loss: relative L2 per sample (i = map, j = pixel)",size=10.5,bold=True,color=NAVY)
r.shapes.add_picture(str(FIG/"eq_loss.png"),Inches(8.55),Inches(4.9),width=Inches(3.9))
# data/res slide (old 7): explain resolution error
for sh in old[7].shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Works, but about 2 to 3x worse"):
        run=sh.text_frame.paragraphs[0].runs[0]
        run.text="2 to 3x worse on a new grid: edges and boundary are sampled differently"; run.font.size=Pt(12)
# solver slide (old 8): lead with the GPU speed result; CPU timing only as a small footnote
RGBc=g["RGBColor"] if "RGBColor" in g else None
for sh in old[8].shapes:
    if not sh.has_text_frame: continue
    t=sh.text_frame.text
    runs=sh.text_frame.paragraphs[0].runs
    if t=="Not faster":
        runs[0].text="0.2 ms"; runs[0].font.color.rgb=GREEN
    elif t=="on a single CPU core":
        runs[0].text="per solution, batched on a GPU"
    elif t.startswith("Best when used on GPUs"):
        runs[0].text="Best used on GPUs and in batches, whenever many solutions are needed"
textbox(old[8],8.1,5.62,4.6,0.4,"Timing note: on one CPU core the solver is faster (7 ms vs 37 ms); see report, Section 7.",size=9.5,color=GREY,italic=True)

# ---------- slide 12 (solver): one solver series, simple graph, plain wording
def drop_pictures(sl):
    for sh in list(sl.shapes):
        if sh.shape_type==13: sh._element.getparent().remove(sh._element)
def set_text(sl,old_t,new_t):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text==old_t:
            sh.text_frame.paragraphs[0].runs[0].text=new_t; return
    raise SystemExit("not found: "+old_t)
drop_pictures(old[8]); picture(old[8],FIG/"fig_deck_solver.png",0.6,1.55,7.2,4.25,card=False)
set_text(old[8],"Compared with a finite-volume solver tuned to the same data","Neural network versus the classical calculation")
set_text(old[8],"FNO vs solver error at 64 x 64","error: FNO vs solver (64 x 64 grid)")
set_text(old[8],"faster than the solver on a GPU","faster than the solver, on a GPU")
# ---------- slide 13 (physics): plain labels, simple graphs
drop_pictures(old[9]); picture(old[9],FIG/"fig_deck_physics.png",0.6,3.05,12.1,2.85,card=False)
set_text(old[9],"Checks beyond a single error number","Do the predictions behave like real pressure?")
set_text(old[9],"boundary level (truth 0.1157)".replace("0.1157","0.1159") if False else "boundary level (truth 0.1159)","edge pressure level (real data 0.1159)")
set_text(old[9],"equation residual (truth)","equation mismatch (real data 4.01)")
set_text(old[9],"Largest errors: nearly uniform permeability with large pressure, where the amplitude is hard to infer","Worst cases: almost uniform maps, where the pressure height is hard to guess")

# ---------- slide 16: remove the role tags under the names
TAGS={"Lead Developer","Data & Baseline Verification","Architecture & Ablation Review","Scaling & Diagnostics"}
for sh in list(old[12].shapes):
    if sh.has_text_frame and sh.text_frame.text.strip() in TAGS: sh._element.getparent().remove(sh._element)

# ---------- slide 13: drop the equation-mismatch card, respread the other three
from pptx.util import Emu
cards={0:[],1:[],2:[],3:[]}
for sh in list(old[9].shapes):
    x=sh.left/914400; y=sh.top/914400
    if 1.5<y<2.95 and sh.width/914400<3.2:
        cards[0 if x<3.5 else 1 if x<6.5 else 2 if x<9.6 else 3].append(sh)
for sh in cards[3]: sh._element.getparent().remove(sh._element)
for i in range(3):
    xn=0.6+i*4.1
    for sh in cards[i]:
        isl=sh.has_text_frame and sh.text_frame.text.strip()!="" and sh.top/914400>2.0
        sh.left=Inches(xn+(0.1 if isl else 0)); sh.width=Inches(3.9-(0.2 if isl else 0))

# ---------- new slides from the post-presentation studies (numbers read from results/stage3_extra/summary.json)
import json
X=json.load(open(R/"results/stage3_extra/summary.json"))
nf=X["noise"]["fno"]; nm=X["noise"]["mlp"]; ad=X["adaptive"]; wt=X["weights"]
s=base_slide("Finding 4: Noisy Labels","Training pressure maps made noisy (up to 10%), testing on clean maps",0); N1=s
picture(s,R/"results/stage3_extra/fig_noise.png",0.6,1.55,6.7,4.35)
stat_card(s,7.6,1.6,5.1,1.3,f"{nf['0']['mean']:.3f} to {nf['0.1']['mean']:.3f}","FNO error on clean maps, from no noise to 10% noise",color=BLUE,vsize=26)
stat_card(s,7.6,3.0,5.1,1.3,f"{nf['0.1']['mean']:.3f} vs {nm['0.1']['mean']:.3f}",f"FNO vs MLP at 10% noise (gap {nm['0']['mean']/nf['0']['mean']:.1f}x to {nm['0.1']['mean']/nf['0.1']['mean']:.1f}x)",color=GREEN,vsize=26)
stat_card(s,7.6,4.4,5.1,1.3,f"10% to {nf['0.1']['mean']*100:.1f}%","labels are 10% off; the FNO is only 3.6% off the clean truth",color=GREEN,vsize=26)
textbox(s,7.6,5.72,5.1,0.3,"Gaussian noise, sized as a % of each map; 3 seeds, 5,700 steps, full recipe.",size=9.5,color=GREY,italic=True)
callout(s,"Label noise hurts, but the FNO stays ahead of the MLP at every noise level",y=6.1,h=0.7)
s=base_slide("Finding 5: Self-Adjusting Weights","Per-sample loss weights that adjust themselves from each map's own error",0); N2=s
stat_card(s,0.6,1.55,3.9,1.3,f"{wt['alpha1']['hard']:.1f}x",f"weight on the 27 hard training maps (others {wt['alpha1']['other']:.2f})",color=BLUE,vsize=28)
stat_card(s,4.7,1.55,3.9,1.3,f"{ad['plain']['hard']['mean']:.3f} vs {ad['alpha1']['hard']['mean']:.3f}","error on the 8 hard test maps: plain loss vs adaptive",color=ORANGE,vsize=28)
stat_card(s,8.8,1.55,3.9,1.3,f"{ad['plain']['all']['mean']:.4f} to {ad['alpha1']['all']['mean']:.4f}","overall error got slightly worse with adaptive weights",color=ORANGE,vsize=26)
picture(s,R/"results/stage3_extra/fig_adaptive_slide.png",0.6,3.05,12.1,2.85,card=False)
callout(s,"It found the hard maps itself, but weighting did not fix them: only 27 of 900 are like this",y=6.05,h=0.7)
# ---------- reorder
final=[old[1],A,B,C,D,old[4],old[5],E,old[6],F,old[7],N1,N2,old[8],old[9],old[10],G,old[12]]
lst=Pk.slides._sldIdLst; ids=list(lst)
by={sl.slide_id:el for sl,el in zip(Pk.slides,ids)}
new_order=[by[sl.slide_id] for sl in final]
drop=[el for el in ids if el not in new_order]
for el in ids: lst.remove(el)
for el in new_order: lst.append(el)
for el in drop: Pk.part.drop_rel(el.rId)
# ---------- renumber
total=len(final)
for i,sl in enumerate(Pk.slides,1):
    for sh in sl.shapes:
        if sh.has_text_frame and re.fullmatch(r"\d+ / \d+",sh.text_frame.text.strip()):
            sh.text_frame.paragraphs[0].runs[0].text=f"{i} / {total}"
Pk.save(str(R/"stage3/PINNacles_Stage3_FinalPresentation.pptx")); print("slides",len(Pk.slides))
