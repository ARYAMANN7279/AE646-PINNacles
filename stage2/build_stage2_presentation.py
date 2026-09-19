"""
Builds the Stage 2 (interim) presentation: PINNacles_Stage2_Presentation.pptx

Scope: dataset pipeline, EDA, preprocessing + baseline ablations, MLP vs FNO
preliminary results, issues, and the Stage 3 plan. Figures come from
stage2/figures/ (regenerate with stage2/make_stage2_figures.py).
No code snippets on slides (per the Stage 2 instructions).

Run:  python3 stage2/build_stage2_presentation.py     (needs python-pptx, Pillow)
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

DARK_BLUE   = RGBColor(0x1A, 0x2E, 0x4A)
MID_BLUE    = RGBColor(0x1F, 0x4E, 0x79)
ACCENT_BLUE = RGBColor(0x2E, 0x86, 0xC1)
LIGHT_BLUE  = RGBColor(0xD6, 0xE4, 0xF0)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE   = RGBColor(0xF4, 0xF6, 0xF9)
DARK_GRAY   = RGBColor(0x2C, 0x3E, 0x50)
GREEN       = RGBColor(0x1E, 0x8B, 0x4C)
ORANGE      = RGBColor(0xCA, 0x6F, 0x1E)

W, H = Inches(13.33), Inches(7.5)
CL, CR = Inches(0.45), Inches(12.88)
CW = CR - CL
CT = Inches(1.5)
CB = Inches(7.05)
CAH = CB - CT
TOTAL = 10

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def box(sl, l, t, w, h, fill=None, border=None, bw=Pt(1.2)):
    s = sl.shapes.add_shape(1, l, t, w, h)
    s.line.width = bw
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    else:
        s.fill.background()
    if border:
        s.line.color.rgb = border
    else:
        s.line.fill.background()
    return s


def tx(sl, text, l, t, w, h, sz=Pt(14), bold=False, color=DARK_GRAY, align=PP_ALIGN.LEFT):
    tb = sl.shapes.add_textbox(l, t, w, h)
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = "Arial"; p.font.size = sz; p.font.bold = bold; p.font.color.rgb = color
    return tb


def bullets(sl, items, l, t, w, h, sz=Pt(16), color=DARK_GRAY, gap=Pt(8)):
    tb = sl.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + item
        p.font.name = "Arial"; p.font.size = sz; p.font.color.rgb = color
        p.space_after = gap
    return tb


def pic(sl, name, l, t, maxw, maxh):
    """Insert an image scaled to fit inside (maxw x maxh), centred in that box."""
    path = os.path.join(FIG, name)
    iw, ih = Image.open(path).size
    scale = min(maxw / iw, maxh / ih)
    w, h = int(iw * scale), int(ih * scale)
    sl.shapes.add_picture(path, l + (maxw - w) // 2, t + (maxh - h) // 2, width=w, height=h)


def header(sl, title, sub):
    box(sl, 0, 0, W, H, fill=OFF_WHITE)
    box(sl, 0, 0, W, Inches(1.15), fill=DARK_BLUE)
    box(sl, 0, Inches(1.15), W, Inches(0.08), fill=ACCENT_BLUE)
    tx(sl, title, Inches(0.5), Inches(0.18), W - Inches(1.0), Inches(0.55), sz=Pt(30), bold=True, color=WHITE)
    tx(sl, sub, Inches(0.52), Inches(0.72), W - Inches(1.0), Inches(0.35), sz=Pt(16), color=LIGHT_BLUE)


def footer(sl, num):
    box(sl, 0, H - Inches(0.35), W, Inches(0.35), fill=DARK_BLUE)
    tx(sl, "AE646 Stage 2 Interim Evaluation  |  Team PINNacles", CL, H - Inches(0.32), CW, Inches(0.3),
       sz=Pt(12), color=LIGHT_BLUE)
    tx(sl, f"{num} / {TOTAL}", W - Inches(1.5), H - Inches(0.32), Inches(1.0), Inches(0.3),
       sz=Pt(12), bold=True, color=WHITE, align=PP_ALIGN.RIGHT)


def card(sl, l, t, w, h, title=None, title_color=DARK_BLUE):
    box(sl, l, t, w, h, fill=WHITE, border=LIGHT_BLUE, bw=Pt(1.0))
    if title:
        tx(sl, title, l + Inches(0.25), t + Inches(0.15), w - Inches(0.5), Inches(0.4),
           sz=Pt(19), bold=True, color=title_color)


# ── 1 Title ───────────────────────────────────────────────────────────────────
sl = slide()
box(sl, 0, 0, W, H, fill=DARK_BLUE)
box(sl, 0, H - Inches(0.8), W, Inches(0.8), fill=MID_BLUE)
tx(sl, "AE646 Stage 2 (Interim) Presentation", Inches(1.0), Inches(1.7), W - Inches(2.0), Inches(0.5),
   sz=Pt(20), color=LIGHT_BLUE)
tx(sl, "Fourier Neural Operator for\nParametric Darcy Flow", Inches(1.0), Inches(2.2), W - Inches(2.0),
   Inches(1.5), sz=Pt(44), bold=True, color=WHITE)
tx(sl, "Team: PINNacles (Aryamann Srivastava, Varun Sathaye, Atishay Jain, Vedant S. Tiwari)",
   Inches(1.0), Inches(4.1), W - Inches(2.0), Inches(0.5), sz=Pt(16), color=LIGHT_BLUE)

# ── 2 Problem ─────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "Problem Statement", "Learning the solution operator of the 2D Darcy flow equation")
footer(sl, 2)
card(sl, CL, CT, CW, CAH, "Governing equation and goal")
tx(sl, "-div( κ(x,y) ∇u(x,y) ) = f(x,y)   on (0,1)²,    u = 0 on the boundary,   f = 1",
   CL + Inches(0.3), CT + Inches(0.75), CW - Inches(0.6), Inches(0.45), sz=Pt(20), bold=True, color=MID_BLUE)
bullets(sl, [
    "κ(x,y): piecewise-constant permeability, values in {0.1, 1.0};  u(x,y): steady pressure field.",
    "Goal: learn the operator  G : κ → u  once, then predict for any new κ in milliseconds.",
    "Numerical solvers (FDM / FEM) must re-solve the PDE for every new κ: too slow for optimisation, UQ, control.",
    "Project theme #8: operator learning for parametric PDEs.  Baseline: MLP.  SciML model: Fourier Neural Operator (FNO).",
], CL + Inches(0.3), CT + Inches(1.5), CW - Inches(0.6), Inches(3.6), sz=Pt(18))

# ── 3 Data pipeline + EDA ─────────────────────────────────────────────────────
sl = slide()
header(sl, "Dataset Pipeline and Exploratory Analysis", "Real PDEBench 2D Darcy Flow (β = 1.0)  -  pipeline completed")
footer(sl, 3)
card(sl, CL, CT, Inches(5.2), CAH, "Pipeline")
bullets(sl, [
    "10,000 real samples, 128×128; checksum-verified download",
    "Reproducible subset (seed 42): 1,000 train/val (900 / 100) + 200 test",
    "Every 2nd grid point → 64×64; add (x, y) channels → 64×64×3",
    "Standardise with training-set statistics only",
    "200 test fields also kept at native 128×128",
], CL + Inches(0.2), CT + Inches(0.7), Inches(4.8), Inches(2.7), sz=Pt(15), gap=Pt(6))
tx(sl, "EDA findings", CL + Inches(0.25), CT + Inches(3.25), Inches(4.7), Inches(0.35), sz=Pt(17), bold=True, color=DARK_BLUE)
bullets(sl, [
    "κ exactly bimodal; high-κ area fraction 0.48 ± 0.21",
    "180/200 test fields: a single high-κ blob; 2/200 exactly uniform",
    "Peak pressure varies widely (0.1 to >1.0)",
], CL + Inches(0.2), CT + Inches(3.65), Inches(4.8), Inches(1.9), sz=Pt(14), gap=Pt(4))
card(sl, CL + Inches(5.35), CT, CW - Inches(5.35), CAH)
pic(sl, "fig8_eda_dataset.png", CL + Inches(5.45), CT + Inches(0.1), CW - Inches(5.55), CAH - Inches(0.2))

# ── 4 Preprocessing ablation ──────────────────────────────────────────────────
sl = slide()
header(sl, "Preprocessing Ablation (MLP)", "One choice varied at a time; 3 seeds each; same split and target")
footer(sl, 4)
card(sl, CL, CT, Inches(7.4), CAH)
pic(sl, "fig10_preprocess_ablation.png", CL + Inches(0.1), CT + Inches(0.3), Inches(7.2), CAH - Inches(0.6))
card(sl, CL + Inches(7.55), CT, CW - Inches(7.55), CAH, "Result")
bullets(sl, [
    "Reference (standardised + coordinates): 0.0842",
    "No coordinate channels: 0.0843  (no effect on the MLP)",
    "No input normalisation: 0.0864",
    "No target normalisation: 0.0917  (+9%)",
    "Target normalisation is the choice that matters; the pipeline choices are justified, not assumed.",
], CL + Inches(7.75), CT + Inches(0.7), CW - Inches(7.95), Inches(4.2), sz=Pt(15), gap=Pt(8))

# ── 5 Models + training ───────────────────────────────────────────────────────
sl = slide()
header(sl, "Models and Training", "MLP baseline and Fourier Neural Operator - both implemented and trained")
footer(sl, 5)
half = (CW - Inches(0.3)) / 2
card(sl, CL, CT, half, Inches(3.3), "MLP baseline", ORANGE)
bullets(sl, [
    "Flatten 64×64×3 → 3 hidden layers × 2048 (GELU) → 64×64",
    "42.0 M parameters",
    "No spatial inductive bias; fixed to 64×64 input",
], CL + Inches(0.25), CT + Inches(0.7), half - Inches(0.5), Inches(2.5), sz=Pt(18))
card(sl, CL + half + Inches(0.3), CT, half, Inches(3.3), "Fourier Neural Operator (FNO)", GREEN)
bullets(sl, [
    "Lift 3→64 channels; 4 spectral-convolution blocks",
    "Each block: FFT → keep 12×12 modes → learned complex weights → inverse FFT, plus 1×1 bypass",
    "4.7 M parameters (9× fewer than the MLP)",
], CL + half + Inches(0.55), CT + Inches(0.7), half - Inches(0.5), Inches(2.5), sz=Pt(18))
card(sl, CL, CT + Inches(3.5), CW, Inches(2.0), "Shared training protocol")
bullets(sl, [
    "AdamW (weight decay 1e-4), peak LR 1e-3, cosine schedule; batch 16; 100 epochs; best-validation checkpoint kept",
    "Loss: MSE on standardised pressure.  Reported metric: relative L2 error per sample, in physical units",
], CL + Inches(0.25), CT + Inches(4.05), CW - Inches(0.5), Inches(1.4), sz=Pt(17), gap=Pt(6))

# ── 6 Baseline ablation ───────────────────────────────────────────────────────
sl = slide()
header(sl, "Baseline Development: Depth and Optimiser", "Real re-trained variants on the same data, split and schedule")
footer(sl, 6)
card(sl, CL, CT, Inches(7.4), CAH)
pic(sl, "fig7_mlp_ablation.png", CL + Inches(0.1), CT + Inches(0.3), Inches(7.2), CAH - Inches(0.6))
card(sl, CL + Inches(7.55), CT, CW - Inches(7.55), CAH, "Result")
bullets(sl, [
    "1 layer: 0.1008    2 layers: 0.0796",
    "3 layers (final baseline): 0.0857",
    "3 layers with SGD+momentum: 0.1209",
    "SGD is clearly worse: justifies AdamW.",
    "2 vs 3 layers differ by less than the run-to-run noise (~0.004): treated as equivalent.",
    "Baseline was fixed before the study, so it is a fair, not a weakened, comparison for the FNO.",
], CL + Inches(7.75), CT + Inches(0.7), CW - Inches(7.95), Inches(4.5), sz=Pt(15), gap=Pt(8))

# ── 7 Results ─────────────────────────────────────────────────────────────────
sl = slide()
header(sl, "Preliminary Quantitative Results", "MLP baseline vs FNO - 200 held-out test samples, relative L2 in physical units")
footer(sl, 7)
rows, cols = 3, 7
tbl = sl.shapes.add_table(rows, cols, CL, CT, CW, Inches(1.4)).table
data = [["Model", "Params", "Mean", "Median", "Std", "Min", "Max"],
        ["MLP baseline", "42.0 M", "0.0820", "0.0695", "0.0456", "0.0325", "0.3367"],
        ["FNO", "4.7 M", "0.0521", "0.0398", "0.0451", "0.0155", "0.3467"]]
for r in range(rows):
    for c in range(cols):
        cell = tbl.cell(r, c)
        cell.text = data[r][c]
        para = cell.text_frame.paragraphs[0]
        para.font.name = "Arial"; para.font.size = Pt(16)
        para.font.bold = (r == 0) or (r == 2 and c in (2, 3, 5))
        para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = MID_BLUE if r == 0 else (WHITE if r == 1 else LIGHT_BLUE)
        para.font.color.rgb = WHITE if r == 0 else DARK_GRAY
card(sl, CL, CT + Inches(1.6), Inches(6.6), Inches(3.85))
pic(sl, "fig1_error_histogram.png", CL + Inches(0.1), CT + Inches(1.7), Inches(6.4), Inches(3.65))
card(sl, CL + Inches(6.75), CT + Inches(1.6), CW - Inches(6.75), Inches(3.85), "Observations", GREEN)
bullets(sl, [
    "FNO: 36% lower mean error with 9× fewer parameters",
    "Both error distributions are right-skewed: most samples < 0.1, a small tail up to ~0.35",
    "FNO's gain is in the bulk of the distribution; the worst-case errors of the two models are similar",
], CL + Inches(6.95), CT + Inches(2.3), CW - Inches(7.15), Inches(3.0), sz=Pt(15), gap=Pt(8))

# ── 8 Predictions + failure case ──────────────────────────────────────────────
sl = slide()
header(sl, "FNO Predictions and a Failure Case", "Best, median and worst test sample (rows)  -  κ | target | prediction | absolute error")
footer(sl, 8)
card(sl, CL, CT, Inches(8.2), CAH)
pic(sl, "fig_sample_predictions.png", CL + Inches(0.1), CT + Inches(0.1), Inches(8.0), CAH - Inches(0.2))
card(sl, CL + Inches(8.35), CT, CW - Inches(8.35), CAH, "Reading the figure", ORANGE)
bullets(sl, [
    "Best (sample 17): error 0.016",
    "Median (sample 191): error 0.040",
    "Worst (sample 95): error 0.347",
    "The worst case has exactly uniform κ = 0.1: the FNO over-predicts the pressure amplitude.",
    "Only 2 of 200 test fields are uniform: a rare degenerate input, reported openly rather than excluded.",
], CL + Inches(8.55), CT + Inches(0.7), CW - Inches(8.75), Inches(4.6), sz=Pt(14), gap=Pt(8))

# ── 9 Issues and future work ──────────────────────────────────────────────────
sl = slide()
header(sl, "Issues Faced and Future Work", "What we hit at this stage, and the plan for Stage 3")
footer(sl, 9)
half = (CW - Inches(0.3)) / 2
card(sl, CL, CT, half, CAH, "Issues faced", ORANGE)
bullets(sl, [
    "Metric convention: relative L2 must be computed in physical units, not on standardised fields",
    "Uniform-κ samples: degenerate input; FNO over-predicts amplitude",
    "Compute: 1.3 GB raw file and training moved to the lab GPU workstation",
    "Run-to-run variance (~0.004): early runs were unseeded; now fixed, and only larger gaps are interpreted",
], CL + Inches(0.25), CT + Inches(0.7), half - Inches(0.5), Inches(4.6), sz=Pt(18), gap=Pt(12))
card(sl, CL + half + Inches(0.3), CT, half, CAH, "Future work (Stage 3)", GREEN)
bullets(sl, [
    "FNO scaling study: larger width / modes / depth vs the baseline",
    "Resolution generalisation: evaluate at native 128×128",
    "Computational cost: speed vs the FDM solver; FNO operation profiling",
    "Physical interpretation: error vs κ geometry",
    "Learning-rate schedule comparison; final report, presentation, viva",
], CL + half + Inches(0.55), CT + Inches(0.7), half - Inches(0.5), Inches(4.6), sz=Pt(18), gap=Pt(12))

# ── 10 Summary ────────────────────────────────────────────────────────────────
sl = slide()
box(sl, 0, 0, W, H, fill=DARK_BLUE)
box(sl, 0, Inches(1.12), W, Inches(0.07), fill=ACCENT_BLUE)
tx(sl, "Summary", Inches(0.6), Inches(0.18), W - Inches(1.2), Inches(0.85), sz=Pt(36), bold=True, color=WHITE)
summary = [
    ("Data",     "Real PDEBench pipeline complete: 900 / 100 / 200 split at 64×64; EDA and preprocessing ablation done"),
    ("Baseline", "MLP (42 M) implemented and tuned by depth / optimiser / preprocessing ablations: mean rel. L2 0.082"),
    ("SciML",    "FNO (4.7 M) implemented and trained: mean rel. L2 0.052, 36% lower with 9× fewer parameters"),
    ("Insight",  "Worst FNO case is a degenerate uniform-κ field, not a complex one"),
    ("Next",     "Scaling study, resolution generalisation, cost analysis, physical interpretation"),
]
RH = Inches(0.95)
for j, (label, text) in enumerate(summary):
    RT = Inches(1.5) + j * RH
    box(sl, Inches(0.5), RT, Inches(2.0), RH - Inches(0.12), fill=ACCENT_BLUE)
    tx(sl, label, Inches(0.5), RT + Inches(0.2), Inches(2.0), RH - Inches(0.15), sz=Pt(17), bold=True,
       color=WHITE, align=PP_ALIGN.CENTER)
    tx(sl, text, Inches(2.7), RT + Inches(0.16), W - Inches(3.2), RH - Inches(0.18), sz=Pt(16), color=LIGHT_BLUE)
box(sl, 0, H - Inches(0.62), W, Inches(0.62), fill=MID_BLUE)
tx(sl, "Thank you  -  Questions welcome", Inches(0.5), H - Inches(0.55), W - Inches(1.0), Inches(0.46),
   sz=Pt(19), color=WHITE, align=PP_ALIGN.CENTER)

OUT = os.path.join(HERE, "PINNacles_Stage2_Presentation.pptx")
prs.save(OUT)
print(f"Saved -> {OUT}")
