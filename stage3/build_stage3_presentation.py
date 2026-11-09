"""
Build the Stage 3 final presentation (11 slides, ~10 minutes).

Style: light-blue title slide, white content slides, one idea per slide, big-number cards,
real plots from stage3/figures, short plain-language text. No code, little maths.
All numbers come from results/stage3/*.json or the final report tables.

Run (from repo root):  python3 stage3/build_stage3_presentation.py
Output: stage3/PINNacles_Stage3_FinalPresentation.pptx
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
OUT = HERE / "PINNacles_Stage3_FinalPresentation.pptx"

# palette
NAVY = RGBColor(0x14, 0x2F, 0x4E)
BLUE = RGBColor(0x2F, 0x7F, 0xD1)
LIGHTBLUE = RGBColor(0xCF, 0xE6, 0xF8)
PALE = RGBColor(0xF3, 0xF8, 0xFD)
LINE = RGBColor(0xD3, 0xE3, 0xF3)
GREY = RGBColor(0x5F, 0x6B, 0x7A)
GREEN = RGBColor(0x1E, 0x8E, 0x5A)
ORANGE = RGBColor(0xC9, 0x6F, 0x1E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


# ---------------------------------------------------------------- numbers
def stored_numbers():
    sm = json.load(open(ROOT / "results/stage3/summary.json"))

    def ms(pat):
        v = [x["test"] for n, x in sm.items() if re.fullmatch(pat, n)]
        return float(np.mean(v)), float(np.std(v, ddof=1))

    return {
        "fno_final": ms("final_fno_s."),
        "mlp_final": ms("final_mlp_s."),
        "fno_s2recipe": ms("e1_fno_leg_mse_none_s."),
        "mlp_s2recipe": ms("e1_mlp_leg_mse_none_s."),
        "fno_s3recipe": ms("e1_fno_cos_rel_l2_d4_s."),
        "mlp_s3recipe": ms("e1_mlp_cos_rel_l2_d4_s."),
    }


N = stored_numbers()


def make_story_figure():
    """Four bars: Stage 2 recipe vs final models, FNO and MLP."""
    labels = ["Stage 2", "Final"]
    fno = [N["fno_s2recipe"][0], N["fno_final"][0]]
    mlp = [N["mlp_s2recipe"][0], N["mlp_final"][0]]
    fno_e = [N["fno_s2recipe"][1], N["fno_final"][1]]
    mlp_e = [N["mlp_s2recipe"][1], N["mlp_final"][1]]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=200)
    b1 = ax.bar(x - w / 2, fno, w, yerr=fno_e, color="#1E8E5A", capsize=4, label="FNO")
    b2 = ax.bar(x + w / 2, mlp, w, yerr=mlp_e, color="#C96F1E", capsize=4, label="MLP")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("test error (lower is better)", fontsize=11)
    ax.set_ylim(0, 0.095)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG / "fig_story.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def crop_first_row_of_samples():
    """Top row of fig_samples (best sample): permeability, truth, FNO, error."""
    im = Image.open(FIG / "fig_samples.png").convert("RGB")
    w, h = im.size
    row = im.crop((0, 0, w, int(h * 0.335)))
    path = FIG / "fig_sample_row.png"
    row.save(path)
    return path


# ---------------------------------------------------------------- helpers
def rgb_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def textbox(slide, x, y, w, h, text, size=14, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = FONT
        r.font.color.rgb = color
    return tb


def rounded(slide, x, y, w, h, fill=PALE, line=LINE, line_w=1.0, radius=0.06):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    rgb_fill(s, fill)
    s.line.color.rgb = line
    s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s


def base_slide(title, subtitle, number):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.52), Inches(0.09), Inches(0.82))
    rgb_fill(bar, BLUE)
    bar.line.fill.background()
    textbox(s, 0.82, 0.42, 11.6, 0.6, title, size=30, bold=True, color=NAVY)
    textbox(s, 0.84, 0.98, 11.6, 0.4, subtitle, size=15, color=GREY)
    textbox(s, 0.6, 7.05, 6, 0.3, "AE646  |  Team PINNacles  |  Stage 3", size=10, color=GREY)
    textbox(s, 11.7, 7.05, 1.05, 0.3, f"{number} / 12", size=10, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def stat_card(slide, x, y, w, h, value, label, color=BLUE, vsize=34):
    rounded(slide, x, y, w, h)
    textbox(slide, x, y + 0.12, w, h * 0.55, value, size=vsize, bold=True, color=color,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, x + 0.1, y + h * 0.58, w - 0.2, h * 0.4, label, size=12, color=GREY,
            align=PP_ALIGN.CENTER)


def callout(slide, text, y=6.25, h=0.62, x=0.6, w=12.1):
    s = rounded(slide, x, y, w, h, fill=RGBColor(0xFF, 0xF8, 0xF0), line=ORANGE, line_w=1.5)
    textbox(slide, x + 0.2, y, w - 0.4, h, text, size=16, bold=True, color=ORANGE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def picture(slide, path, x, y, w, h, card=True):
    """Place an image inside a white card, preserving aspect ratio (fit inside w x h)."""
    if card:
        rounded(slide, x, y, w, h, fill=WHITE, line=LINE)
    iw, ih = Image.open(path).size
    pad = 0.12
    bw, bh = w - 2 * pad, h - 2 * pad
    scale = min(bw / iw, bh / ih)
    pw, ph = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                             Inches(pw), Inches(ph))


def bullets(slide, x, y, w, items, size=16, gap=0.78, num_color=BLUE):
    """Short numbered points: bold lead word(s) + one plain sentence."""
    for i, (lead, body) in enumerate(items):
        yy = y + i * gap
        circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(yy + 0.05), Inches(0.42), Inches(0.42))
        rgb_fill(circ, num_color)
        circ.line.fill.background()
        tf = circ.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = str(i + 1)
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.name = FONT
        r.font.color.rgb = WHITE
        textbox(slide, x + 0.55, yy, w - 0.55, 0.3, lead, size=size, bold=True, color=NAVY)
        textbox(slide, x + 0.55, yy + 0.32, w - 0.55, 0.45, body, size=size - 3, color=GREY)


# ================================================================ slides
# 1. Title (light blue)
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = LIGHTBLUE
textbox(s, 0.9, 0.7, 10.5, 0.4, "AE646: Scientific Machine Learning for Fluid Mechanics  |  End-Semester Presentation", size=15, color=NAVY)
textbox(s, 0.9, 1.85, 11.5, 1.45, ["Operator Learning for", "Parametric Darcy Flow"], size=46, bold=True, color=NAVY)
textbox(s, 0.9, 3.45, 11.5, 0.85,
        "Fourier Neural Operator (FNO) vs. MLP Baseline: Optimization, Scaling & Solver Comparison",
        size=21, color=NAVY)
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(4.55), Inches(1.4), Inches(0.07))
rgb_fill(acc, BLUE)
acc.line.fill.background()
textbox(s, 0.9, 4.85, 11.5, 0.45, "Team PINNacles", size=20, bold=True, color=NAVY)
textbox(s, 0.9, 5.3, 11.5, 0.45,
        "Aryamann Srivastava, Varun Sathaye, Atishay Jain, Vedant S. Tiwari", size=17, color=NAVY)
textbox(s, 0.9, 6.7, 10.5, 0.4, "Department of Aerospace Engineering, IIT Kanpur  |  November 2026", size=13, color=GREY)

# 2. The task
sl = base_slide("The Task", "Predict how fluid pressure spreads through a porous material, quickly", 2)
picture(sl, crop_first_row_of_samples(), 0.6, 1.65, 8.4, 2.5)
textbox(sl, 0.6, 4.2, 8.4, 0.35,
        "Left to right: permeability map (input), true pressure, FNO prediction, error",
        size=12, color=GREY, italic=True, align=PP_ALIGN.CENTER)
stat_card(sl, 9.3, 1.65, 3.4, 1.15, "10,000", "real simulations (PDEBench)", vsize=28)
stat_card(sl, 9.3, 2.95, 3.4, 1.15, "900 / 100", "train / validation samples", vsize=28)
stat_card(sl, 9.3, 4.25, 3.4, 1.15, "200", "held-out test samples", vsize=28)
textbox(sl, 0.6, 4.75, 8.4, 0.9,
        ["Two models learn this map from data:",
         "FNO: sees the whole field, works with patterns in space.",
         "MLP: a plain fully connected network, no spatial structure."],
        size=15, color=NAVY)
callout(sl, "Error = relative difference to the true field (smaller is better)", y=6.0)

# 3. Where we started
sl = base_slide("Where We Started", "Stage 2 gave a clear winner, but we suspected we could do much better", 3)
stat_card(sl, 0.6, 1.7, 3.9, 1.7, "0.052", "FNO error in Stage 2", color=GREEN, vsize=44)
stat_card(sl, 4.7, 1.7, 3.9, 1.7, "0.08", "MLP error in Stage 2", color=ORANGE, vsize=44)
stat_card(sl, 8.8, 1.7, 3.9, 1.7, "1.6x", "FNO advantage", color=BLUE, vsize=44)
textbox(sl, 0.6, 3.75, 12.1, 0.4, "How Stage 2 trained both models", size=18, bold=True, color=NAVY)
textbox(sl, 0.6, 4.2, 12.1, 1.4,
        ["Standard squared-error loss",
         "Learning rate that restarted again and again",
         "No data augmentation"],
        size=16, color=GREY)
callout(sl, "Question: was the remaining error about the model, or about how we trained it?", y=5.9, h=0.8)

# 4. Training recipe
sl = base_slide("Finding 1: The Training Recipe", "Changing how we train, not the model, cut the error roughly in half", 4)
picture(sl, FIG / "fig_recipe_ablation.png", 0.6, 1.55, 7.4, 4.3)
bullets(sl, 8.3, 1.7, 4.5, [
    ("Match the loss to the metric", "Train on relative error, the number we report"),
    ("One smooth learning-rate decay", "Instead of repeated restarts"),
    ("Flip and rotate the inputs", "Free extra training data (symmetries)"),
], size=17, gap=1.05)
fa, ma = N["fno_s2recipe"][0], N["mlp_s2recipe"][0]
fb, mb = N["fno_s3recipe"][0], N["mlp_s3recipe"][0]
callout(sl, f"Same model, same budget: FNO error {fa:.3f} to {fb:.3f}, MLP {ma:.3f} to {mb:.3f}", y=6.05)

# 5. Architecture
sl = base_slide("Finding 2: Architecture Matters Little", "Width, depth and Fourier modes were tuned one at a time on validation data", 5)
picture(sl, FIG / "fig_arch_sweeps.png", 0.6, 1.6, 12.1, 3.2)
stat_card(sl, 0.6, 5.0, 3.9, 1.1, "Modes, width", "almost no effect past modest values", color=BLUE, vsize=20)
stat_card(sl, 4.7, 5.0, 3.9, 1.1, "Depth, training length", "small, steady gains", color=BLUE, vsize=20)
stat_card(sl, 8.8, 5.0, 3.9, 1.1, "Final choice", "9.5 M parameters, 8 layers", color=GREEN, vsize=20)
callout(sl, "The training recipe was the big lever; architecture tuning added only a few percent", y=6.25, h=0.6)

# 6. Final models
sl = base_slide("The Final Models", "Both models tuned fairly, validation data only, 5 random seeds each", 6)
picture(sl, make_story_figure(), 0.6, 1.55, 6.2, 4.35)
stat_card(sl, 7.1, 1.65, 2.75, 1.45, f"{N['fno_final'][0]:.4f}", "FNO error, 9.5 M parameters", color=GREEN, vsize=32)
stat_card(sl, 10.0, 1.65, 2.75, 1.45, f"{N['mlp_final'][0]:.4f}", "MLP error, 100.7 M parameters", color=ORANGE, vsize=32)
stat_card(sl, 7.1, 3.3, 2.75, 1.45, "2x", "more accurate", color=BLUE, vsize=36)
stat_card(sl, 10.0, 3.3, 2.75, 1.45, "10x", "fewer parameters", color=BLUE, vsize=36)
textbox(sl, 7.1, 4.95, 5.65, 0.9,
        "The gap between FNO and MLP grew from 1.6x to 2x once both were trained well.",
        size=15, color=NAVY)
callout(sl, "Spatial structure helps: the FNO wins with a tenth of the parameters", y=6.05)

# 7. Data and resolution
sl = base_slide("Finding 3: Data and Resolution", "The FNO needs less data and can be used on other grid sizes, with a cost", 7)
picture(sl, FIG / "fig_data_scaling.png", 0.6, 1.55, 6.1, 3.85)
picture(sl, FIG / "fig_resolution.png", 6.9, 1.55, 3.85, 3.85)
rounded(sl, 10.95, 1.55, 1.75, 3.85)
textbox(sl, 11.05, 1.65, 1.55, 3.6,
        ["How to read the grid", "", "Rows: grid used for training", "", "Columns: grid used for testing", "",
         "Numbers: test error"],
        size=11, color=GREY)
textbox(sl, 0.6, 5.5, 6.1, 0.35, "400 samples", size=16, bold=True, color=NAVY)
textbox(sl, 0.6, 5.83, 6.1, 0.35, "The FNO already beats an MLP trained on 9,700", size=13, color=GREY)
textbox(sl, 6.9, 5.5, 5.8, 0.35, "Other grid sizes", size=16, bold=True, color=NAVY)
textbox(sl, 6.9, 5.83, 5.8, 0.35, "Works, but about 2 to 3x worse than a model trained on that grid", size=13, color=GREY)
callout(sl, "More data helps until about 5,000 samples, then error levels off near 0.02", y=6.35, h=0.55)

# 8. vs classical solver
sl = base_slide("FNO versus a Classical Solver", "Compared with a finite-volume solver tuned to the same data", 8)
picture(sl, FIG / "fig_pareto.png", 0.6, 1.55, 7.2, 4.25)
stat_card(sl, 8.1, 1.65, 4.6, 1.2, "0.020 vs 0.044", "FNO vs solver error at 64 x 64", color=GREEN, vsize=28)
stat_card(sl, 8.1, 3.0, 4.6, 1.2, "5x to 35x", "faster than the solver on a GPU", color=BLUE, vsize=28)
stat_card(sl, 8.1, 4.35, 4.6, 1.2, "Not faster", "on a single CPU core", color=ORANGE, vsize=28)
callout(sl, "Best when used on GPUs and in batches; the solver stays competitive on one core", y=6.05)

# 9. Physics
sl = base_slide("Does It Obey the Physics?", "Checks beyond a single error number", 9)
stat_card(sl, 0.6, 1.55, 2.9, 1.3, "0.1157", "boundary level (truth 0.1159)", color=GREEN, vsize=26)
stat_card(sl, 3.65, 1.55, 2.9, 1.3, "1.7%", "error in average pressure", color=GREEN, vsize=26)
stat_card(sl, 6.7, 1.55, 2.9, 1.3, "0.0015%", "pixels with negative pressure", color=GREEN, vsize=26)
stat_card(sl, 9.75, 1.55, 2.95, 1.3, "4.25 vs 4.01", "equation residual (truth)", color=GREEN, vsize=26)
picture(sl, FIG / "fig_diagnostics.png", 0.6, 3.05, 12.1, 2.85)
callout(sl, "Largest errors: nearly uniform permeability with large pressure, where the amplitude is hard to infer", y=6.05, h=0.7)

# 10. Examples
sl = base_slide("Example Predictions", "Best, typical and worst test cases of the final FNO", 10)
picture(sl, FIG / "fig_samples.png", 0.6, 1.5, 7.9, 5.1)
bullets(sl, 8.8, 1.8, 4.0, [
    ("Best", "Error 0.2%, almost no visible error"),
    ("Typical", "Error 0.7%, shapes match closely"),
    ("Worst", "Uniform permeability: right shape, wrong height (36%)"),
], size=17, gap=1.35)

# 11. Summary and limits
sl = base_slide("Summary and Limits", "What we learned and what we do not claim", 11)
rounded(sl, 0.6, 1.6, 6.0, 4.3)
textbox(sl, 0.85, 1.7, 5.5, 0.5, "What we learned", size=20, bold=True, color=BLUE)
bullets(sl, 0.9, 2.35, 5.5, [
    ("Training mattered most", "Error halved with a better recipe"),
    ("FNO beats a tuned MLP", "2x more accurate, 10x fewer parameters"),
    ("Strong against a solver", "More accurate at the same grid size"),
], size=16, gap=1.15)
rounded(sl, 6.85, 1.6, 5.85, 4.3, fill=RGBColor(0xFF, 0xF8, 0xF0), line=ORANGE)
textbox(sl, 7.1, 1.7, 5.4, 0.5, "Honest limits", size=20, bold=True, color=ORANGE)
bullets(sl, 7.15, 2.35, 5.3, [
    ("One dataset", "Results are about agreement with this data"),
    ("Timing caveats", "Solver tuned to the data, shared machine"),
    ("Stage 2 speed claim withdrawn", "The old reference solver was slow"),
], size=16, gap=1.15, num_color=ORANGE)
callout(sl, "Better training beat a bigger model, and the FNO stayed accurate and fast", y=6.1, h=0.65)

# 12. Team contributions & AI declaration
sl = base_slide("Team Contributions & AI Declaration", "Individual ownership and adherence to AE646 academic integrity guidelines", 12)

members = [
    ("Aryamann Srivastava", "Lead Developer",
     ["Stage 3 pipeline & D4 augmentation",
      "Recipe & architecture studies; final models",
      "Finite-volume solver & latency benchmarks",
      "Reports, decks & code packaging"],
     BLUE),
    ("Varun Sathaye", "Data & Baseline Verification",
     ["PDEBench train/val/test split verification",
      "Normalisation & Stage 2 baseline checks",
      "Solver calibration & timing review",
      "Section 7 results & presentation"],
     NAVY),
    ("Atishay Jain", "Architecture & Ablation Review",
     ["Training recipe & mode/width/depth sweeps",
      "Ablation tables & Pareto analysis",
      "Model-selection protocol verification",
      "Section 4-5 results & presentation"],
     GREEN),
    ("Vedant S. Tiwari", "Scaling & Diagnostics",
     ["Data-scaling & cross-resolution transfer",
      "Physical diagnostics & residual checks",
      "Failure mode analysis (uniform kappa)",
      "Section 6 & 8 results & presentation"],
     ORANGE),
]

card_w = 2.85
gap = 0.23
for i, (name, role, tasks, col) in enumerate(members):
    x_pos = 0.6 + i * (card_w + gap)
    rounded(sl, x_pos, 1.55, card_w, 3.65, fill=WHITE, line=LINE)
    top_bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x_pos), Inches(1.55), Inches(card_w), Inches(0.08))
    rgb_fill(top_bar, col)
    top_bar.line.fill.background()
    textbox(sl, x_pos + 0.12, 1.7, card_w - 0.24, 0.42, name, size=14, bold=True, color=NAVY)
    textbox(sl, x_pos + 0.12, 2.08, card_w - 0.24, 0.32, role, size=10, bold=True, color=col)

    tb = sl.shapes.add_textbox(Inches(x_pos + 0.08), Inches(2.42), Inches(card_w - 0.16), Inches(2.7))
    tf = tb.text_frame
    tf.word_wrap = True
    for j, t in enumerate(tasks):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = "• " + t
        p.font.name = FONT
        p.font.size = Pt(10)
        p.font.color.rgb = GREY
        p.space_after = Pt(3)

# AI Tool Use Declaration Callout
rounded(sl, 0.6, 5.35, 12.1, 1.5, fill=PALE, line=BLUE, line_w=1.2)
textbox(sl, 0.85, 5.43, 11.6, 0.3, "AI-Tool-Use Declaration", size=13, bold=True, color=NAVY)
textbox(sl, 0.85, 5.75, 11.6, 1.0,
        ["AI tools were used for coding assistance, debugging, experiment orchestration (Colab/Kaggle GPU runtimes), and LaTeX/PPT drafting.",
         "All numerical results come from real runs on real PDEBench data; no results or metrics were generated or invented. Every line of code, equation, and conclusion is fully owned, verified, and defendable by all team members in the viva."],
        size=10.5, color=GREY)

prs.save(OUT)
print("Saved ->", OUT)
