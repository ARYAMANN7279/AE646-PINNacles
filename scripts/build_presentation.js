const pptxgen = require("pptxgenjs");

const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const INK = "222222";
const MUTE = "666666";
const CARD = "F4F6FC";
const GOOD = "1E7F4C";

const path = require("path");
// Repo root resolved relative to this script (which lives in <root>/scripts/), so the
// build is portable — no hardcoded user paths.
const REPO = path.join(__dirname, "..") + path.sep;
const IMG_COMPARE = REPO + "results/comprehensive_comparison.png";
const IMG_SAMPLES = REPO + "results/run_001/evaluation/sample_predictions.png";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 in
const PW = 13.333, PH = 7.5;

const FONT = "Calibri";
const FONT_HEAD = "Cambria";

function footer(slide, pageNum) {
  slide.addText(`AE646 — Fourier Neural Operator for Parametric Darcy Flow`, {
    x: 0.5, y: PH - 0.42, w: 9, h: 0.3, fontFace: FONT, fontSize: 9, color: MUTE, margin: 0,
  });
  slide.addText(String(pageNum), {
    x: PW - 1.0, y: PH - 0.42, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 9, color: MUTE,
    align: "right", margin: 0,
  });
}

function contentSlide(title, pageNum, kicker) {
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: 0.6, y: 0.35, w: 10, h: 0.3, fontFace: FONT, fontSize: 12, color: NAVY,
      bold: true, charSpacing: 2, margin: 0,
    });
  }
  slide.addText(title, {
    x: 0.6, y: kicker ? 0.62 : 0.45, w: 12.1, h: 0.8, fontFace: FONT_HEAD, fontSize: 30,
    bold: true, color: NAVY, margin: 0,
  });
  footer(slide, pageNum);
  return slide;
}

function bulletBox(slide, items, opts) {
  const paras = items.map((t, i) => ({
    text: t,
    options: {
      bullet: { code: "2022", indent: 18 },
      breakLine: i < items.length - 1,
      color: opts.color || INK,
      fontSize: opts.fontSize || 15,
      paraSpaceAfter: opts.spaceAfter != null ? opts.spaceAfter : 10,
      bold: false,
    },
  }));
  slide.addText(paras, {
    x: opts.x, y: opts.y, w: opts.w, h: opts.h, fontFace: FONT, valign: "top", margin: 0,
    lineSpacingMultiple: 1.15,
  });
}

function statCard(slide, x, y, w, h, value, label, valueColor) {
  slide.addShape("roundRect", {
    x, y, w, h, rectRadius: 0.08, fill: { color: CARD }, line: { color: CARD },
    shadow: { type: "outer", color: "888888", opacity: 0.25, blur: 6, offset: 2, angle: 90 },
  });
  slide.addText(value, {
    x: x + 0.15, y: y + 0.12, w: w - 0.3, h: h * 0.58, fontFace: FONT_HEAD, fontSize: 30,
    bold: true, color: valueColor || NAVY, align: "center", valign: "bottom", margin: 0,
  });
  slide.addText(label, {
    x: x + 0.1, y: y + h * 0.62, w: w - 0.2, h: h * 0.35, fontFace: FONT, fontSize: 11.5,
    color: MUTE, align: "center", valign: "top", margin: 0,
  });
}

function dataTable(slide, rows, opts) {
  slide.addTable(rows, {
    x: opts.x, y: opts.y, w: opts.w, colW: opts.colW,
    fontFace: FONT, fontSize: opts.fontSize || 13, border: { type: "solid", color: "D8DEEC", pt: 0.75 },
    autoPage: false,
  });
}

function headerCellStyle(text) {
  return { text, options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "left", valign: "middle" } };
}
function cellStyle(text, opts) {
  return { text, options: Object.assign({ color: INK, align: "left", valign: "middle", fill: { color: WHITE } }, opts || {}) };
}

// ---------- Slide 1: Title ----------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };
  slide.addText("SCIENTIFIC MACHINE LEARNING — FINAL PROJECT", {
    x: 0.9, y: 2.15, w: 11.5, h: 0.4, fontFace: FONT, fontSize: 13, color: ICE, bold: true,
    charSpacing: 3, margin: 0,
  });
  slide.addText("Fourier Neural Operator for\nParametric Darcy Flow", {
    x: 0.9, y: 2.65, w: 11.5, h: 2.0, fontFace: FONT_HEAD, fontSize: 42, bold: true, color: WHITE,
    lineSpacingMultiple: 1.08, margin: 0,
  });
  slide.addText("AE646: Scientific Machine Learning for Fluid Mechanics", {
    x: 0.9, y: 4.85, w: 11.5, h: 0.45, fontFace: FONT, fontSize: 17, color: ICE, margin: 0,
  });
  slide.addText("Aryaman Srivastava", {
    x: 0.9, y: 5.35, w: 11.5, h: 0.4, fontFace: FONT, fontSize: 15, color: "9FB3E0", margin: 0,
  });
  slide.addShape("ellipse", { x: 11.0, y: -1.6, w: 4.6, h: 4.6, fill: { color: "273775" }, line: { color: "273775" } });
  slide.addShape("ellipse", { x: 12.0, y: 5.6, w: 3.2, h: 3.2, fill: { color: "273775" }, line: { color: "273775" } });
}

// ---------- Slide 2: Problem & Motivation ----------
{
  const s = contentSlide("The Problem & Motivation", 2, "Parametric PDEs");
  slide2Body(s);
}
function slide2Body(s) {
  s.addText("PARAMETRIC PDE CHALLENGE", { x: 0.6, y: 1.55, w: 5.9, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  bulletBox(s, [
    "Darcy flow: −∇·(κ∇u) = f, with κ piecewise-constant in {0.1, 1.0}",
    "Real PDEBench data: 1000 train/val, 200 test (of 10,000 total, seed=42 subset)",
    "Need a fast surrogate for optimization, UQ, inverse problems",
  ], { x: 0.6, y: 1.95, w: 5.9, h: 2.6, fontSize: 15 });

  s.addText("WHY OPERATOR LEARNING?", { x: 6.9, y: 1.55, w: 5.9, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  bulletBox(s, [
    "Traditional solvers: re-mesh + re-solve per parameter (slow — see slide 10)",
    "FNO/DeepONet: learn G: κ → u once, evaluate in ~1 ms",
    "Zero-shot super-resolution: train at 64×64, test at PDEBench's real native 128×128",
  ], { x: 6.9, y: 1.95, w: 5.9, h: 2.6, fontSize: 15 });

  s.addShape("line", { x: 6.55, y: 1.6, w: 0, h: 2.9, line: { color: "E3E7F2", width: 1 } });

  statCard(s, 0.6, 5.15, 3.7, 1.55, "-∇·(κ∇u)=f", "Steady-state Darcy flow equation", NAVY);
  statCard(s, 4.55, 5.15, 3.7, 1.55, "10,000", "Real PDEBench samples (128×128)", NAVY);
  statCard(s, 8.5, 5.15, 4.2, 1.55, "2 models", "FNO (spectral) vs MLP (dense) baseline", NAVY);
}

// ---------- Slide 3: Dataset & Baseline ----------
{
  const s = contentSlide("Dataset & Baseline", 3, "Setup");
  s.addText("REAL PDEBENCH 2D DARCY FLOW (β=1.0)", { x: 0.6, y: 1.55, w: 12.1, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  bulletBox(s, [
    "Downloaded from the official source, verified by checksum against PDEBench's manifest",
    "900 train / 100 val / 200 test, 64×64 (downsampled from native 128×128)",
    "Piecewise-constant permeability (κ ∈ {0.1, 1.0}) from a thresholded Gaussian random field",
  ], { x: 0.6, y: 1.95, w: 12.1, h: 1.9, fontSize: 15 });

  s.addShape("roundRect", { x: 0.6, y: 4.05, w: 12.13, h: 2.6, rectRadius: 0.06, fill: { color: CARD }, line: { color: CARD } });
  s.addText("BASELINE: MLP", { x: 1.0, y: 4.3, w: 6, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  bulletBox(s, [
    "Flatten 64×64×3 → 3×2048 hidden layers → flatten output",
    "42.0M parameters",
    "No spatial inductive bias — treats the field as a dense vector",
  ], { x: 1.0, y: 4.7, w: 5.6, h: 1.8, fontSize: 14.5 });

  s.addShape("line", { x: 6.65, y: 4.3, w: 0, h: 2.1, line: { color: "D8DEEC", width: 1 } });

  s.addText("Why this dataset?", { x: 6.95, y: 4.3, w: 5.4, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  s.addText("PDEBench is the handout's own suggested source for this project theme (operator learning for parametric PDEs) — a peer-reviewed, real 10,000-sample benchmark, not a hand-generated stand-in.", {
    x: 6.95, y: 4.7, w: 5.4, h: 1.8, fontFace: FONT, fontSize: 13.5, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.2,
  });
}

// ---------- Slide 4: Fourier Neural Operator ----------
{
  const s = contentSlide("Fourier Neural Operator", 4, "Method");
  s.addText("ARCHITECTURE", { x: 0.6, y: 1.55, w: 12.1, h: 0.32, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  s.addShape("roundRect", { x: 0.6, y: 1.92, w: 12.13, h: 0.85, rectRadius: 0.06, fill: { color: NAVY }, line: { color: NAVY } });
  s.addText("Input (3 ch)  →  Lift  →  4× SpectralConv + 1×1 Conv + LayerNorm + GELU  →  Project  →  Output (1 ch)", {
    x: 0.6, y: 1.92, w: 12.13, h: 0.85, fontFace: "Courier New", fontSize: 13.5, color: WHITE, align: "center", valign: "middle", margin: 0,
  });

  s.addText("SPECTRAL CONVOLUTION", { x: 0.6, y: 3.05, w: 6.0, h: 0.32, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  bulletBox(s, [
    "FFT → multiply learned weights in Fourier space → IFFT",
    "Keeps only low-frequency modes (modes=12)",
    "Resolution-invariant by construction",
  ], { x: 0.6, y: 3.45, w: 6.0, h: 2.0, fontSize: 14.5 });

  statCard(s, 6.95, 3.05, 2.7, 1.9, "4.7M", "FNO parameters", NAVY);
  statCard(s, 9.85, 3.05, 2.9, 1.9, "42.0M", "MLP parameters (baseline)", MUTE);

  s.addText("Result: FNO matches or beats the MLP with 8.8× fewer parameters — see next slide.", {
    x: 6.95, y: 5.15, w: 5.8, h: 0.9, fontFace: FONT, fontSize: 13.5, italic: true, color: NAVY, valign: "top", margin: 0, lineSpacingMultiple: 1.2,
  });
}

// ---------- Slide 5: Results Summary ----------
{
  const s = contentSlide("Results Summary", 5, "Headline numbers");
  const rows = [
    [headerCellStyle("Model"), headerCellStyle("Test Mean Rel L2"), headerCellStyle("Parameters")],
    [cellStyle("FNO (original)", { bold: true, fill: { color: CARD } }), cellStyle("0.0521", { bold: true, color: NAVY, fill: { color: CARD } }), cellStyle("4.7M", { bold: true, fill: { color: CARD } })],
    [cellStyle("MLP baseline"), cellStyle("0.0820"), cellStyle("42.0M")],
    [cellStyle("FNO (improved)", { bold: true, fill: { color: CARD } }), cellStyle("0.0456", { bold: true, color: NAVY, fill: { color: CARD } }), cellStyle("78.8M", { bold: true, fill: { color: CARD } })],
  ];
  dataTable(s, rows, { x: 0.9, y: 1.75, w: 11.5, colW: [5.1, 3.7, 2.7], fontSize: 15 });

  statCard(s, 0.9, 4.35, 3.7, 1.65, "36%", "lower error vs MLP", GOOD);
  statCard(s, 4.85, 4.35, 3.7, 1.65, "8.8×", "fewer parameters than MLP", GOOD);
  statCard(s, 8.8, 4.35, 3.6, 1.65, "12.6%", "further gain from more capacity", NAVY);

  s.addText("FNO: 36% lower error than MLP, using 8.8× fewer parameters.", {
    x: 0.9, y: 6.25, w: 11.5, h: 0.5, fontFace: FONT, fontSize: 15, bold: true, italic: true, color: NAVY, margin: 0,
  });
}

// ---------- Slide 6: Error Distributions (image) ----------
{
  const s = contentSlide("Error Distributions", 6, "Diagnostics");
  const compareAspect = 2684 / 1784; // width / height
  const imgH = 4.1, imgW = imgH * compareAspect;
  const imgX = (PW - imgW) / 2, imgY = 1.6;
  s.addImage({ path: IMG_COMPARE, x: imgX, y: imgY, w: imgW, h: imgH });
  bulletBox(s, [
    "FNO and MLP show a similar error spread (std ≈ 0.045), but FNO sits at a lower mean/median",
    "Both models' worst cases come from the same kind of input (near-uniform permeability, slide 9) — a genuinely hard case, not a model-specific failure",
  ], { x: 1.35, y: imgY + imgH + 0.18, w: 10.63, h: 0.95, fontSize: 12.5, spaceAfter: 4 });
}

// ---------- Slide 7: Sample Predictions (full-bleed tall image) ----------
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Sample Predictions", {
    x: 0.5, y: 0.28, w: 8, h: 0.5, fontFace: FONT_HEAD, fontSize: 22, bold: true, color: NAVY, margin: 0,
  });
  slide.addText("Permeability input  ·  Target pressure  ·  FNO prediction  ·  Absolute error", {
    x: 8.6, y: 0.34, w: 4.2, h: 0.4, fontFace: FONT, fontSize: 11.5, color: MUTE, align: "right", margin: 0,
  });
  const availH = PH - 0.95 - 0.4;
  const imgAspect = 2397 / 4701;
  let imgH = availH, imgW = imgH * imgAspect;
  if (imgW > 6.4) { imgW = 6.4; imgH = imgW / imgAspect; }
  slide.addImage({ path: IMG_SAMPLES, x: (PW - imgW) / 2, y: 0.85, w: imgW, h: imgH });
  footer(slide, 7);
}

// ---------- Slide 8: Zero-Shot Super-Resolution ----------
{
  const s = contentSlide("Zero-Shot Super-Resolution", 8, "Resolution invariance");
  s.addText("Trained at 64×64. Evaluated at PDEBench's real native 128×128 — no retraining.", {
    x: 0.6, y: 1.55, w: 12.1, h: 0.45, fontFace: FONT, fontSize: 14.5, italic: true, color: INK, margin: 0,
  });
  const rows = [
    [headerCellStyle("Model"), headerCellStyle("64×64 (train)"), headerCellStyle("128×128 (zero-shot)"), headerCellStyle("Relative increase")],
    [cellStyle("FNO (original)", { bold: true }), cellStyle("0.0521"), cellStyle("0.0594"), cellStyle("+14.0%", { color: NAVY, bold: true })],
    [cellStyle("FNO (improved)", { bold: true }), cellStyle("0.0456"), cellStyle("0.0563"), cellStyle("+23.5%", { color: NAVY, bold: true })],
  ];
  dataTable(s, rows, { x: 0.6, y: 2.2, w: 12.13, colW: [3.5, 2.9, 3.03, 2.7], fontSize: 14.5 });

  bulletBox(s, [
    "Both FNOs evaluate on a resolution with 4× as many pixels, without retraining",
    "The improved FNO degrades more in relative terms — consistent with its larger capacity fitting some 64×64-specific discretization detail more tightly",
    "MLP cannot do this at all — fixed input dimension",
  ], { x: 0.6, y: 4.1, w: 12.13, h: 2.2, fontSize: 15 });
}

// ---------- Slide 9: A Genuine Failure Mode ----------
{
  const s = contentSlide("A Genuine Failure Mode", 9, "Physical interpretation");
  s.addShape("roundRect", { x: 0.6, y: 1.8, w: 12.13, h: 3.6, rectRadius: 0.07, fill: { color: CARD }, line: { color: CARD } });
  s.addText("Worst-case test samples (both models) have near-uniform permeability.", {
    x: 1.0, y: 2.1, w: 11.3, h: 0.7, fontFace: FONT, fontSize: 17, bold: true, color: NAVY, margin: 0,
  });
  bulletBox(s, [
    "Little spatial contrast for the model to exploit, producing a smooth radial pressure bump",
    "Both models slightly over-predict the peak amplitude of that bump",
    "Consistent with the training distribution: GRF correlation length 0.1 favors sharp blob structure, so near-uniform fields are rare in training",
  ], { x: 1.0, y: 2.85, w: 11.3, h: 2.4, fontSize: 15.5, spaceAfter: 12 });
}

// ---------- Slide 10: Inference-Speed Benchmark ----------
{
  const s = contentSlide("Real Inference-Speed Benchmark", 10, "Measured, not asserted");
  const rows = [
    [headerCellStyle("Method"), headerCellStyle("Time / sample"), headerCellStyle("Device")],
    [cellStyle("FNO (original)"), cellStyle("0.71 ms"), cellStyle("CUDA")],
    [cellStyle("MLP"), cellStyle("0.13 ms", { bold: true, color: NAVY }), cellStyle("CUDA")],
    [cellStyle("FDM solver (scipy sparse)"), cellStyle("1030 ms"), cellStyle("CPU")],
  ];
  dataTable(s, rows, { x: 0.6, y: 1.7, w: 7.3, colW: [3.6, 2.0, 1.7], fontSize: 14.5 });

  statCard(s, 8.2, 1.7, 4.53, 1.55, "1000×+", "faster than solving the PDE numerically", GOOD);

  s.addShape("roundRect", { x: 0.6, y: 4.35, w: 12.13, h: 2.35, rectRadius: 0.07, fill: { color: CARD }, line: { color: CARD } });
  s.addText("Counter-intuitive but real:", { x: 1.0, y: 4.6, w: 11.3, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("MLP is faster per sample than FNO despite 9× more parameters. FNO's FFT/IFFT round-trips and complex-valued arithmetic aren't as well amortized as MLP's large dense matmuls at single-sample inference on a modern GPU — parameter count and wall-clock latency are different things.", {
    x: 1.0, y: 5.0, w: 11.3, h: 1.55, fontFace: FONT, fontSize: 14, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.25,
  });
}

// ---------- Slide 11: Key Findings ----------
{
  const s = contentSlide("Key Findings", 11, "Summary");
  const items = [
    ["1", "FNO beats MLP on accuracy with far fewer parameters — spectral inductive bias helps"],
    ["2", "Original FNO is the most parameter-efficient configuration; improved FNO trades 16.6× more parameters for a 12.6% error reduction"],
    ["3", "Zero-shot super-resolution genuinely works, verified against real PDEBench ground truth"],
    ["4", "Parameter count ≠ inference latency: measured speed tells a different story than parameter counts alone"],
    ["5", "The gap to commonly-cited FNO literature numbers (~0.01–0.02) traces to real dataset differences (piecewise-constant vs continuous permeability), not a metric-definition artifact"],
  ];
  let y = 1.7;
  const rowH = [0.72, 0.95, 0.72, 0.85, 1.15];
  items.forEach((it, i) => {
    const h = rowH[i];
    s.addShape("ellipse", { x: 0.6, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(it[0], { x: 0.6, y: y + 0.05, w: 0.5, h: 0.5, fontFace: FONT, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText(it[1], { x: 1.35, y: y, w: 11.4, h: h, fontFace: FONT, fontSize: 14.5, color: INK, valign: "middle", margin: 0, lineSpacingMultiple: 1.15 });
    y += h + 0.06;
  });
}

// ---------- Slide 12: Limitations & Future Work ----------
{
  const s = contentSlide("Limitations & Future Work", 12, "Looking ahead");
  s.addShape("roundRect", { x: 0.6, y: 1.7, w: 5.9, h: 4.3, rectRadius: 0.07, fill: { color: CARD }, line: { color: CARD } });
  s.addText("LIMITATIONS", { x: 1.0, y: 1.95, w: 5.1, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  bulletBox(s, [
    "FFT implies periodic BC; the PDE itself uses Dirichlet boundaries",
    "Fixed modes cannot adapt to varying frequency content per-sample",
    "FNO's parameter efficiency doesn't automatically mean lower latency (slide 10)",
  ], { x: 1.0, y: 2.4, w: 5.1, h: 3.4, fontSize: 14.5, spaceAfter: 14 });

  s.addShape("roundRect", { x: 6.83, y: 1.7, w: 5.9, h: 4.3, rectRadius: 0.07, fill: { color: NAVY }, line: { color: NAVY } });
  s.addText("FUTURE WORK", { x: 7.23, y: 1.95, w: 5.1, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: ICE, margin: 0 });
  bulletBox(s, [
    "U-FNO / WNO architectures for sharper interfaces",
    "Physics-informed loss (PDE residual)",
    "Time-dependent / 3D extensions",
  ], { x: 7.23, y: 2.4, w: 5.1, h: 3.4, fontSize: 14.5, color: WHITE, spaceAfter: 14 });
}

// ---------- Slide 13: Reproducibility ----------
{
  const s = contentSlide("Reproducibility", 13, "Everything is versioned");
  s.addText("ONE-COMMAND REPRODUCTION", { x: 0.6, y: 1.55, w: 12.1, h: 0.32, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  s.addShape("roundRect", { x: 0.6, y: 1.92, w: 12.13, h: 2.55, rectRadius: 0.05, fill: { color: "F0F1F5" }, line: { color: "D8DEEC" } });
  const cmds = [
    "pip install -r requirements.txt",
    "python src/download_data.py       # real PDEBench, checksum-verified",
    "python src/preprocess.py",
    "python src/train.py --config configs/fno.yaml",
    "python src/evaluate.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt",
    "python src/superres_eval.py --config configs/fno.yaml --checkpoint results/run_001/best_model.pt",
    "python src/benchmark_speed.py",
  ];
  s.addText(cmds.join("\n"), {
    x: 0.85, y: 2.1, w: 11.6, h: 2.2, fontFace: "Courier New", fontSize: 12, color: "1E2761", valign: "top", margin: 0, lineSpacingMultiple: 1.3,
  });

  bulletBox(s, [
    "All configs, seeds, and metrics are versioned in this repo",
    "Results independently cross-checked across two machines (Apple Silicon and an NVIDIA GPU)",
  ], { x: 0.6, y: 4.75, w: 12.13, h: 1.3, fontSize: 15 });
}

// ---------- Slide 14: Thank You ----------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };
  slide.addShape("ellipse", { x: -1.4, y: -1.8, w: 4.6, h: 4.6, fill: { color: "273775" }, line: { color: "273775" } });
  slide.addShape("ellipse", { x: 10.6, y: 4.8, w: 3.6, h: 3.6, fill: { color: "273775" }, line: { color: "273775" } });
  slide.addText("Thank You", {
    x: 0.9, y: 2.7, w: 11.5, h: 1.1, fontFace: FONT_HEAD, fontSize: 44, bold: true, color: WHITE, margin: 0,
  });
  slide.addText("Questions?", {
    x: 0.9, y: 3.75, w: 11.5, h: 0.6, fontFace: FONT, fontSize: 19, color: ICE, margin: 0,
  });
  slide.addText("Report:  FINAL_REPORT.md      Results:  results/      Code:  src/", {
    x: 0.9, y: 4.6, w: 11.5, h: 0.4, fontFace: "Courier New", fontSize: 13.5, color: "9FB3E0", margin: 0,
  });
}

pres.writeFile({ fileName: REPO + "docs/PRESENTATION.pptx" }).then(() => {
  console.log("WROTE PPTX");
}).catch((e) => { console.error(e); process.exit(1); });
