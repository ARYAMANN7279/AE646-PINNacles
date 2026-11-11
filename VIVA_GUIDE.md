# AE646 Stage 3 viva guide (team PINNacles)

Presentation and viva: Thu 12 Nov 2026, 8:50-9:15 AM (10-minute talk, then viva).
All numbers below come from the stored results (`results/stage3/`) and the final report.

---

## 1. The one-minute story

**Problem:** predict the pressure field `u` in porous-medium flow (Darcy flow) from a permeability map `kappa`, using a neural network instead of a PDE solver.

**Models:** Fourier Neural Operator (FNO) versus an MLP baseline.

**What happened:**
- Stage 2: FNO error 0.052, MLP 0.08.
- Stage 3 question: was the remaining error about the model, or about how we trained it?
- Answer: mostly the training. A better recipe halved the error. Final FNO 0.0202 versus MLP 0.0415.
- We also compared against a classical finite-volume solver and checked the physics of the predictions.

---

## 2. The physics

- Equation: -div(kappa grad u) = f on the unit square. Darcy's law gives flow q = -kappa grad u; mass conservation gives div q = f.
- Boundary condition: u = 0 on all four edges. Source: f = 1 everywhere.
- kappa: permeability, only two values (0.1 and 1.0), made by thresholding a smooth random field (blobs of high and low permeability). This is the input.
- u: pressure. This is the output.
- Operator learning: learn the map G: kappa -> u. After training, a new kappa gives u in one forward pass, with no solve.
- Not a PINN: the physics is not in the loss. The data comes from PDE solves, and we check the physics afterwards.

---

## 3. Data

- PDEBench 2D Darcy flow (beta = 1.0), 10,000 real samples at 128 x 128, checksum verified.
- Subsampled with stride 2 to 64 x 64.
- Split: 900 train / 100 validation / 200 test, fixed seed 42.
- Inputs: 3 channels (kappa, x coordinate, y coordinate), standardised with training-set statistics.
- Metric: per-sample relative L2 error, ||u_hat - u||_2 / ||u||_2, computed in physical units, averaged over the test set. Lower is better.

---

## 4. The two models

**MLP**
- Flattens the 3 x 64 x 64 input to 12,288 numbers, then 3 hidden layers of 4,096 units with GELU, then 4,096 outputs reshaped to 64 x 64.
- 100.7 M parameters. No built-in notion of neighbouring pixels.

**FNO**
- Lift: 3 channels to 64.
- 8 Fourier layers. Each does an FFT, keeps only the lowest 12 x 12 frequency modes, multiplies them by learned weights, applies the inverse FFT, adds a 1 x 1 skip connection, then GELU.
- Projection: 64 to 128 to 1.
- 9.5 M parameters.
- Why it suits the problem: the FFT lets every layer see the whole field, and the same operator is reused at every location.

---

## 5. Training setup

- AdamW, learning rate 1e-3, weight decay 1e-4, batch size 16.
- Loss: relative L2 per sample, averaged over the batch.
- One smooth cosine learning-rate decay.
- D4 augmentation (random rotations and flips of the square), applied before subsampling. Rotating an already subsampled grid would shift the grid phase, so it is done on the 128 x 128 data.
- Test-time averaging: at 64 x 64 only identity and transpose are phase-consistent, so we average over those two.
- Checkpoint: best validation step.
- Budget: 5,700 to 20,000 steps for the studies, 60,000 steps and 5 seeds for the final models.

---

## 6. Findings

### Finding 1: the training recipe
Stage 2 used MSE loss, a cosine schedule that restarted every 200 steps (about 28 times), and no augmentation. Changing all three at the same budget (3 seeds):

| | Stage 2 recipe | New recipe |
|---|---|---|
| FNO | 0.0524 +- 0.0018 | 0.0249 +- 0.0002 |
| MLP | 0.0802 | 0.0454 |

### Finding 2: architecture barely matters
- Modes 4 to 32: error only 0.022 to 0.026.
- Width from 32 upwards: flat.
- Depth 2 to 8: error 0.030 down to 0.020.
- Learning rate, weight decay and batch size (8, 16, 32): each changes error by less than 0.001.
- Steps 5.7k to 80k: 0.0247 to 0.0205 (slow gains).
- Final choice (by validation error): 12 modes, width 64, 8 layers.

### Finding 3: data and resolution
- FNO vs training samples: 100 -> 0.044, 400 -> 0.026, 900 -> 0.022, 5,000 -> 0.020, 9,700 -> 0.0198.
- MLP: 900 -> 0.041, then stuck near 0.036 even with 9,700.
- The FNO with 400 samples (0.026) beats the MLP with 9,700 (0.036).
- Same-grid train and test: 32 -> 0.027, 64 -> 0.022, 128 -> 0.021.
- Trained on 64, tested on other grids: 0.065 on 32, 0.038 on 128 (2 to 3 times worse).

### Final models (5 seeds, 60,000 steps)
- FNO 0.0202 +- 0.0004, MLP 0.0415 +- 0.0009.
- The FNO is 2x more accurate with 10x fewer parameters.

---

## 7. Why the MLP cannot do better

- 10x more parameters (18 M to 100 M): error stays between 0.040 and 0.046.
- More training (20k to 60k steps): no gain (0.040 to 0.042).
- More data: helps up to about 5,000 samples, then flat near 0.035.
- Likely reason (our interpretation, not directly tested): it flattens the grid, so it has no idea which pixels are neighbours and must relearn the same local physics at every location. The FNO reuses one operator everywhere.

---

## 8. FNO versus a classical solver

- The solver is a finite-volume solver calibrated to the PDEBench data. The reference arrays behave as if the zero-pressure boundary sat about one cell outside the grid, so an uncalibrated solver would not be a fair comparison.
- Solver error: 0.0939 (32 x 32), 0.0438 (64 x 64), 0.0241 (128 x 128).
- Direct solve time on one CPU thread: 1.8, 7.1, 37.4 ms.
- FNO at 64 x 64: error 0.020; 1.4 ms at batch 1 on a GPU, 0.20 ms per sample at batch 128.
- 5x to 35x faster than the solver on a GPU. NOT faster on one CPU core (37 ms versus 7 ms).
- Break-even versus the 128 x 128 solver: about 50,000 solves.
- Stage 2 speed-up claims (1,443x and 7,740x) are withdrawn: the reference solver then was inefficient. The final report says so.
- Careful phrasing: the FNO agrees better with this dataset, partly because it learns the data generator's quirks. That is not the same as being more accurate than the true PDE.

---

## 9. Does it obey the physics?

- Mean pressure error: 1.7%.
- Boundary level: 0.1157 versus 0.1159 for the truth.
- Negative-pressure pixels: 0.0015%.
- PDE residual: 4.245 for the prediction versus 4.010 for the truth itself. Both are large because even the true data is only second-order accurate on the subsampled grid.
- Worst cases: nearly uniform permeability with large pressure. Right shape, wrong height (36% error).

---

## 10. Limitations (say these before they are asked)

1. One PDE and one dataset; scaling trends may not transfer.
2. Agreement with this data, not necessarily with the exact PDE.
3. Tied to the training grid: 2 to 3 times worse on other grids.
4. Speed caveats: not faster on one CPU core; timings on a shared machine.
5. Model selection used one seed per candidate; the top architectures are within seed noise.
6. The physics is checked, not enforced.

---

## 11. Practical and honesty facts

- The 142 training runs of the study ran on the lab GPU. The notebook ran end to end on Google Colab (T4, about 50 minutes) and Kaggle (T4 x2, about 35 minutes). Section 10 of the notebook loads the stored results instead of recomputing them, and says so explicitly.
- AI tools were used for coding, debugging and drafting. All numerical results come from real runs on real data.
- Be ready to explain the parts listed against your name in the contribution statement, and to say what each teammate did.
- Marking (TA email): implementation 8, baseline vs SciML comparison 7, quantitative evaluation 7, physical interpretation 7, reproducibility 4, report 10, presentation 2, viva 5. The viva counts for more than the slides.

---

## 12. Likely questions, with short answers

**Why relative L2 and not MSE?** The reported metric is relative, and MSE under-weights low-amplitude samples.

**Why is D4 augmentation valid?** With f = 1 and zero boundary the PDE is exactly equivariant under rotations and flips of the square.

**Why 12 modes?** Error is flat from 8 to 32 modes. 32 modes cost 7x the parameters and run 3.6x slower for no gain.

**Why batch size 16?** It was the Stage 2 setting. Batch 8, 16, 32 give 0.0222, 0.0219, 0.0221, so it does not matter here.

**Does a bigger batch raise the error?** Not in our runs; the differences are within noise.

**Why learning rate 1e-3?** 0.0005 to 0.002 stay within 0.001 of each other; 1e-3 is the standard middle value.

**Why train on only 900 samples?** That was the Stage 2 protocol, and the data-scaling study shows the FNO already does well at 400.

**Why does the FNO beat the MLP?** It shares one operator across the whole grid and sees the full field through the FFT; the MLP has no spatial structure.

**Is it a PINN?** No. It is purely data-driven. Physics enters through the data, the symmetry augmentation and the checks.

**Why is the FNO not faster on a CPU?** It does large FFTs and many wide layers, while the solver does one sparse solve at 64 x 64.

**Why does error rise on another grid?** A different grid samples the sharp permeability edges and the boundary differently, so the learned map is not exactly grid-independent. (Our interpretation.)

**What is operator learning?** Learning a map between functions (kappa to u) rather than one solution.

**What would you do next?** Other PDEs, a physics-informed loss term, more resolutions, uncertainty estimates.

**What did you withdraw or correct?** The Stage 2 speed-up claim, because the reference solver was slow, and the Stage 2 training recipe, which was sub-optimal.
