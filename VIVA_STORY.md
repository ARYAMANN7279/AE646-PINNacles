# AE646 Stage 3: the full story, from zero, plus the questions you will be asked

How to use this file:
- **Part A** is the story in order. Read it once top to bottom. Each chapter is: the idea in plain words, then the light maths.
- **Part B** is a question bank (topic, dataset, training, concepts, results, tricky ones) with answers you can say out loud.
- **Part C** is a glossary and a one-page cheat sheet of numbers.

All numbers are from our stored results (`results/stage3/`) and the final report.

---

# PART A: THE STORY

## Chapter 0: What is this project about?

Engineers often need to know how a fluid behaves inside a material, for example water seeping through soil or oil moving through rock. The usual way is to **run a simulation**: a program that solves a physics equation on a computer. Simulations are accurate but slow, and you often need thousands of them (different rocks, different designs, uncertainty studies).

**Idea of the project (this field is called scientific machine learning, SciML):** train a neural network on many (input, answer) examples produced by a simulator. After training, the network gives a new answer almost instantly. Such a network is called a **surrogate** (a stand-in for the simulator).

**Our question:** which kind of neural network is the best surrogate for this problem, how good is it compared with the real simulation, and what actually makes it good?

Our theme (project theme 8) is **operator learning for parametric PDEs**:
- **PDE** = partial differential equation, the physics law.
- **Parametric** = the equation has an input that changes from case to case (here, the permeability map).
- **Operator learning** = learn the whole map from "input function" to "solution function", not a single solution.

---

## Chapter 1: The physics

Picture a square slab of porous material, like a sponge with some dense parts and some open parts. Fluid is pushed through it.

- **kappa (the permeability map):** how easily fluid flows at each spot. In our data it has only two values, **0.1** (hard to flow) and **1.0** (easy). It looks like a black-and-white blob image. **This is the input.**
- **u (the pressure):** the pressure at each spot. It looks like a smooth glowing bump. **This is the output.**
- **f = 1:** fluid is injected evenly everywhere (the "source").
- **u = 0 on the four edges:** the pressure is held at zero at the boundary (a "Dirichlet boundary condition").

The equation connecting them is

**-div( kappa * grad(u) ) = f**

In plain words:
- **grad(u)** is the slope of the pressure. Fluid flows from high to low pressure.
- **kappa * grad(u)** is the actual flow: slope times how easy it is to flow (this is **Darcy's law**).
- **div(...)** ("divergence") asks, at one point, how much more flow leaves than arrives.
- The equation says: **what leaves a point equals what the source puts in.** It is "fluid is conserved".

So the physics fixes u once kappa is known. Our job is to learn the map **kappa -> u**. That map is the **operator**.

Why it is a good test problem: it is a real physical equation, a solver exists to make ground truth, the answers are smooth, and the input is a sharp two-valued image (a hard input), so it tests whether a network understands spatial structure.

---

## Chapter 2: The dataset and how we measure success

**Dataset: PDEBench, 2D Darcy flow, beta = 1.0**
- PDEBench is a public benchmark for SciML from the University of Stuttgart (Takamoto et al., NeurIPS 2022). We downloaded the real file and verified its checksum.
- Each sample is a pair (kappa image, u image). The data was made by a numerical PDE solver at **128 x 128** resolution.
- kappa is **piecewise constant** (values 0.1 and 1.0), made by thresholding a smooth random field. That is why it looks like black-and-white blobs.
- "beta = 1.0" is the value of the constant source f. The benchmark has other beta values; we use 1.0.
- **10,000 samples** in total.

**What we did to it**
1. **Shrink to 64 x 64** by keeping every second pixel (cheaper to train).
2. **Split:** 900 train / 100 validation / 200 test, with a fixed random seed (42).
   - **Train:** the network learns from these.
   - **Validation:** we use these to choose settings (like a practice exam). We never train on them.
   - **Test:** the final score only, the surprise exam. Never used to choose anything.
3. **Standardise:** subtract the mean and divide by the standard deviation (computed from the training set only). This keeps numbers in a friendly range so training is stable.
4. **Add coordinate channels:** besides kappa we also give the network two images holding the x and y position of every pixel. So the input has **3 channels** x 64 x 64.

**How we score a prediction: relative L2 error**

error = || u_hat - u || / || u ||

where u_hat is the network's pressure image and u is the true one, and ||.|| is the usual "size" of an image (square root of the sum of squared pixels). It is the size of the mistake divided by the size of the true answer. **0.02 means about 2% off.** We compute it in the real physical units (undoing the standardising), per sample, then average over the 200 test samples. Lower is better.

---

## Chapter 3: How any neural network learns (needed for everything after)

A neural network is a big formula with millions of adjustable numbers called **weights** (or parameters).

1. Feed in an input; get a prediction.
2. Compare to the true answer with a **loss** (a number saying how wrong it was).
3. Nudge every weight slightly in the direction that lowers the loss. This is **gradient descent** (the gradient tells you which direction is "downhill"). The size of the nudge is the **learning rate**.
4. Repeat thousands of times.

Vocabulary we use:
- **Batch:** how many examples you look at for one nudge (we use **16**).
- **Step:** one nudge. **Epoch:** one pass over all the training data (900 / 16 is about 57 steps).
- **Optimiser:** the rule for nudging. We use **AdamW**, a smarter gradient descent that adapts the step per weight, with **weight decay** (a small pull of weights toward zero, 1e-4) to discourage overfitting.
- **Learning-rate schedule:** the learning rate shrinks over time (big steps early, tiny steps late). A **cosine schedule** shrinks it smoothly along a cosine curve from 1e-3 to about 0.
- **Overfitting:** memorising training examples without learning the real rule. It shows up as good training error but bad validation or test error.
- **Seed:** the random starting point (initial weights, order of batches). Different seeds give slightly different results, so we run several and report mean plus or minus standard deviation.

---

## Chapter 4: The first model, the MLP

**MLP = multi-layer perceptron**, the plainest neural network.

**How it works, step by step**
1. Take the input (3 x 64 x 64 = **12,288 numbers**) and **flatten it into one long list**.
2. A **layer** turns a list into a new list. Each new number is a weighted sum of *all* the old numbers plus a bias: **y = W x + b** (W is a big table of weights).
3. Then apply a bendy function, **GELU**, to every number. Without this, stacking layers is pointless, because a stack of weighted sums is just one big weighted sum. The bend is what lets the network learn curved, complicated relationships.
4. Repeat. Our MLP has **3 hidden layers of 4,096 numbers each**.
5. The last layer gives **4,096 numbers**, reshaped into the 64 x 64 pressure image.

**Size:** 12,288 x 4,096 is about 50 million weights in layer 1; the other layers add about 50 million more. **Total: 100.7 million parameters.**

**Its big weakness:** after flattening, the network does not know that pixel 5 and pixel 6 are neighbours, or that there is a picture at all. Every pixel is just a number in a list. So it must learn the same local physics ("if the neighbourhood looks like this, the pressure does that") **separately at every position**. That is wasteful, needs lots of data, and has no built-in understanding of space (no **inductive bias**).

**Why we use it:** as the **baseline** ("what do you get with the plainest network?"). If a physics-aware network clearly beats it, we have learned that structure matters.

---

## Chapter 5: The second model, the FNO

**FNO = Fourier Neural Operator** (Li et al., 2021). It is designed for exactly this kind of problem.

**Key idea 1: any image is a sum of waves.** The **Fourier transform (computed with the FFT, a fast algorithm)** splits an image into waves of different frequencies.
- **Low frequencies** = big, smooth shapes.
- **High frequencies** = fine, sharp detail.

Pressure fields here are smooth, so most of the information sits in the low frequencies.

**Key idea 2: mixing across space becomes simple multiplication in frequency space.** To let every pixel "talk" to every other pixel in normal space you would need a huge table of weights (that is the MLP). In frequency space, it is just multiplying each wave by a number. (This is the **convolution theorem**: a convolution in space equals a multiplication in frequency.)

**One Fourier layer does this**
1. Take the FFT of the current feature maps.
2. **Keep only the lowest 12 x 12 frequencies** (these are the "modes"). Throw the rest away.
3. Multiply each kept wave by **learned weights**.
4. Take the inverse FFT to get back an image.
5. Add a **parallel simple path** (a 1 x 1 layer that mixes the channels at each pixel), so sharp local detail is not lost.
6. Apply GELU.

**The whole FNO**
- **Lift:** turn the 3 input channels into 64 channels (a small layer at each pixel).
- **8 Fourier layers**, stacked.
- **Project:** squeeze 64 channels to 128, then to 1 (the pressure).
- **9.5 million parameters.** The spectral weights per layer are 2 x (64 x 64 x 12 x 12), about 1.2 million (counted as complex numbers), times 8 layers.

**Why it works well here**
- Every layer sees the **whole picture at once**, which matters because pressure at one point depends on kappa everywhere.
- The **same operator is reused at every location** (no relearning per pixel).
- It keeps only smooth information, which is right for this kind of PDE solution.
- **Because the weights live on frequencies and not on pixels**, an FNO can in principle be applied to a different grid size. (In practice we found a cost; see Chapter 10.)

**One honest caveat:** the FFT silently assumes the image repeats (is periodic), but our problem has zero pressure at the edges, which is not periodic. We used no padding. The 1 x 1 skip path and the coordinate channels help handle the boundary.

---

## Chapter 6: Stage 1 and Stage 2 (what we had before this final stage)

- **Stage 1:** the proposal. We picked the problem and the data and planned to compare FNO against the MLP.
- **Stage 2:** we trained both on the real data. Results: **FNO error 0.052, MLP error 0.08** (about 1.6x better). We also claimed the FNO was thousands of times faster than a classical solver. **That speed claim was later withdrawn (Chapter 11).**

So at the end of Stage 2 the FNO clearly won. But we suspected the numbers were not as good as they could be, and that is where Stage 3 starts.

---

## Chapter 7: Stage 3, part 1: "Was our training done badly?"

We went back and examined *how* Stage 2 trained. We found three weak points.

**1. The wrong loss.** Stage 2 trained with **MSE** (mean squared error: the average of squared mistakes). But we *score* with a *relative* error. MSE cares most about samples with large pressure, so samples with small pressure were neglected.
- **Fix:** train on the same relative L2 error we score with.
- The loss is L = (1/B) * sum over the batch of || u_hat_i - u_i || / || u_i ||.

**2. A badly behaved learning-rate schedule.** By a quirk of the setup, the cosine schedule went down to zero and then **jumped back up every 200 steps, about 28 times**. It is like walking downhill and being teleported back up the hill every few minutes.
- **Fix:** one smooth cosine decay.

**3. No data augmentation.** The physics is symmetric: rotate or flip the square, and the equation still holds, so the rotated kappa gives the correspondingly rotated u. That is a free way to get up to **8x more training examples** (the 8 rotations and flips are the **D4 group**).
- **Fix:** randomly rotate and flip each training sample.
- Detail: we do it on the original 128 x 128 image *before* shrinking, because rotating an already-shrunk grid would shift which pixels were sampled.
- At test time we also average over identity and transpose (**test-time averaging**); only those two are exactly consistent on the 64 x 64 grid.

**Result (same model, same number of steps, 3 seeds):**

| | Before | After |
|---|---|---|
| FNO | 0.0524 | 0.0249 |
| MLP | 0.0802 | 0.0454 |

**Training better, with no other change, cut the error roughly in half for both models.** This was the biggest finding.

---

## Chapter 8: Stage 3, part 2: "Does the model design matter?"

We changed one setting at a time and watched the validation error (a **sweep**):

- **Number of modes** (4 to 32): barely moved, only 0.022 to 0.026. We chose **12** (32 modes cost 7x the parameters and ran 3.6x slower for no gain).
- **Width** (channels): flat from 32 upward.
- **Depth** (layers): this helped (2 layers: 0.030; 8 layers: 0.020). We chose **8**.
- **Learning rate, weight decay, batch size (8, 16, 32):** each changed the error by less than 0.001. We kept **batch 16**, the Stage 2 setting.
- **Training length:** helps slowly (5,700 steps: 0.0247; 80,000 steps: 0.0205).

**Takeaway:** the training recipe mattered far more than architecture details, and the result does not depend on a lucky setting.

The MLP was tuned just as fairly; its best layout was 3 hidden layers of 4,096.

**Final models:** retrain the chosen settings for **60,000 steps** with **5 seeds** each:
- **FNO: 0.0202 +- 0.0004**
- **MLP: 0.0415 +- 0.0009**
- So the FNO is **2x more accurate with 10x fewer parameters**.

---

## Chapter 9: Why can't the MLP do better?

We tried to rescue it: bigger (18 M to 100 M parameters), longer training, more data. It never got below about 0.036 to 0.04. And the FNO with only **400 training samples (0.026)** beats the MLP with **9,700 (0.036)**.

Our explanation: the MLP has no spatial structure, so it spends its capacity relearning the same local rule at every pixel. This is **our interpretation**; we did not test it directly.

---

## Chapter 10: Data and resolution studies

**Data scaling** (how error falls with more training samples), FNO: 100 samples: 0.044, 400: 0.026, 900: 0.022, 5,000: 0.020, 9,700: 0.0198. So the FNO is **data-efficient** and levels off near 0.020.

**Resolution** (train on one grid, test on another):
- Same grid for training and testing: 0.027 (32 x 32), 0.022 (64 x 64), 0.021 (128 x 128).
- Trained on 64 x 64, tested on other grids: **2 to 3x worse** (0.065 on 32 x 32, 0.038 on 128 x 128).
- FNOs are *supposed* to be grid-independent, but here they are not fully. Likely reason (our interpretation): a different grid samples the sharp edges of kappa and the boundary differently, so the learned map is not exactly the same.

---

## Chapter 11: The classical solver (what it is and how it works)

This is the thing the network is trying to replace, and the fair comparison.

**What the solver does (the finite-volume method)**
1. Cut the square into a **64 x 64 grid of little cells**. Each cell gets one unknown number: its pressure.
2. For each cell write a rule: **flow in = flow out + what the source adds**. The flow across the wall between two neighbouring cells is (permeability at that wall) x (pressure difference) / (cell size).
3. Doing this for all 4,096 cells gives **4,096 equations in 4,096 unknowns**, written **A u = b**. A is a table that is mostly zeros, because each cell only touches its 4 neighbours (a **sparse matrix**).
4. **Solve the linear system**: with a direct method (sparse LU) or an iterative one (algebraic multigrid). That gives every pressure at once.

**Making the comparison fair**
- The PDEBench "truth" came from its own 128 x 128 solver. A standard solver at 64 x 64 does not match that exactly (a plain one was off by 5 to 10%). We found the data behaves as if the zero-pressure boundary sits **about one cell outside the grid**, so we **calibrated** our solver to match.
- Calibrated solver error against the dataset: **0.0939 (32 x 32), 0.0438 (64 x 64), 0.0241 (128 x 128)**. Time on one CPU core: **1.8, 7.1, 37.4 ms**.

**FNO versus solver at 64 x 64**
- **Accuracy:** FNO 0.020, solver 0.044. The FNO is 2.2x closer to the dataset. Careful wording: partly because the FNO learned the dataset's own quirks, so it is a statement about *this data*, not "more physically correct than the PDE".
- **Speed:** on a **GPU**, the FNO is **5x (one sample at a time) to 35x (many at once) faster**. On **one CPU core it is not faster** (37 ms versus 7 ms).
- **Break-even:** training took about 32 minutes, so the FNO pays off only after roughly **50,000 solves** (against the 128 x 128 solver).

**The correction we made:** in Stage 2 we said the FNO was over 1,000x faster. That was against a slow, inefficient reference solver. We **withdrew that claim**, and the final report says so.

---

## Chapter 12: Does the prediction obey the physics?

A low error does not prove the physics is respected, so we checked other things:

- **Mean pressure:** off by only **1.7%**.
- **Boundary:** the pressure at the edge ring is 0.1157 versus 0.1159 in the truth, so the boundary condition is respected.
- **Negative pressure** (impossible physically): only 0.0015% of pixels.
- **PDE residual:** plug the prediction into the discretised equation and see what is left over. It is **4.245 for the prediction versus 4.010 for the truth itself**. Both are large because the truth comes from the 128 x 128 solution shrunk to 64 x 64, so it does not satisfy the 64 x 64 equation exactly. The point is that the prediction is no worse than the truth.
- **Worst cases:** when permeability is nearly uniform and the pressure is large, the network gets the *shape* right but the *height* wrong (up to 36% error).

The physics is **checked, not enforced**. This is not a PINN (a network with the equation inside its loss).

---

## Chapter 13: Limitations (say these before you are asked)

1. One equation and one dataset (f = 1, two-valued kappa); trends may not transfer.
2. It matches *this data's* quirks, not necessarily the exact PDE.
3. It is tied to the training grid (2 to 3x worse on other grids).
4. The speed advantage depends on GPUs and batching.
5. Model selection used a single seed per candidate; the top architectures are within seed noise.
6. The physics is not built into the loss.

---

## Chapter 14: The whole story in one breath

"We learn the map from a permeability image to a pressure image for 2D Darcy flow on real PDEBench data. We trained two networks: an MLP, which sees the image as a flat list of numbers, and an FNO, which works in frequency space and reuses one operator across the whole grid. In Stage 2 the FNO won, 0.052 against 0.08. In Stage 3 we found our training was weak (MSE loss, restarting learning rate, no augmentation). Fixing that halved the error for both. Tuning the architecture added little. With both tuned fairly and 5 seeds, the FNO scored 0.0202 against 0.0415 with a tenth of the parameters, and the MLP stays stuck however much we scale it. We compared against a calibrated finite-volume solver: the FNO is closer to the data and 5 to 35 times faster on a GPU, but not on one CPU core, and we withdrew our earlier overstated speed claim. We checked the physics in several ways, and we state the limits openly."

---

# PART B: THE QUESTION BANK

## B1. Project topic questions

**Q: What is your project about, in two lines?**
We train neural networks to predict the pressure field of 2D Darcy flow from a permeability map, and compare a Fourier Neural Operator with an MLP baseline and with a classical finite-volume solver.

**Q: Why use a neural network when a solver exists?**
Solvers are accurate but cost time for every new input. If you need thousands of solves (design, optimisation, uncertainty), a trained surrogate answers in a fraction of a millisecond on a GPU.

**Q: What is operator learning, and how is it different from ordinary function learning?**
Ordinary learning maps numbers to numbers. Operator learning maps a whole function (the kappa image) to another function (the pressure image). The trained model works for any new kappa, not just one case.

**Q: What does "parametric PDE" mean?**
The same equation, solved for many different inputs (here many different permeability maps).

**Q: What is a PINN, and is this one?**
A physics-informed neural network puts the PDE residual into the loss so the network obeys the equation. Ours is not a PINN: it is purely data-driven. Physics enters through the training data, the symmetry augmentation and the checks afterward.

**Q: PINN vs FNO, when would you use which?**
A PINN is good when you have little or no data but know the equation, and it solves one instance. An FNO needs a lot of simulation data but learns the whole operator and then evaluates quickly for new inputs.

**Q: Why Darcy flow?**
It is a standard benchmark with a real physical meaning (flow in porous media, used in groundwater and oil reservoir engineering), smooth solutions, and a sharp input that tests spatial understanding.

**Q: What are real applications?**
Groundwater and reservoir modelling, CO2 storage, design optimisation, and uncertainty quantification where thousands of solves are needed.

**Q: What is new in your Stage 3 over Stage 2?**
We found the Stage 2 training recipe was sub-optimal and fixed it, tuned both models fairly with seeds, studied data and resolution, compared with a calibrated solver, and checked the physics.

---

## B2. Dataset questions

**Q: Where does the data come from?**
PDEBench (Takamoto et al., NeurIPS 2022), 2D Darcy flow, beta = 1.0, downloaded from the official source and verified by checksum. 10,000 real samples at 128 x 128. It was generated by a numerical PDE solver.

**Q: What is in one sample?**
A permeability map kappa (two values, 0.1 and 1.0) and the corresponding pressure field u.

**Q: What does beta = 1.0 mean?**
It is the constant source term f in the equation. The benchmark has other beta values; we use 1.0.

**Q: How is kappa generated?**
By thresholding a smooth Gaussian random field, which gives blobs of two values. Note this differs from the original FNO paper's Darcy data, which uses a continuous field.

**Q: What are the boundary conditions?**
u = 0 on all four edges.

**Q: Why 64 x 64 and not 128 x 128?**
Cost. We keep every second pixel. We also studied 32 and 128 to see the effect of resolution.

**Q: Why 900 / 100 / 200? Why not 80 / 10 / 10?**
900 train is the protocol we kept from Stage 2. We then ran a data-scaling study (100 to 9,700 samples) to show how results depend on the training size. 200 test samples is enough to score reliably, and validation is only for choosing settings.

**Q: How do you avoid data leakage?**
The split is by sample with a fixed seed. Normalisation statistics come from the training set only. Model selection used validation data only. The test set is touched only for the final score.

**Q: Why standardise the data?**
So inputs and outputs are on a similar scale, which makes optimisation stable. We always report error in physical units after undoing it.

**Q: Why add x and y coordinates as input channels?**
They tell the network where each pixel is. The MLP gets position information this way, and for the FNO it helps with the non-periodic boundary.

**Q: Is 900 samples enough?**
For the FNO yes: the data-scaling study shows it reaches 0.026 with only 400 and levels off near 0.020. The MLP is the one that struggles.

**Q: What is the dataset's limitation?**
One equation, one source value, two-valued permeability, one resolution family. Conclusions may not transfer.

---

## B3. Training-method questions

**Q: What loss did you use and why?**
Relative L2 error per sample, averaged over the batch. It matches the metric we report, and it does not under-weight low-amplitude samples (which MSE does).

**Q: What exactly was wrong in Stage 2?**
Three things: MSE loss on standardised pressure, a learning-rate schedule that restarted every 200 steps (about 28 times) instead of decaying once, and no augmentation.

**Q: Which optimiser, and why?**
AdamW, with learning rate 1e-3 and weight decay 1e-4. Adam adapts the step size per weight and is the standard choice for neural operators. The weight decay is a mild regulariser.

**Q: What is weight decay?**
A small penalty that pulls the weights toward zero each step, which discourages overfitting.

**Q: What is a learning-rate schedule, and why cosine?**
It changes the learning rate during training. Cosine decay starts large and shrinks smoothly to about zero, so early steps move fast and late steps fine-tune.

**Q: Why batch size 16?**
It was the Stage 2 setting. We tested 8, 16 and 32 and got 0.0222, 0.0219 and 0.0221, so it does not matter here.

**Q: Would a larger batch increase the error?**
Not in our runs; the differences are within noise. In general, a larger batch gives smoother gradients but fewer updates per epoch, and the effect depends on the learning rate.

**Q: What is an epoch versus a step?**
An epoch is one pass over the 900 training samples (about 57 steps at batch 16). A step is one weight update. We fix the number of steps so models get the same budget.

**Q: How many steps did you train?**
5,700 steps for the recipe study (equal to the Stage 2 budget of 100 epochs), 20,000 for the sweeps, and 60,000 for the final models.

**Q: How did you pick the best checkpoint?**
By the best validation error during training. The test set was not used for this.

**Q: How did you tune hyperparameters?**
One at a time, on validation error: modes, width, depth, learning rate, weight decay, batch size and training length. We tuned the MLP the same way.

**Q: Why are many settings flat?**
Because the model is robust here. The training recipe (loss, schedule, augmentation) mattered far more than these settings.

**Q: Why 5 seeds, and what does "+-" mean?**
A seed sets the random initial weights and batch order. Different seeds give slightly different results, so we report the mean and the standard deviation across 5 runs.

**Q: What is data augmentation, and why is yours valid?**
Creating extra training examples by transforming the existing ones. With a constant source and zero boundary, the PDE is exactly equivariant under the 8 rotations and flips of the square (the D4 group), so a transformed kappa has the correspondingly transformed solution.

**Q: Why augment before subsampling?**
Rotating an already-subsampled grid shifts which pixels are sampled, so it would not be a valid transformation. We rotate the 128 x 128 data first.

**Q: What is test-time averaging?**
Averaging the prediction over symmetric versions of the input. On the 64 x 64 grid only identity and transpose are exactly consistent, so we average those two.

**Q: How do you prevent overfitting?**
A held-out validation set for checkpoint selection, weight decay, augmentation, and the final test set for honest scoring.

**Q: Did you use gradient clipping or mixed precision?**
No gradient clipping. TF32 gave no speed-up, and bfloat16 is not supported by PyTorch for the complex FFT operations, so we did not use mixed precision.

**Q: Why is mean error 0.022 but median 0.009?**
A few hard samples (nearly uniform permeability with large pressure) have large errors and pull the mean up. Most samples are predicted very well.

**Q: Why is validation error higher than test error?**
Only 100 validation samples, so the average depends heavily on a few hard cases. It is noise, not a sign of a problem.

---

## B4. Concept questions

**Q: What is a neural network layer?**
A weighted sum of the inputs plus a bias, followed by a nonlinearity.

**Q: Why do you need an activation function?**
Without it, stacked layers collapse into a single linear map. The nonlinearity (GELU here, a smooth version of ReLU) lets the network represent curved relationships.

**Q: How many parameters does the MLP have and where do they come from?**
100.7 million: the input layer 12,288 x 4,096 is about 50 million, the two hidden layers are 4,096 x 4,096 each, and the output layer is 4,096 x 4,096.

**Q: Why does the FNO have fewer parameters?**
Its learned weights live on only 12 x 12 frequency modes per layer (64 x 64 channel mixing x 12 x 12, twice), not on every pixel. About 1.2 million per layer, 9.5 million in total.

**Q: What is the Fourier transform doing?**
It rewrites an image as a sum of waves of different frequencies. Low frequencies are smooth shapes, high frequencies are fine detail.

**Q: What are Fourier modes, and why keep only 12?**
Modes are the frequency components. The pressure solution is smooth, so most information is in the lowest ones. Keeping 12 gave the same accuracy as 32 at a fraction of the cost.

**Q: Why multiply in frequency space?**
By the convolution theorem, a convolution that spans the whole image (every pixel affecting every pixel) becomes a simple multiplication of frequency components, which is cheap.

**Q: Why does the FNO also have a 1 x 1 skip path?**
The Fourier part keeps only smooth information and assumes periodicity. The pointwise path restores local detail and helps handle the non-periodic boundary.

**Q: What is the inductive bias of an MLP versus an FNO?**
An MLP has almost none (it does not know about space). An FNO bakes in the idea that the output depends on the whole input through a translation-invariant operator, which suits PDEs.

**Q: What does "resolution invariant" mean, and is your FNO?**
In principle the same weights can be applied to any grid size. In our tests it was only partly true: a model trained on 64 x 64 was 2 to 3x worse on other grids.

**Q: What is the universal approximation theorem, and why does the MLP still fail?**
A big enough MLP can approximate any function in theory, but that says nothing about how much data and training it needs. Without structure, it needs far more of both.

**Q: What is the finite-volume method?**
Divide the domain into cells and enforce conservation of flow in each cell, with fluxes across cell walls computed from the pressure difference and permeability. It gives a sparse linear system A u = b.

**Q: Why a sparse matrix?**
Each cell only interacts with its 4 neighbours, so each row of A has about 5 non-zero entries out of 4,096.

**Q: How is the linear system solved?**
Either a direct method (sparse LU factorisation, which we used for timing) or an iterative one (algebraic multigrid).

**Q: What does "calibrated" solver mean?**
We adjusted the solver's face averaging and boundary placement until it matched the dataset's reference arrays. The data behaves as if the zero-pressure boundary is about one cell outside the grid.

**Q: What is the PDE residual?**
Plug the predicted pressure into the discretised equation. The leftover (how far it is from satisfying it) is the residual. Smaller is more physical.

**Q: Why is the residual large even for the truth?**
The truth is a 128 x 128 solution shrunk to 64 x 64, so it does not satisfy the 64 x 64 equation exactly. What matters is that the prediction's residual is about the same as the truth's.

**Q: What is relative error versus absolute error?**
Absolute error is just ||u_hat - u||. Relative divides by ||u||, which makes samples with different pressure sizes comparable.

**Q: What is L2 norm?**
The square root of the sum of squared entries: the usual notion of size.

**Q: Bias versus variance, applied here?**
The MLP has high bias for this problem (wrong structure), so more data does not fix it. The FNO has the right bias, so it learns from few samples.

---

## B5. Results and analysis questions

**Q: What are your main results?**
Final FNO 0.0202 +- 0.0004 versus MLP 0.0415 +- 0.0009 (5 seeds). The recipe change halved the error. The calibrated solver at 64 x 64 is 0.0438.

**Q: What mattered most?**
The training recipe, not the architecture.

**Q: Is the FNO really better than the solver?**
At the same 64 x 64 resolution it agrees with the dataset 2.2x better, and it is faster on a GPU. But that is partly because it learns the dataset's own quirks, and on one CPU core it is not faster.

**Q: Why not faster on a CPU?**
It does large FFTs and many wide layers (37 ms), while the solver does one small sparse solve (7 ms) at 64 x 64.

**Q: When does the FNO pay off?**
When many solves are needed, on a GPU, in batches. Break-even is about 50,000 solves.

**Q: What fails?**
Nearly uniform permeability with large pressure: right shape, wrong amplitude (up to 36% error).

**Q: Why is the data-scaling curve flat after about 5,000 samples?**
The error reaches a floor set by the model and training budget and by the intrinsic difficulty of the hardest cases.

**Q: Why does the error rise with a different grid?**
Our interpretation: the sharp kappa edges and the boundary are sampled differently on a new grid, so the learned map is not exactly grid-independent.

---

## B6. Tricky and honesty questions

**Q: You said in Stage 2 the FNO was 1,000x faster. What happened?**
The reference solver we compared against was inefficient. We built a fair, calibrated solver, found the real speed-up is 5 to 35x on a GPU and none on one CPU core, and withdrew the old claim in the final report.

**Q: Did you really run everything on Colab?**
The 142 training runs of the full study ran on a lab GPU. The notebook ran end to end on Google Colab (T4, about 50 minutes) and Kaggle (T4 x2, about 35 minutes), and its Section 10 loads the stored results instead of recomputing them. The notebook states this.

**Q: Did you use AI tools?**
Yes: for coding, debugging and drafting, as declared in the contribution statement. All numbers come from real runs on real data, and we verified them against the stored result files.

**Q: Who did what?**
Be ready to explain the parts listed against your name in the contribution statement and to say what your teammates did.

**Q: Why should we trust these numbers?**
They come from stored result files, with 5 seeds for the final models, a held-out test set never used for selection, and the code and notebook are reproducible.

**Q: What would you do differently or next?**
Other PDEs and source terms, a physics-informed loss term, more resolutions with proper multi-resolution training, and uncertainty estimates.

---

# PART C: GLOSSARY AND CHEAT SHEET

**Glossary**
- **Surrogate:** a fast learned stand-in for a slow simulator.
- **Operator:** a map from functions to functions.
- **PDE:** partial differential equation.
- **Epoch / step / batch:** a pass over the data / one weight update / examples per update.
- **Loss:** a number measuring how wrong a prediction is, which training minimises.
- **Overfitting:** memorising the training set.
- **Inductive bias:** structure built into a model that fits the problem.
- **FFT:** fast Fourier transform.
- **Mode:** one frequency component.
- **D4:** the 8 rotations and flips of a square.
- **Sparse matrix:** a matrix that is mostly zeros.
- **Residual:** leftover when you plug a candidate answer into an equation.

**Numbers to remember**
- Data: 10,000 real samples, 128 x 128 shrunk to 64 x 64; split 900 / 100 / 200; seed 42.
- Equation: -div(kappa grad u) = f = 1, u = 0 on the edges, kappa in {0.1, 1.0}.
- Stage 2: FNO 0.052, MLP 0.08.
- Recipe fix (same budget): FNO 0.0524 -> 0.0249, MLP 0.0802 -> 0.0454.
- Final (5 seeds, 60,000 steps): FNO 0.0202 +- 0.0004 with 9.5 M parameters, MLP 0.0415 +- 0.0009 with 100.7 M.
- Chosen FNO: 12 modes, width 64, 8 layers. Training: AdamW, lr 1e-3, weight decay 1e-4, batch 16, relative L2 loss, cosine decay, D4 augmentation.
- Data scaling (FNO): 400 samples 0.026; plateau near 0.020. The MLP stays near 0.036.
- Solver error: 0.0939 / 0.0438 / 0.0241 at 32 / 64 / 128; direct-solve times 1.8 / 7.1 / 37.4 ms.
- FNO at 64 x 64: 0.020; 1.4 ms (batch 1 GPU), 0.20 ms per sample (batch 128); 5x to 35x faster on GPU; not faster on one CPU core (37 ms versus 7 ms).
- Physics: mean pressure error 1.7%, boundary 0.1157 versus 0.1159, negative pixels 0.0015%, residual 4.245 versus 4.010.
- Marking: report 10, implementation 8, comparison 7, metrics 7, physics 7, reproducibility 4, viva 5, presentation 2.
