# Redesigning GWPINN: which physics-informed formulation for this benchmark?

This document is the *design* half of the exercise: what the state of the art in
physics-informed machine learning actually offers, which parts of it address the
five things this benchmark measures, and how the redesigned `gwpinn` is
factored so those parts can be compared rather than argued about.

The *results* half — what actually won — is in
[`pinn_formulation_results.md`](pinn_formulation_results.md), generated from
`runs/pinn_formulation_study/results.csv`.

---

## 1. What the benchmark is asking for

The MODFLOW 6 benchmark (`scripts/build_mf6_hetero_benchmark.py`) was built to
be hostile in five specific ways, and each one breaks a different assumption
that a vanilla PINN quietly makes.

| Requirement | What the benchmark does | Why a vanilla PINN struggles |
|---|---|---|
| **Head simulation** | 2-hourly forcing, M2 tide at 12.4 h, storms, flash floods | Spectral bias: a tanh MLP fits low frequencies first, and the tide is near Nyquist for the stress-period clock. All-at-once training also violates temporal causality — the loss is happy to fit t=4 d before t=0.1 d. |
| **Head across faults** | Two HFB barriers, `hydchr` 1e-8 and 1e-3, mean head jump **2.46 m** across a single 100 m cell face | A continuous network cannot represent a discontinuity. Worse, the strong form needs *second* derivatives of a field that is nearly discontinuous. |
| **Mass balance** | Reference budget closes to ~1e-7 relative | Collocation enforces the PDE at sampled points only. Nothing links neighbouring points, so there is no telescoping and no discrete conservation law. |
| **River exchange** | Meandering 139-cell RIV boundary, tidal stage, conductance-limited switch at `rbot` | The source is a piecewise-constant cell indicator on a one-cell-wide meander, plus a non-smooth `max(h, rbot)` kink. |
| **Inverse (K)** | log-normal K, var(lnK)=2, correlation length 200 m = 2 cells | Only `div(T grad h)` is observable. The null space is enormous and what fills it is the parameterisation's implicit prior — which for a Fourier-featured MLP is not a geostatistical statement. |

The existing `gwpinn` already handles several of these thoughtfully (variogram
and kriging priors for the inverse problem, smeared anisotropic barriers,
gradient-norm balancing). What it lacks is (a) anything that gives *discrete
conservation*, (b) anything that respects *temporal causality*, and (c) a
formulation in which a head discontinuity is natural rather than fought against.

---

## 2. The state of the art, and what each part is good for

### 2.1 Architecture — fixing optimisation, not physics

| Method | Idea | Relevance here |
|---|---|---|
| **Fourier features** (Tancik et al. 2020) | Embed `x → [sin Bx, cos Bx]` | Necessary. Without it the network cannot resolve a head field shaped by a 2-cell correlation length K field. |
| **SIREN** (Sitzmann et al. 2020) | Sine activations with principled init | Similar effect; higher derivatives stay well behaved, which matters for the strong form. |
| **Modified MLP** (Wang, Teng & Perdikaris 2021) | Two global encoders gate every layer | Cheap, consistently better than a plain MLP on stiff problems. This is `gwpinn`'s current best coordinate backbone. |
| **PirateNet** (Wang et al. 2024) | Adaptive residual blocks with zero-initialised skips + random weight factorisation | Current SOTA for *deep* PINNs. Starts as a shallow linear map and recruits depth as needed, so depth stops hurting. |
| **Random weight factorisation** (Wang, Sankaran, Wang & Perdikaris 2023) | `W = diag(exp(s)) V` | Free per-neuron learning rates; a drop-in accelerator for any backbone. |
| **Separable PINN** (Cho et al. 2023) | One MLP per axis, combined by tensor product | Cost becomes additive rather than multiplicative in axis resolution. Attractive when the time axis needs hundreds of samples — as it does here. But it imposes a low-rank separable prior, and a meandering river and an oblique fault are precisely *not* separable. Genuinely uncertain which way this cuts, hence worth measuring. |

**Verdict going in:** architecture fixes optimisation pathologies. It does not
create conservation, and it does not create discontinuities. Necessary, not
sufficient.

### 2.2 PDE formulation — where conservation and discontinuities live

This is the axis the literature treats as most consequential for exactly the
properties this benchmark measures.

- **Strong form** (Raissi et al. 2019). Residual written directly, two
  derivatives of the head net. Simple, mesh-free, no conservation guarantee.
- **Mixed / dual form** (deep mixed residual methods, e.g. Lyu et al. 2022).
  Introduce the Darcy flux `q` as an independent output; split into a
  constitutive law `q = −T A ∇h` and continuity `S ∂h/∂t + ∇·q = f`. Only first
  derivatives. Crucially the *flux* becomes the represented quantity — and
  across a fault, flux is continuous while head is not. The representation
  finally matches the physics.
- **Weak / variational form** (VPINN, hp-VPINN; Kharazmi, Zhang & Karniadakis
  2019, 2021; Deep Ritz, E & Yu 2018). Integrate against test functions;
  integration by parts removes a derivative and gives element-level balance.
- **Control-volume / finite-volume-informed.** Keep the network continuous but
  evaluate the balance over the benchmark's own control volumes, with harmonic
  face conductances and barriers in series on the faces they block. Because each
  face flux enters two cells with opposite sign, the residual vanishing implies
  **local conservation** — the thing collocation cannot give at any finite number
  of points. It also lets the fault be applied exactly where MODFLOW applies it.
- **cPINN / XPINN** (Jagtap, Kharazmi & Karniadakis 2020; Jagtap & Karniadakis
  2020). Decompose the domain and impose flux continuity at interfaces. This is
  the textbook answer for a discontinuity: put the interface *on* the fault and
  let head jump across it. The cost is bookkeeping — with two oblique faults
  cutting an irregular domain, the subdomains are awkward — which is why the
  mixed and FV forms are tested first as cheaper routes to the same property.

### 2.3 Temporal strategy

- **Causal weighting** (Wang, Sankaran & Perdikaris, *Respecting causality is all
  you need for training PINNs*, 2022/24). Weight time bin *i* by
  `exp(−ε Σ_{j<i} L_j)`, so a bin is only unlocked once its predecessors have
  converged. Directly targets the failure mode of all-at-once transient training.
- **Time marching / expanding windows.** Cruder, but it also bounds the
  space-time volume that must be fitted at once.
- **Discrete-time PINNs** (Runge–Kutta collocation in the original PINN paper).
  Attractive here because the benchmark already *has* a natural time
  discretisation — 48 stress periods — though it changes the model from a
  space-time surrogate into a stepper.

### 2.4 Loss balancing

`fixed` → **gradient-norm balancing** (Wang et al. 2021) → **NTK balancing**
(Wang, Yu & Perdikaris 2022) → **self-adaptive weights** (McClenny &
Braga-Neto) and **residual-based attention** (Anagnostopoulos et al. 2024). The
last is different in kind: it reweights *within* the PDE term, per collocation
point, which on this problem means automatically concentrating effort on the
fault zones and the river.

### 2.5 Inverse parameterisation

This, not the network, is what determines inverse quality.

- **Network parameterisation** — flexible, prior is whatever an MLP finds easy.
- **Pixel/grid + edge-preserving regularisation** — honest about the
  dimensionality; the pseudo-Huber TV penalty keeps facies contacts.
- **Truncated Karhunen–Loève / principal-component parameterisation**
  (the classical geostatistical inverse approach, Kitanidis 1995). For a
  *stationary* covariance the KL modes on a periodic box are exactly the Fourier
  modes with eigenvalues equal to the spectral density — so the expansion is
  available in closed form from the variogram, with no 10⁴×10⁴
  eigendecomposition. Reduces 10,000 unknowns to a few hundred **and** forces
  every reachable field to have the right spatial statistics.
Groundwater-specific precedent for the PINN inverse problem is Tartakovsky et
al. (2020, *WRR*), which recovers conductivity and constitutive relationships in
subsurface flow — but on smooth fields, without barriers, and in steady state.

- **Ensemble methods (ES-MDA / EnKF)** and **B-PINNs** (Yang, Meng &
  Karniadakis 2021) for uncertainty — not tested here, but the KL
  parameterisation is exactly the interface they would plug into.

### 2.6 What was deliberately *not* tested, and why

**Neural operators** — FNO (Li et al. 2021), DeepONet (Lu et al. 2021) and
physics-informed DeepONets (Wang, Wang & Perdikaris 2021) — are the strongest
tools for the *forward* problem when you need many solves for many parameter
fields, and the benchmark's GeoTIFF exports were designed with them in mind.
They are out of scope here because they answer a different question: they need a
*dataset* of solved cases to amortise over, whereas this study asks which
formulation best solves *one* case from physics plus sparse data. An operator
learned over K-field realisations is the natural follow-up, and the multi-band
head stacks exported by the benchmark are the training set for it.

---

## 3. The redesign

The redesigned code factors a formulation into six independent axes, so a
change in a metric can be attributed to the axis that moved. Comparing six
named methods from six papers confounds architecture with formulation with
schedule and teaches you very little.

```
gwpinn/
  benchmark/mf6case.py     the reduced case + its MODFLOW reference
  formulations/
    problem.py             geometry, forcing, faults, sampling  (shared physics)
    arch.py                mlp | modified | pirate | spinn   (+ Fourier, RWF)
    residuals.py           strong | mixed | fv
    inverse.py             true | net | grid | kl
    strategy.py            fixed|gradnorm|ntk|rba  ×  plain|causal|march
    runner.py              Spec -> composed model -> trained model
  eval/criteria.py         the five criteria, all "lower is better"
```

Held fixed across every variant so they never become confounders: the hard
initial condition `h = h_init + s(t)·hs·NN` (so `h(t₀) = h_init` exactly and no
IC weight has to be chosen), the normalisation and residual scaling, the
collocation budget and resampling period, and the optimiser schedule.

### Three details that are easy to get wrong

**Coefficients must be differentiable in space.** `div(T ∇h)` expands to
`∇T·∇h + T ∇²h`. Sampling K, the layer top and the layer bottom by nearest cell
makes `∇T ≡ 0` and silently deletes a term — the surrogate would then be solving
a *different equation* from the reference and would be blamed for the
discrepancy. Coefficients are interpolated bilinearly for the pointwise
formulations; the FV form deliberately uses cell averages, because that is what
its face conductances are built from and what MODFLOW uses.

**The barrier width is calibrated, not tuned.** MODFLOW's HFB gives a face a
conductance `hydchr·b·w_face` in series with the cells. The continuous analogue
is a zone of width `w` with normal conductivity `K_n = hydchr·w`, so the
anisotropy multiplier is `α = hydchr·w/K`. That is ~2e-7 for the tight fault and
~2e-2 for the leaky one — three orders of magnitude apart, as they should be.

**Sources use the exact cell indicator.** The river is a meandering 139-cell set
and MODFLOW applies its conductance to those cells and nothing else. Smearing it
into a Gaussian ridge would make every variant carry the same smearing error and
the river-exchange criterion would be measuring my smoothing kernel.

---

## 4. How it is scored

Against the MODFLOW reference, which solves the *same* equation on the same grid
and closes its budget to ~1e-7 relative — so every error reported belongs to the
surrogate.

| Criterion | Headline metric | Note |
|---|---|---|
| head | RMSE (m) over all cells and times | plus NSE and a transient-amplitude error, because a model can score a fine RMSE while capturing none of the dynamics |
| fault | RMSE of the head *jump* across blocked faces | reported separately for the tight and leaky faults; a "recovery fraction" says how much of the true 2.46 m jump survived |
| mass | local imbalance fraction, evaluated with the conservative FV operator | a strong-form PINN does not get to grade its own homework with its own collocation points |
| river | relative error in total exchange | plus per-cell RMSE, NSE and sign agreement |
| inverse | log₁₀ RMSE of K | plus pattern correlation and a *variogram* error, because a smooth field can have a decent RMSE and completely the wrong texture |

**Every metric is also applied to the MODFLOW reference itself** to establish its
floor. The river criterion floors at 6e-7 (i.e. it is exact); the mass criterion
floors at **0.079**, because the reference is stored at stress-period ends so
`dh/dt` is reconstructed piecewise-linearly and the storage switch is smoothed.
A surrogate reaching 0.079 is perfect as far as this study can tell — without
that number the mass column would be uninterpretable.

## 5. Limitations, stated up front

- **Equal wall-clock, not convergence.** Per-iteration cost varies four-fold
  across these variants, so every run gets the same budget (240 s on 4 CPU
  threads) and iterations completed is reported as a result. This measures which
  formulation gets furthest per unit compute — a real and practical question —
  but it is *not* asymptotic accuracy. A method that starts slowly and finishes
  strongly is ranked unfairly; `--budget` exists to test that.
- **Screening uses one seed**, with the leaders repeated across seeds.
- **The case is a single layer.** Nothing here tests vertical discretisation or
  the quasi-3D leakage terms of the full five-layer benchmark.
- **No uncertainty quantification.** Deep ensembles and B-PINNs are the obvious
  extension and the KL parameterisation is the natural interface for them.
