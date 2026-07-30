# Scaling to 3D XPINN + mixed formulation: evaluation, test plan, complexity

> The `<formulation>` block in the request was an unfilled placeholder. This
> critiques the formulation **as built in this repository** (stated in §0.1)
> against the target described in the request: full 3D XYZ, multilayer,
> XPINN/cPINN domain decomposition, mixed `(h, q)` output.

All quantitative statements are measured on the committed test dataset
(`scripts/make_sample_data.py`, 9700 × 7650 m, two layers, 59 wells, 6 gauges,
28 pumping tests, 2 faults) or benchmarked on this machine (4 CPU threads,
float64, width 96 / depth 4).

---

## 0. Baseline: where the current formulation actually stands

### 0.1 What is implemented

| aspect | as built | target |
|---|---|---|
| dimensionality | quasi-3D: depth-integrated 2D per layer, coupled by leakance $C_{l,l+1}$ | full 3D XYZ |
| formulation | **primal**, $h$ only; $\mathbf q$ derived post hoc | **mixed** $(h,\mathbf q)$ |
| decomposition | single global network + Fourier features | XPINN / cPINN |
| faults | smeared anisotropic tensor $A = I-(1-\alpha)\psi\,\mathbf{nn}^T$, optional `tanh` coordinate map | explicit interface conditions |
| free surface | Boussinesq $T_0 = K_0(h_0-z^{bot}_0)$, smoothly clamped | *unaddressed in the request* |

Governing residual currently minimised, per layer $l$:

$$\mathcal R_l = \nabla\!\cdot\!\big(T_l A \nabla h_l\big) + \textstyle\sum C_{l\pm}(h_{l\pm}-h_l) + R_l + C_{riv}\mathbb 1_{riv}(h_{riv}-h_l) - S_l\partial_t h_l$$

### 0.2 Measured performance on the test dataset

| quantity | n | value |
|---|---|---|
| head, held-out wells | 12 | RMSE 0.81 m, R² 0.947 |
| head field vs reference, L0 / L1 | 28,010 | RMSE 1.08 / 1.07 m, R² 0.870 / 0.868 |
| log10 T field vs reference, L0 | 28,010 | RMSE 0.611, **R² −0.97** |
| Moran's I of head residuals | 59 | 0.034 (p = 0.36) — unstructured |
| impermeable fault step recovered | — | **12%** (72% with coordinate mapping, at cost) |
| riverbed conductance $C_{riv}$ | — | 0.0038 vs 0.05 true (**13× low**) |
| vertical leakance $C_{01}$ | — | 1.9e-4 vs 1e-3 true (**5× low**) |

**Three failures carry directly into the 3D design and should drive it:**

1. **Barriers are under-resolved** (12% of throw). This is a *representational*
   limit — the fault sits in the interior of a smooth network.
   **XPINN fixes this structurally**: put the fault *on* the interface and the
   jump is imposed by a constraint, not approximated by a basis. This is the
   single strongest argument for the proposed decomposition.
2. **Lumped conductances are non-identifiable** ($C_{riv}$, $C_{01}$ both an
   order out while heads stay accurate). Adding a third dimension and more
   interfaces *increases* the number of such lumped parameters. Fix the ones
   you can measure; do not train them all simultaneously.
3. **$T$ has negative R².** Texture is right, pointwise skill is not. 3D will
   make this worse, not better — the same well count now has to constrain
   $K(x,y,z)$.

---

## 1. Test Case Generation

Four cases, each isolating one new failure mode. Every case ships with a
finite-difference / analytic reference so error is measured, not inspected.

### TC-1 — 2D homogeneous, one straight impermeable fault, analytic reference

**Purpose:** verify interface conditions and the mixed formulation in isolation.

- **Domain** 1000 × 1000 m, $K = 10$ m/d, $b = 10$ m. Fault at $x=500$,
  full height, conductance $C_f$.
- **Decomposition** $N_{sub}=2$, interface **on** the fault.
- **BCs** $h=50$ at $x=0$, $h=30$ at $x=1000$; no-flow on $y=0,1000$. No recharge.
- **Analytic reference** 1D three-resistor series:
  $q = \Delta h\big/\big(\tfrac{L_1}{T}+\tfrac{1}{C_f}+\tfrac{L_2}{T}\big)$,
  so the step across the fault is exactly $q/C_f$. Sweep
  $C_f \in \{10^{-3},10^{-2},10^{-1},1\}$ d⁻¹ to trace the barrier→transparent limit.
- **Collocation** 1024 interior per subdomain, uniform (the solution is linear —
  no refinement needed). 256 interface points, **uniform along the trace**.
  256 boundary points. Resample every 100 iterations.
- **Acceptance** relative error on the step $<2\%$ across all four $C_f$;
  flux continuity $|\mathbf q_1\!\cdot\!\mathbf n - \mathbf q_2\!\cdot\!\mathbf n| / q_\star < 10^{-3}$.

> This case will expose a **gauge failure** the moment you set $C_f=0$: the two
> subdomains decouple and each becomes pure-Neumann unless it retains a Dirichlet
> edge. See Blindspot B4.

### TC-2 — 2D heterogeneous, two faults + river Robin (the current dataset)

**Purpose:** first case with a real inverse problem; direct comparison against
the measured baseline in §0.2.

- **Domain / data** exactly the committed dataset — heterogeneous $K$ from a GRF
  with known variogram, 1 impermeable + 1 leaky fault, meandering river.
- **Decomposition** $N_{sub}=4$–6. Faults **partition** the domain; artificial
  interfaces close the partition where fault traces do not reach the boundary.
- **BCs** no-flow outer boundary; river as a Robin **boundary** on the wetted
  polygon perimeter (not a distributed source — see B6); recharge $1.2\times10^{-4}$ m/d.
- **Collocation** per subdomain, area-weighted, total ≈ 4096 interior. Plus:
  - **river band**: 768 points within $3\delta_{riv}$ of the wetted polygon,
    where $\delta_{riv}=\sqrt{T/C_{riv}} = 68$ m (measured) — this is the
    boundary layer and uniform sampling misses it;
  - **interface**: 128 points per interface, sampled by arc length;
  - **control volumes**: 64 random disks of radius $2$–$10\,\delta_{riv}$ for the
    mass-balance constraint (B3).
- **Acceptance** held-out head RMSE $\le$ 0.81 m (baseline parity) **and** fault
  step recovery $>60\%$ **without** the global degradation the coordinate map
  caused (head-field R² must stay $>0.80$; the mapped run fell to 0.55).

### TC-3 — 3D single confined layer, one fault, river

**Purpose:** introduce the vertical dimension, anisotropy, and 3D interfaces —
**confined only**, so the free surface is deliberately excluded.

- **Domain** 9700 × 7650 × 67 m (aspect **145:1**, measured). Confined:
  top and bottom are no-flow, $h$ is not the free surface.
- **Anisotropy** $K_h/K_v \in \{1, 10, 100\}$ as a swept parameter, with the
  stretched vertical coordinate of B2.
- **Decomposition** $N_{sub}=4$ laterally; **no vertical splitting** (the layer
  is thin — a vertical interface would be almost all interface).
- **BCs** no-flow top/bottom/outer; river Robin on the streambed surface
  (a *surface* in 3D, not an area-source); recharge as a Neumann flux on the top.
- **Collocation** 17,000 interior at $\Delta z = 10$ m and 165 m horizontal
  (measured scaling: **6.7×** the 2D count). Stratified: equal counts per
  vertical decile so the thin dimension is not starved. 1024 streambed,
  1024 top-surface recharge, 512 per interface.
- **Acceptance** vertical head gradient within 5% of the FD reference at
  $K_h/K_v=100$; mass balance closure $<1\%$ over the whole domain.

### TC-4 — Full 3D multilayer + rivers + faults + free surface

**Purpose:** the target configuration.

- **Domain** as TC-3 with 2–3 hydrostratigraphic units as *material zones*, not
  separate 2D models. Aquitards are thin low-$K$ volumes.
- **Decomposition** $N_{sub}=8$–16, lateral partition by fault traces, **plus a
  coarse global network** (B5). Do not decompose vertically.
- **Free surface** unconfined top: iterate the phreatic surface (B1).
- **BCs** free-surface condition on $\Gamma_{ws}$; river Robin; no-flow base;
  no-flow lateral.
- **Collocation** 35,000+ interior ($\Delta z=5$ m), 2048 free-surface, 2048
  streambed, 512 per interface, 256 control volumes. **Adaptive (RAR)**
  refinement on residual magnitude after iteration 2000.
- **Acceptance** held-out head RMSE, per-compartment mass balance $<2\%$, and
  fault step recovery $>70\%$ *simultaneously*.

---

## 2. Complexity & Feasibility Analysis

### 2.1 Cost model, measured

Per-iteration cost $\propto N_c\,(P_{fwd} + n_{bwd}P_{bwd})$ with
$n_{bwd} = 1 + d$ for **both** formulations ($1$ for $\nabla h$, $d$ for the
divergence). The formulations differ in the *cost of each backward pass*: the
primal divergence differentiates $\nabla h$, building a second-order graph;
the mixed divergence differentiates the network output $\mathbf q$ directly.

Benchmarked (width 96, depth 4, float64, 4 CPU threads, warm-started):

| case | primal (2nd-order) | mixed (1st-order) | speedup |
|---|---|---|---|
| 2D, N=3,000 | 66.3 ms | 29.3 ms | **2.26×** |
| 2D, N=12,000 | 224.4 ms | 125.5 ms | 1.79× |
| 3D, N=3,000 | 73.5 ms | 37.6 ms | 1.95× |
| 3D, N=12,000 | 286.8 ms | 161.6 ms | 1.78× |

**The mixed formulation is the right call and buys ~1.8–2.3×**, plus it removes
the second-derivative conditioning problem entirely. It is what makes 3D
tractable at all on modest hardware.

**XPINN cost is $O(N_c)$, not $O(N_c N_{sub})$** — collocation points are
*partitioned*, so each point hits exactly one subnetwork. Adding subdomains is
nearly free in FLOPs. The cost is interface points, which scale as
$N_{iface} \propto V/H$ — for TC-4 with $N_{sub}=16$, interfaces are ~5–8% of
total points.

Projected wall clock (mixed, CPU / single modern GPU):

| case | $N_c$ | ms/iter (CPU) | 5k iters (CPU) | GPU estimate |
|---|---|---|---|---|
| TC-1 | 2.6k | ~30 | 2.5 min | seconds |
| TC-2 | 5.1k | ~60 | 5 min | < 1 min |
| TC-3 | 19k | ~250 | 21 min | 1–2 min |
| TC-4 | 40k | ~530 | 44 min (15k iters ≈ 2.2 h) | 5–10 min |

### 2.2 Spectral bias at fault and river interfaces

Measured on this dataset, with $L_\star = 4850$ m (normalisation half-diagonal):

| feature | width | $\sigma$ needed | $\sigma$ used | shortfall |
|---|---|---|---|---|
| river boundary layer $\sqrt{T/C_{riv}}$ | 68 m | 36 | 6 | **6×** |
| fault barrier $w$ | 60 m | 40 | 6 | **6.7×** |

- **The fault problem is solved by the decomposition.** With the fault on an
  interface, the jump is a constraint; neither subnetwork has to represent a
  steep gradient. Expect fault-step recovery to jump from the measured 12% to
  near-exact — this is the main payoff of XPINN here.
- **The river problem is not.** $\Gamma_{riv}$ is a Robin boundary, not an
  interface: the boundary layer lives *inside* one subdomain and still needs
  resolving. Raising $\sigma$ globally is not the answer — the measured
  consequence of an over-wide band is property-field overfitting (this repo's
  `fault_sigma_scale` experiment: full bandwidth on a steep coordinate
  **diverged**, PDE loss $2.5{\times}10^3 \to 1.5{\times}10^5$).
  **Use a locally-supported high-frequency band**: a second Fourier bank gated
  by $\exp(-d_{riv}^2/(2\delta_{riv})^2)$, or a boundary-layer coordinate
  $\xi = 1-e^{-d_{riv}/\delta_{riv}}$.
- **Mixed formulation helps here too**: $\mathbf q$ is smoother than $\nabla h$
  near a Robin boundary, and it is $\mathbf q$ that the continuity residual
  differentiates. The stiff object is moved from the derivative into a network
  output.

### 2.3 Loss balance

Five families with **different physical units** — this is the most common way a
mixed XPINN silently fails. Non-dimensionalise *before* adaptive weighting:

$$q_\star = K_\star H_\star / L_\star,\qquad
\mathcal R^{c} = \frac{\mathbf q + \mathbf K\nabla h}{q_\star},\qquad
\mathcal R^{m} = \frac{\nabla\!\cdot\!\mathbf q - f}{q_\star/L_\star}$$

Measured pathology in the current code, to avoid repeating: the textbook
gradient-norm rule $\lambda_i = \max|\nabla_\theta\mathcal L_{pde}| \big/
\overline{|\nabla_\theta \mathcal L_i|}$ **saturated 5 of 8 weights at the
ceiling** because $\max/\overline{\;\cdot\;}$ over $10^5$ parameters is itself
$O(10^2$–$10^3)$. Use mean/mean with a ceiling, or NTK weighting
(Wang, Yu & Perdikaris 2022). Additionally:

- **Interface losses must not be adaptively down-weighted.** They are
  *constraints*, not objectives. Use a fixed large weight or an augmented
  Lagrangian: $\lambda \leftarrow \lambda + \rho\,\mathcal I$.
- **Noise-floor the data terms.** Measured: removing the floor let the network
  fit 47 wells to 0.041 m (below the 0.05 m measurement noise) and *raised*
  error between wells. This effect was larger than any architecture difference.

### 2.4 Expected bottlenecks — and one the request mis-identifies

**Memory is not the bottleneck.** For TC-4, $N_{sub}=16$ × 100k params ×
8 bytes × 3 (Adam) ≈ **38 MB**. Activation memory at $N_c=40$k, width 96,
depth 4, first-order graph ≈ **300–400 MB**. Both are 1–2 orders of magnitude
below a 16 GB budget. Domain decomposition *reduces* memory pressure by
partitioning points.

**The real bottleneck is that XPINN has no coarse space.** Steady groundwater
flow is elliptic; information is global. XPINN/cPINN is a one-level Schwarz
method, and classical DD theory gives condition number $\kappa = O(1/H^2)$
without coarse correction, with information propagating one subdomain per
"sweep". Going from $N_{sub}=2$ (TC-1) to 16 (TC-4) should be expected to
*degrade* convergence per iteration, not improve it. This is the documented
failure mode of XPINN on elliptic problems and the reason FBPINNs
(Moseley, Markham & Nissen-Meyer 2023) added overlapping subdomains, a partition
of unity, and **multilevel** variants.

Secondary: **vanishing gradients are not the issue** (depth 4, tanh, residual
connections available); **stiffness is**. The measured 5-orders-of-magnitude
initial PDE residual when a steep coordinate map was introduced is the
signature.

---

## 3. Comparative Matrix

| | **TC-1** 2D homog. 1 fault | **TC-2** 2D hetero. 2 faults + river | **TC-3** 3D confined + fault + river | **TC-4** 3D multilayer + free surface |
|---|---|---|---|---|
| **Configuration overhead** | Low — 2 subdomains, analytic reference, no geostatistics | Medium — 4–6 subdomains, fault-driven partition, existing data pipeline reused | High — 3D meshless partition, anisotropic scaling, 3D FD reference must be written | Very high — free-surface iteration, 8–16 subdomains + coarse net, RAR, per-compartment well-posedness checks |
| **Computational cost** | ~2.5 min CPU (5k iters, 2.6k pts) | ~5 min CPU (5k iters, 5.1k pts) | ~20 min CPU / 1–2 min GPU (19k pts) | ~2.2 h CPU / 5–10 min GPU (40k pts, 15k iters); GPU effectively required |
| **Expected mass-balance accuracy** | Excellent, <0.1% — analytic 1D, flux is an explicit output | Good, 1–3% — with control-volume constraints; 5–15% without | Good, 1–2% globally; per-compartment worse where interfaces are dense | Moderate, 2–8% — free-surface flux is the dominant error term; **will not close without explicit $\mathcal M_V$ constraints** |
| **Overall feasibility** | **Certain.** Do this first; it is a unit test, not research | **High.** Direct baseline comparison exists (0.81 m held-out). Main risk: interface partition where fault traces do not span the domain | **High but gated on B2.** Fails immediately if the 145:1 aspect ratio is normalised naively | **Medium.** Gated on B1 (free surface) and B5 (coarse space). Budget 2–3× the effort of TC-3 and expect the property field, not the head field, to be the limiting metric |

---

## 4. Implementation Blindspots

Ordered by how likely they are to prevent convergence.

### B1 — The free surface is not mentioned, and it is the hardest part of 3D

Quasi-3D hides the water table inside $T_0 = K_0(h_0 - z^{bot}_0)$. In a
continuous XYZ domain it becomes a **free boundary**: the top of the saturated
domain is itself unknown. Two options, both costly:

- **Variably saturated (Richards).** $\partial_t(\theta) = \nabla\cdot(K(\psi)\nabla(\psi+z))$.
  The wetting front is a moving near-discontinuity in $\theta(\psi)$ — the
  single most-reported PINN failure mode in unsaturated flow. Do not choose this
  unless the vadose zone is the object of study.
- **Sharp free surface with a fixed mesh-free domain (recommended).** Keep the
  domain fixed to the full stack and impose, on the phreatic surface $\Gamma_{ws}$:

$$h = z \quad\text{and}\quad \mathbf q\cdot\mathbf n = -R\,(\mathbf n\cdot\hat z) \quad\text{on }\Gamma_{ws}$$

  with $\Gamma_{ws}$ represented implicitly as the level set $\phi(x,y,z)=h-z=0$,
  and collocation points on it resampled every $N$ iterations by root-finding
  along vertical lines. This keeps the network fixed-domain and is the 3D
  analogue of what the Boussinesq term does in 2D.

**Corollary:** TC-3 should be confined precisely so that this risk is isolated
to TC-4.

### B2 — Aspect ratio 145:1 will destroy conditioning under naive normalisation

Measured: 9700 m lateral, 67 m saturated stack. Mapping each axis independently
to $[-1,1]$ multiplies $\partial/\partial z$ by 145 relative to
$\partial/\partial x$; in the continuity residual the vertical term is then
$\sim\!145$× mis-scaled ($145^2$ in a primal second-order form). Use the
**anisotropy-stretched vertical coordinate** that makes the operator isotropic:

$$\tilde z = z\sqrt{K_h/K_v},\qquad
\hat{\mathbf x} = (x, y, \tilde z)/L_\star$$

For $K_h/K_v = 100$ this converts 145:1 into ~14:1. Give $\hat z$ its **own
Fourier bandwidth** — vertical head structure is smooth in a confined unit but
sharp across an aquitard, and one shared $\sigma$ serves neither.

### B3 — A mixed formulation is not conservative unless you make it so

Outputting $\mathbf q$ does **not** give local mass conservation. $\nabla\cdot\mathbf q = f$
is enforced in a least-squares sense at collocation points only; between them,
flux does not balance. If mass balance is an acceptance criterion, add explicit
control-volume constraints on random volumes $V$:

$$\mathcal M_V = \frac{1}{|\partial V|}\oint_{\partial V}\mathbf q\cdot\mathbf n\,dS
- \frac{1}{|V|}\int_V f\,dV$$

evaluated by quadrature on sampled spheres/boxes. This is a *different*
constraint from the pointwise residual and catches exactly the error the
pointwise residual misses.

### B4 — Per-subdomain well-posedness: the gauge and compatibility trap

An impermeable fault ($C_f\to0$) **decouples** its two subdomains. Any subdomain
whose entire boundary is no-flow (outer no-flow + impermeable faults) is:

1. **singular up to an additive constant** — steady Neumann-only problems
   determine $h$ only to a gauge; and
2. **infeasible unless $\int_\Omega f = 0$** — the compatibility condition. With
   recharge and no outlet, no solution exists.

The optimiser will not report either cleanly; it will drift or stall. **Add a
pre-flight check**: for every subdomain, assert at least one of {Dirichlet edge,
river boundary, head observation}, and assert the net source balances. This is
cheap and will save days.

### B5 — XPINN on an elliptic problem needs a coarse space

As §2.4: one-level Schwarz, $\kappa = O(1/H^2)$, no global information channel.
Concretely, add a **coarse global network** $h_c(\mathbf x)$ shared by all
subdomains and write

$$h_k(\mathbf x) = h_c(\mathbf x) + \tilde h_k(\mathbf x)$$

with $\tilde h_k$ the local correction. $h_c$ carries the regional gradient in
one hop; $\tilde h_k$ only has to represent local structure, which is what
decomposition is good at. Alternatively adopt FBPINN's overlapping subdomains
with a partition of unity $\sum_k \omega_k = 1$, which removes interface losses
entirely (continuity is structural, not penalised) — strictly better-conditioned
than cPINN's penalty formulation and worth serious consideration.

### B6 — Interface conditions: cPINN's defaults are wrong for a semi-permeable fault

Standard cPINN enforces $h_k = h_\ell$ and residual continuity. A fault is
**not** a continuity interface — the head is *supposed* to jump. Required:

$$\mathcal I^{\text{mass}}_{k\ell} = \mathbf q_k\!\cdot\!\mathbf n - \mathbf q_\ell\!\cdot\!\mathbf n = 0
\qquad\text{(always)}$$

$$\mathcal I^{\text{jump}}_{k\ell} = \mathbf q_k\!\cdot\!\mathbf n + C_f\,(h_k - h_\ell) = 0,
\qquad C_f = K_f/b_f$$

with $C_f\to\infty$ (i.e. impose $h_k=h_\ell$ directly) on *artificial*
interfaces that are not faults. For conditioning, prefer the **Robin–Robin
(optimised Schwarz)** form over the raw pair:

$$\mathbf q_k\!\cdot\!\mathbf n + \gamma h_k = \mathbf q_\ell\!\cdot\!\mathbf n + \gamma h_\ell,
\qquad \gamma \sim K/H$$

which is the transmission condition with the best convergence factor and
degenerates gracefully at both $C_f\to0$ and $C_f\to\infty$.

Also note the **river is a boundary, not a source, in 3D.** The current code
applies $C_{riv}\mathbb 1_{riv}(h_{riv}-h)$ as a volumetric source over the
wetted polygon — correct for a depth-integrated layer, wrong in XYZ. In 3D it
must be a Robin condition on the streambed *surface*:

$$\mathbf q\cdot\mathbf n = C_{riv}\,(h - h_{riv}) \quad \text{on }\Gamma_{riv}$$

Carrying the 2D form into 3D double-counts by the aquifer thickness (~30×).

### B7 — Assembled loss

$$
\mathcal L=\underbrace{\sum_k\Big[\lambda_c\|\mathcal R^c_k\|^2+\lambda_m\|\mathcal R^m_k\|^2\Big]}_{\text{PDE, non-dimensionalised (§2.3)}}
+\underbrace{\sum_{k\ell}\Big[\mu_1\|\mathcal I^{\text{mass}}_{k\ell}\|^2+\mu_2\|\mathcal I^{\text{jump}}_{k\ell}\|^2\Big]}_{\text{constraints — fixed or augmented-Lagrangian }\mu}
+\lambda_d\mathcal L_{\text{data}}^{\epsilon}
+\lambda_M\|\mathcal M_V\|^2
+\lambda_\Gamma\|\mathcal B_{ws}\|^2
$$

with $\mathcal L^{\epsilon}_{\text{data}}$ the noise-floored misfit
$\langle(\max(|h-h^{obs}|-\epsilon,0))^2\rangle$, $\epsilon$ = stated
measurement accuracy.

### B8 — Identifiability does not improve in 3D

$S$ remains absent from the steady equation — transient data or no $S$.
And the measured 13×/5× errors on $C_{riv}$/$C_{01}$ are a **structural**
non-uniqueness: $C_{riv}$, $C_{01}$ and $K$ trade off against each other at
fixed heads. In 3D the aquitards become explicit low-$K$ volumes, which
*replaces* $C_{01}$ with $K_v$ of the aquitard — an improvement in principle,
but only if the aquitard is resolved by collocation points (it is thin: check
$\Delta z$ against aquitard thickness before trusting it).

---

## Recommended sequence

1. **TC-1 first, today.** It is a unit test with an analytic answer and will
   surface B4 and B6 within minutes.
2. **TC-2 against the committed baseline** (held-out RMSE 0.81 m, fault step
   12%). If XPINN does not beat 60% step recovery *without* losing head-field
   R², the decomposition is not paying for itself.
3. **TC-3 only after B2 is implemented**, confined, sweeping $K_h/K_v$.
4. **TC-4 with the coarse network from the start** (B5) — retrofitting it after
   convergence stalls at $N_{sub}=16$ wastes the intervening runs.
