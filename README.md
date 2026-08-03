# GWPINNS — inverse Physics-Informed Neural Networks for 3D transient groundwater flow across a fault

Infer a **3D hydraulic conductivity field** and, more importantly, the
**hydraulic behaviour of a fault** (barrier or conduit) from nothing but sparse
groundwater head time series `h(x,y,z,t)` at a handful of monitoring wells.

The governing equation, implemented directly in PyTorch autograd with no
high-level PDE wrapper:

$$S_s \frac{\partial h}{\partial t} = \nabla \cdot \left( K \nabla h \right) + W$$

Three architectures are implemented and compared on the same benchmark, because
the thing that makes this problem hard — a conductivity discontinuity spanning
five orders of magnitude — is exactly what a plain PINN handles worst.

---

## Repository structure

```
GWPINNS/
├── gwpinns/
│   ├── config.py               Single source of truth: geometry, fault, stresses, wells
│   ├── benchmark/              PHASE 1 — synthetic truth
│   │   ├── fields.py             Log-normal K via FFT circulant embedding + fault overprint
│   │   ├── modflow6.py           FloPy / MODFLOW 6 forward model
│   │   ├── fdsolver.py           Reference FD solver (same discretisation as MF6)
│   │   ├── observations.py       Sparse monitoring-well sampler with noise
│   │   └── generate.py           Orchestration + on-disk dataset format
│   ├── pinn/                   PHASE 2 — the inverse models
│   │   ├── scaling.py            Non-dimensionalisation and the PDE coefficients
│   │   ├── networks.py           MLP, Fourier features, bounded log-K, fault leakance
│   │   ├── forcing.py            Differentiable W(x,y,z,t,h): wells + recharge + ET
│   │   ├── derivatives.py        Thin autograd helpers
│   │   ├── sampling.py           Collocation / boundary / interface point sampling
│   │   ├── base.py               Shared coordinate handling and loss assembly
│   │   ├── baseline.py           Architecture 1 — standard inverse PINN
│   │   ├── mixed.py              Architecture 2 — mixed-variable (head + Darcy flux)
│   │   ├── cpinn.py              Architecture 3 — conservative domain decomposition
│   │   └── trainer.py            Adam + L-BFGS, loss balancing, physics curriculum
│   └── evaluation/
│       ├── metrics.py            Head / conductivity / fault-characterisation scores
│       ├── report.py             Model → metric record
│       └── plots.py              Figures (colourblind-safe, validated palette)
├── scripts/
│   ├── 01_generate_benchmark.py  Build both scenarios
│   ├── 02_train.py               Train one architecture on one scenario
│   └── 03_run_study.py           Full comparison + figures + Markdown report
├── tests/
│   ├── test_benchmark.py         Incl. Theis and 1-D barrier analytical validation
│   └── test_pinn.py              Incl. manufactured-solution check of the residual
├── data/                         Generated benchmarks (gitignored)
└── runs/                         Training outputs (gitignored)
```

---

## Quick start

```bash
pip install -r requirements.txt

# optional but recommended — otherwise the reference solver is used
python -m flopy.utils.get_modflow ~/.local/bin

python scripts/01_generate_benchmark.py            # Phase 1: both scenarios
python scripts/03_run_study.py --quick             # smoke test, a few minutes
python scripts/03_run_study.py                     # full study
pytest -q                                          # 35 tests
```

---

## Phase 1 — the benchmark

A three-layer **confined** aquifer, 5000 × 3000 × 60 m on a 50 × 30 × 3 grid,
bisected by a fault zone whose plane strikes 12° off the y-axis.

The aquifer is kept confined (`icelltype=0`) deliberately. It makes the
governing equation exactly the one quoted above; an unconfined formulation
would make transmissivity head-dependent and the forward and inverse models
would no longer be solving the same problem.

| Component | Setting |
|---|---|
| Conductivity | Layer-wise log-normal, geometric means 3 / 0.4 / 6 m/d, σ<sub>lnK</sub> = 0.8, correlation lengths (900, 700, 25) m |
| Fault zone | 150 m wide, fully penetrating, K overwritten sharply |
| **Scenario A — barrier** | K<sub>fault</sub> = 10⁻³ m/d (≈3.3 orders *below* host) |
| **Scenario B — conduit** | K<sub>fault</sub> = 200 m/d (≈2 orders *above* host) |
| Specific storage | 10⁻⁴ m⁻¹ |
| Boundaries | Constant head 55 m (west) / 45 m (east); no-flow elsewhere |
| Recharge | Spatially variable, ~10⁻⁴ m/d (MODFLOW `RCHA`) |
| Evapotranspiration | Head-dependent linear ramp, surface 58 m, extinction depth 8 m (`EVTA`) |
| Pumping | 3 wells, layers 0/1/2, staggered on/off schedules over 6 stress periods |
| Time | 1 steady-state spin-up + 6 × 60 d transient, 10 d steps (37 output times) |
| Monitoring | 15 wells × 2 screens = 30 points × 37 times = **1110 head values** |
| Noise | Gaussian, σ = 15 mm |

Monitoring wells within 195 m of the fault plane are dropped — a real
piezometer is not screened in a damage zone, and it keeps the cPINN's domain
decomposition unambiguous. Data coverage is **0.67%** of the space–time grid.

The two scenarios are identical in every respect except the fault conductivity,
and they are cleanly separable in the heads: extrapolating the head field to
each fault wall and differencing gives a **+8.6 m** jump for the barrier against
**+0.001 m** for the conduit at steady state.

### Two forward engines

`gwpinns.benchmark.modflow6` builds and runs a genuine MODFLOW 6 model through
FloPy (DIS / NPF / STO / IC / CHD / RCHA / EVTA / WEL / OC).

`gwpinns.benchmark.fdsolver` is a reference implementation of the *same*
discretisation in NumPy/SciPy — cell-centred finite volume, harmonic-mean
inter-cell conductances, fully implicit time stepping, constant-head rows, and
MODFLOW's segmented linear ET ramp linearised implicitly with Picard outer
iterations. `generate_benchmark(..., engine="auto")` prefers MODFLOW and falls
back to it with a warning.

The fallback is not taken on trust. `tests/test_benchmark.py` validates it
against the **Theis solution** (agreement within 5% over radii of 4–12 cells)
and against the **1-D series-resistance law** for steady flow through a low-K
slab (within 5%).

> **Provenance note.** The results in this repository were produced with the
> reference finite-difference solver: the sandbox used for development had no
> network route to the MODFLOW binary distribution. The FloPy script is
> complete and is the intended default; re-running with `--engine mf6` on a
> machine with `mf6` installed exercises it.

### Outputs per scenario

```
data/benchmark/<scenario>/
    config.json        full configuration, round-trippable
    truth.npz          head grid, true K, stress fields, coordinates
    observations.csv   the sparse training data
    observations.npz   the same, for fast loading
    sources.csv        well coordinates, cell volumes, pumping schedules
    manifest.json      provenance and summary diagnostics
```

---

## Phase 2 — the architectures

### Non-dimensionalisation

Inputs are mapped to `[-1,1]` per axis and the head is centred and scaled on the
*observed* values only. Dividing the flow equation by `Ss·Δh/a_t` gives the
residual every architecture shares:

```
d_T H  −  α · Σ_i μ_i² d_i( K d_i H )  −  β · W  =  0

μ_i = L0/a_i     α = a_t/(Ss·L0²)     β = a_t/(Ss·Δh)
```

`K` enters in **physical units** and the reference conductivity cancels
completely — the physics loss needs no prior guess of K. The first-order (mixed)
form of the identical equation is

```
Darcy       :  U_i + (K/K0)·μ_i·d_i H                    = 0
continuity  :  d_T H + α·K0·Σ_i μ_i·d_i U_i  −  β·W      = 0
```

`tests/test_pinn.py` checks both forms against a manufactured analytic solution
and against each other.

### The forcing term `W`

`W` is *known* to the inverse problem — pumping is metered and recharge/ET are
prescribed constitutive laws — so it is evaluated exactly as the forward model
applied it: wells as `Q/V_cell` inside the screened cell during the relevant
stress period, recharge as `R(x,y)/Δz₀` in the top layer, and ET as the MODFLOW
ramp evaluated on the model's **own predicted head**, keeping it in the autograd
graph. The PINN therefore solves a genuinely head-dependent sink rather than
being handed the answer.

### 1. Baseline inverse PINN (`baseline.py`)

`H = N_h(X,Y,Z,T)` and `K = N_k(X,Y,Z)`, trained on data misfit plus the
second-order residual. The divergence is built by nested autograd, so the
`∇K·∇h` cross term is exact without ever being written out. This is also its
weakness: across the fault `∇K` is near-singular.

### 2. Mixed-variable PINN (`mixed.py`)

The network outputs the Darcy flux alongside the head,
`(H, U_x, U_y, U_z) = N(X,Y,Z,T)`, and the equation is split into Darcy's law
and mass conservation. Only first derivatives appear. Crucially, the quantity
that is physically *continuous* across the fault — the normal flux — is a direct
network output rather than a product of two discontinuous factors. No-flow
boundaries also become algebraic (`U_n = 0`) instead of differential.

### 3. cPINN — conservative domain decomposition (`cpinn.py`)

Two independent head networks and two independent K networks, one per fault
block. Neither ever has to represent the discontinuity; it lives entirely in the
coupling.

Flux continuity across the plane is unconditional:

```
Q_n^west = Q_n^east
```

For the head, note the physics: **strict head continuity cannot represent a
barrier.** A low-permeability fault exists precisely to sustain a head jump, and
forcing `H_west = H_east` would make the model explain that jump with a badly
wrong conductivity field on either side. The correct thin-feature condition is a
*leaky wall*:

```
Q_n = Γ · (H_west − H_east),        Γ = C·L0/K0,    C = K_fault / width
```

`Γ → 0` is a perfect barrier; `Γ → ∞` recovers head continuity. **The fitted Γ
is itself the answer to "barrier or conduit?"** — a single interpretable number,
reported back as `K_fault = C · width` in m/d. It is inferred, never prescribed.
The condition is imposed in the normalised form `[Γ(H_w − H_e) − Q_n]/(1 + Γ)`,
which stays well conditioned at both limits.

This is a deliberate extension of the usual cPINN interface conditions. The
textbook `H_west = H_east` form is retained as `--interface-mode continuity` so
the study can show it failing on the barrier scenario.

### Making the inverse problem converge

Three ingredients, each earning its place:

- **Gradient-norm loss balancing** (Wang, Teng & Perdikaris 2021). The PDE
  residual starts ~10⁵× the data misfit, largely because the 83:1 horizontal-to-
  vertical aspect ratio puts a factor `μ_z² ≈ 6900` on the vertical diffusion
  term. Unbalanced, the optimiser spends its whole budget flattening vertical
  gradients.
- **Physics curriculum.** A data-only warm-up followed by a linear ramp of the
  physics weights. Switching the PDE on at iteration zero, against an untrained
  head field, drives K straight into the bottom of its bounded range.
- **Bounded log-K with a weak Tikhonov prior.** Away from wells and observation
  points the problem is genuinely degenerate — shrinking K towards zero satisfies
  the PDE for *any* smooth head field — and the optimiser reliably finds that
  route. The prior is a round-number bulk estimate (1 m/d, deliberately not the
  benchmark's true 1.93 m/d geometric mean) at low weight.

The `residual_weighting="source"` option divides the residual by `1 + |βW|`.
Inside a pumping cell `βW` is ~1000× its bulk value, so an unweighted
mean-square residual is effectively a well-cell-only loss.

---

## Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full study, figures and the
honest account of what does and does not work.

---

## Citation of methods

- Theis (1935) — analytical validation of the forward solver.
- Langevin et al. — MODFLOW 6 discretisation (harmonic conductances, EVT ramp).
- Raissi, Perdikaris & Karniadakis (2019) — PINNs.
- Jagtap, Kharazmi & Karniadakis (2020) — conservative PINNs.
- Wang, Teng & Perdikaris (2021) — gradient-norm loss balancing.
- Tancik et al. (2020) — Fourier feature encoding.
