# gwpinn — physics-informed inversion of groundwater head and aquifer properties

`gwpinn` fits a **groundwater table** and **heterogeneous aquifer property fields**
(transmissivity, hydraulic conductivity, storage coefficient) to sparse field data,
subject to the multilayer groundwater flow equation.

It takes the data a real project actually has — a few dozen well readings, a handful
of river gauges, some pumping tests, and elevation rasters — and returns rasters of
head and aquifer properties per layer, together with an accuracy assessment.

```
observed heads (points)  ─┐
river gauges + water-surface polygon ─┤
faults / flow barriers (lines) ───────┼──►  PINN  ──►  head raster per layer
DTM + layer bottom rasters ───────────┤              aquifer property rasters
pumping-test T and S (points) ────────┘              accuracy metrics + plots
```

---

## What makes this an inverse problem, and how it is made well-posed

Only the *divergence* of `T ∇h` is visible in head data, so many transmissivity
fields reproduce the same observed heads. Four independent constraints are combined
so the answer is determined by more than the head misfit alone:

| Constraint | Where it comes from |
|---|---|
| **Flow equation** | quasi-3D multilayer PDE enforced at collocation points |
| **Point measurements** | pumping-test `T` and `S`, anchoring the level of each field |
| **Variogram structure** | the *spatial statistics* of the property field must match the variogram fitted to the pumping tests — the sill **and** the correlation range |
| **Edge-preserving regularisation** | pseudo-Huber (total-variation-like) penalty that suppresses speckle without smearing genuine facies contacts |

The variogram term is the distinctive one. A smoothness penalty can only push
variance *down*, so it biases the field toward flat. Matching the variogram forces
the learned field to have the right variance *and* the right correlation length —
it constrains texture, not just roughness.

---

## Governing equations

For each aquifer layer `l`:

```
S_l ∂h_l/∂t  =  ∇·( T_l A ∇h_l )
               + C_{l-1,l} (h_{l-1} − h_l)          vertical leakage from above
               + C_{l,l+1} (h_{l+1} − h_l)          vertical leakage from below
               + R_l                                 areal recharge (layer 0)
               + C_riv · 1_river · (h_riv − h_l)     river exchange (Robin / MODFLOW RIV)
```

* **Unconfined top layer** — `T₀ = K₀ · (h₀ − z_bot,₀)`, so the equation is nonlinear
  in head (Boussinesq). A mesh-free PINN handles this with no outer Picard loop.
* **Confined layers** — `T_l = K_l · (z_bot,l−1 − z_bot,l)`.
* **`A`** is the fault anisotropy tensor (below).

Every derivative is taken by automatic differentiation, so the residual is exact at
every point rather than discretised onto a grid.

### Faults as anisotropic barriers

A fault is a surface across which head drops sharply. Rather than decomposing the
domain (XPINN-style), the fault is represented as a narrow zone in which
conductivity *normal to the fault* is scaled by `α`, while flow along it is
untouched:

```
q = −T ( I − (1−α) ψ(x) n nᵀ ) ∇h        ψ = Gaussian bump of half-width w
```

This is the continuous analogue of MODFLOW's Horizontal Flow Barrier package. It
keeps one global network, and `α` is a plain trainable scalar.

The `perm` attribute on each fault line controls this:

| `perm` | meaning |
|---|---|
| `0` | impermeable barrier — `α` pinned at `physics.fault_perm_min` |
| `1` | `α` is **trainable** — fitted from the head data |
| `0 < p < 1` | fixed permeability multiplier `p` |

Two honest caveats:

* A *perfectly* impermeable fault would need an infinite head gradient, which no
  continuous network can represent. `α` is floored at `fault_perm_min` (default
  1e-3), which is hydraulically indistinguishable from a true barrier.
* Each collocation point also receives a **signed side indicator** per fault
  (`tanh` of the signed distance). This gives the network a ready-made basis for a
  jump across the fault, so it can produce a sharp offset without needing extreme
  Fourier frequencies.

### River stage

Gauges give stage at points; the polygon gives the area over which stage is needed.
Interpolation is done in **along-stream coordinates**, not in the plane:

1. A centerline is extracted from the water-surface polygon by principal-curve
   fitting (or supplied directly).
2. Gauges are projected onto it to give a chainage.
3. Stage is interpolated against chainage with a monotone (PCHIP) fit, optionally
   passed through a decreasing isotonic regression first.

A 2-D interpolator would leak the downstream gradient across meander bends and
produce a water surface that runs uphill along the channel.

### Layer geometry

Independently interpolated bottom surfaces routinely cross. Every bottom is pushed
down so each layer is at least `domain.min_thickness` thick, cascading downward from
the DTM; the number and magnitude of corrections is reported.

### On storage coefficients

**`S` does not appear in the steady-state equation.** In `regime: steady` it is
constrained *only* by the pumping-test points and their variogram — it is a
geostatistical interpolation, not a physical inversion, and the report should be
read that way. Set `regime: transient` (with time-stamped observations) to make `S`
identifiable from the flow field itself.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# 1. generate a synthetic case with known ground truth
python scripts/make_sample_data.py -o sample_data

# 2. fit
python scripts/train.py sample_data/config.yaml -o runs/demo

# 3. compare architectures
python scripts/benchmark_architectures.py sample_data/config.yaml --iters 1000
```

Outputs land in `runs/demo/`:

```
rasters/    head_L*.tif, transmissivity_L*.tif, conductivity_L*.tif,
            storage_L*.tif, saturated_thickness_L*.tif, depth_to_water.tif
            (+ *_std_L*.tif when an ensemble was trained)
plots/      01_inputs … 11_uncertainty
report.json / report.txt      all accuracy metrics
history.json, model.pt, config_used.yaml
```

---

## Inputs

All optional except the DTM, one layer bottom, and head observations. Attribute
names are matched case-insensitively through an alias table, so `HEAD`, `gw_head`,
`water_leve` (DBF truncation) all resolve.

| Config key | Geometry | Attributes |
|---|---|---|
| `head_obs` | point | `head`, opt. `layer`, `time`, `weight` |
| `gauge_obs` | point | `stage`, opt. `time` |
| `river_polygon` | polygon | — |
| `river_centerline` | line | — (derived from the polygon if absent) |
| `faults` | line | `perm` (see above) |
| `dtm` | raster | top of layer 0 |
| `layer_bottoms` | rasters | one per layer, ordered top → bottom |
| `prop_obs` | point | `T` (m²/d), `S` (−), opt. `layer` |
| `domain` | polygon | optional active-area clip |
| `boundary` | line | `bctype` ∈ {`noflow`,`head`,`ghb`}, `value`, opt. `cond` |
| `recharge` | raster | optional; else `physics.recharge` |

---

## Architecture

Four field representations share one interface and one residual, so the comparison
is like-for-like. All are twice-differentiable by autograd.

| `model.arch` | what it is |
|---|---|
| `mlp` | Fourier-feature MLP — the standard PINN backbone |
| `resnet` | the same with residual blocks |
| `modified_mlp` | gated architecture of Wang, Teng & Perdikaris (2021) |
| `cnn` | convolutional decoder to a grid, read back through a **cubic B-spline** — **default** |

The CNN needs the spline because bilinear sampling has an identically zero second
derivative and cannot feed a second-order PDE at all.

The measured comparison held one surprise. At a fixed 900-iteration budget the
**CNN fits the calibration wells worst (0.164 m against 0.056 m) and generalises
best** — lowest held-out RMSE, lowest head-field and transmissivity error. Its
grid-plus-spline representation is band-limited, so unlike a Fourier-feature MLP
it *cannot* put a narrow bump at each of the 47 calibration wells, and is forced
to explain them with a field coherent at the scale of the aquifer. That is
regularisation, not capacity: the CNN has 6× more parameters.

The ordering held across two independent runs on two code revisions, so `cnn`
is the default. Switch to `resnet` when barriers are narrower than the CNN's
pixel pitch (78 m in this case), when the domain is fragmented enough that
masking wastes capacity, or when wall clock binds — it is half the time for
about 85% of the accuracy. Full tables, reasoning and a situation-by-situation
recommendation in
[`docs/architecture_comparison.md`](docs/architecture_comparison.md).

### Why these components

* **Random Fourier features, multi-scale.** Coordinate MLPs are biased toward low
  frequencies and will not resolve a heterogeneous property field. Three frequency
  bands per network avoid guessing one bandwidth (Tancik et al. 2020; Wang et al. 2021).
* **Separate head and property networks.** The head field is smoothed by the PDE and
  is low-frequency; the property field is rough. The property network gets a
  higher-frequency Fourier band.
* **Non-dimensionalisation.** Coordinates to `[-1,1]`, heads standardised, and the
  residual divided by a characteristic magnitude. Without this the residual and the
  data losses differ by many orders of magnitude and no fixed weights work.
* **Gradient-norm adaptive loss balancing** (Wang et al. 2021), with a ceiling: as a
  loss term approaches zero its gradient does too, and the raw rule sends its weight
  to infinity.
* **Noise-floored data loss.** Residuals smaller than `train.head_noise` cost
  nothing, so the model fits the wells to within their stated accuracy and no
  further — the chi-square-of-one target of classical calibration. Fitting below the
  measurement error is fitting noise, and it *raises* the error between wells.
* **Adam then L-BFGS**, with collocation points frozen during L-BFGS (its line
  search assumes a deterministic objective).

---

## Validation

`scripts/make_sample_data.py` builds a two-layer aquifer with a meandering river,
one impermeable and one leaky fault, and Gaussian-random-field property fields with
a **known** variogram, then solves it with a conventional cell-centred
finite-difference model (`gwpinn/data/fdsolver.py`).

Only sparse samples are handed to the PINN — well readings with noise, gauge stages,
pumping tests with log-scale scatter, and the **uncorrected** layer rasters. The full
solution is kept aside for scoring.

This matters: the FD solver and the PINN share no code beyond the physics they
represent — one is a mesh-based linear solve, the other a mesh-free optimisation —
so agreement between them is evidence about the method, not a shared formulation
error. Both are independently checked against closed-form solutions in `tests/`.

Metrics produced: RMSE / MAE / bias / R² / NSE / KGE / PBIAS / RSR / Willmott's d,
on calibration wells, held-out wells, per layer, and cell-by-cell against the
reference; Moran's I and a residual semivariogram (is spatial structure left in the
error?); error growth with distance from the nearest calibration well; recovery of
fault permeability, riverbed conductance and leakance; and, with
`train.n_ensemble > 1`, a spatial uncertainty map from the ensemble spread.

```bash
python -m pytest tests/ -q
```

### Results on the synthetic case

59 wells (47 calibration / 12 held out), 6 gauges, 28 pumping tests, two layers,
10 × 8 km. `modified_mlp`, 4000 Adam + 250 L-BFGS iterations. Full report and all
figures in [`docs/example_run/`](docs/example_run/).

| quantity | n | RMSE | R² |
|---|---|---|---|
| head, calibration wells | 47 | 0.041 m | 0.9998 |
| **head, held-out wells** | 12 | **0.810 m** | **0.947** |
| head field vs reference, layer 0 | 28,010 | 1.08 m | 0.870 |
| head field vs reference, layer 1 | 28,010 | 1.07 m | 0.868 |
| log10 T at pumping tests | 28 | 0.066 | 0.989 |
| log10 T field vs reference, layer 0 | 28,010 | 0.611 | −0.97 |

Head residuals show no significant spatial autocorrelation (Moran's I = 0.034,
p = 0.36), and validation error grows sensibly with distance from the nearest
calibration well (0.17 m within 200 m → 1.23 m at ~1 km).

### What it does *not* do well

Reported plainly, because the figures show it either way:

- **The transmissivity field has negative R² against the truth.** It reproduces
  the observed *texture* (correlation length and variance — see
  `08_variograms.png`) and recovers large-scale anomalies in layer 1, but it is
  biased about 0.3 log10 units low and has little pointwise skill. Recovering a
  log-normal K field from 15 pumping tests and 47 heads is genuinely
  under-determined; the head field is recovered far better than the properties
  that produce it.
- **The head jump across the impermeable fault is not reproduced** where no wells
  sit near the trace (`10_faults.png`): the reference has a 6 m step, the model
  a 1 m ramp. Adding collocation points inside the barrier zone raised coverage
  from 3.6% to 16.7% and did not fix it — the jump is a local feature that the
  flow equation alone does not pin down without nearby data. The practical
  reading is that a barrier's throw needs an observation pair straddling it.
- **The lumped physical parameters trade off against each other.** Riverbed
  conductance came back 13× low and vertical leakance 5× low, while the head
  field stayed accurate — different combinations reproduce the same heads. Treat
  the fitted conductances as effective values, not measurements.

---

## Configuration

Everything is driven by one YAML file, round-tripped from dataclasses in
`gwpinn/config.py` (which carries the per-field documentation). Unknown keys are
rejected rather than silently ignored.

```yaml
paths:
  dtm: dtm.tif
  layer_bottoms: [bottom_layer1.tif, bottom_layer2.tif]
  head_obs: head_obs.shp
  gauge_obs: gauges.shp
  river_polygon: river_polygon.shp
  faults: faults.shp
  prop_obs: aquifer_props.shp

physics:
  regime: steady          # or transient
  recharge: 1.2e-4        # m/d
  leakance: [1.0e-3]      # 1/d, per interface
  fault_width: 60.0       # m, barrier smear half-width

model:
  arch: modified_mlp

train:
  adam_iters: 6000
  lbfgs_iters: 300
  head_noise: 0.05        # m, assumed measurement accuracy
  n_ensemble: 1           # >1 gives uncertainty maps
```

---

## Layout

```
gwpinn/
  config.py            YAML-backed configuration
  dataset.py           reads every input, assembles training batches
  io/                  raster & shapefile IO, layer overlap correction
  geo/                 domain, river centerline & stage, fault fields
  physics/             autodiff operators, flow residual, boundary conditions
  models/              Fourier features, backbones, head & property fields
  stats/               variograms, kriging, accuracy metrics
  train/               losses, adaptive weighting, Adam + L-BFGS
  postproc/            prediction, raster export, reporting, plots
  data/                Gaussian random fields, FD reference solver, sample case
scripts/               make_sample_data.py, train.py, benchmark_architectures.py
tests/
```

## References

- Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks*, JCP 378.
- Tancik et al. (2020), *Fourier features let networks learn high frequency functions*, NeurIPS.
- Wang, Teng & Perdikaris (2021), *Understanding and mitigating gradient flow pathologies in PINNs*, SIAM J. Sci. Comput. 43(5).
- Wang, Wang & Perdikaris (2021), *On the eigenvector bias of Fourier feature networks*, CMAME 384.
- Harbaugh (2005), *MODFLOW-2005*, USGS TM 6-A16 — quasi-3D layering, RIV and HFB packages.
- Gupta et al. (2009), *Decomposition of the mean squared error and NSE performance criteria*, J. Hydrol. 377.
