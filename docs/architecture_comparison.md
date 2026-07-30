# Choosing the backbone

The task asked which of MLP / ResNet / CNN suits this problem. This is the
reasoning and the measurement behind the default.

## Why a coordinate network, not a grid network

The requirement that settles it is **exact derivatives at arbitrary points**.

The flow residual is second order, and it has to be evaluated at collocation
points scattered inside an irregular polygon, densified inside a narrow river
channel, and concentrated near fault traces. A coordinate network
`f(x, y) → h` gives those derivatives directly through automatic
differentiation, at any point, at no extra cost — and it never has to represent
the domain outline, because points are simply sampled inside it.

A convolutional network predicts a field on a **regular grid**. That creates
three problems here, none of which is fatal on its own:

1. **Derivatives.** Reading the grid back at an arbitrary point needs an
   interpolant. Bilinear sampling has an identically zero second derivative, so
   it cannot feed a second-order PDE at all. This implementation therefore reads
   the CNN's grid through a **cubic B-spline**, which is C² and does support the
   residual — but the derivatives are now those of the interpolant, not of the
   field.
2. **Resolution is fixed by the grid**, not by where information is. A fault
   barrier a few tens of metres wide has to be resolved by a grid that is uniform
   over the whole domain.
3. **The irregular domain and the layer stack** have to be handled by masking,
   and masked cells still consume capacity and gradient.

The one thing a CNN buys — a spatial inductive bias toward locally coherent
fields — is supplied here by the variogram term, which states the required
spatial correlation explicitly rather than implying it through an architecture.

## Why `modified_mlp` over a plain MLP or a residual MLP

All three are coordinate networks and differ only in how the hidden layers are
wired.

- **`mlp`** — the standard PINN backbone. Fine, but the data loss and the PDE
  residual produce gradients of very different magnitude on the shared weights,
  and the stiffer one dominates.
- **`resnet`** — skip connections keep gradients from vanishing with depth. Helps
  at depth; does not address the data-vs-residual imbalance.
- **`modified_mlp`** — Wang, Teng & Perdikaris (2021). Two encoder projections
  `U`, `V` of the input multiplicatively modulate *every* hidden layer:

  ```
  H^{k+1} = (1 − Z^k) ⊙ U + Z^k ⊙ V ,    Z^k = σ(W^k H^k + b^k)
  ```

  The multiplicative paths give residual gradients a short route back to the
  input, which is exactly the pathology that makes stiff PINNs stall.

All four are implemented behind one interface (`gwpinn/models/backbones.py`) and
driven by the same residual, so the comparison below changes only the field
representation.

## Measured comparison

Identical data, seed, collocation schedule, loss terms and iteration budget;
only `model.arch` differs. Width and depth are held equal across architectures,
which is the usual protocol — parameter counts therefore differ and are reported.

Reproduce with:

```bash
python scripts/benchmark_architectures.py sample_data/config.yaml --iters 1000
```

<!-- BENCHMARK_TABLE -->

## Caveats

- One seed per architecture at a fixed budget. Ranking at a fixed *wall-clock*
  budget rather than a fixed iteration count would favour the cheaper backbones.
- Results are from the synthetic case in `scripts/make_sample_data.py`. A
  different heterogeneity structure or well density could reorder the middle of
  the table; the coordinate-vs-grid distinction is structural and would not move.
