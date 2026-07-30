# Choosing the backbone

The task asked which of MLP / ResNet / CNN suits this problem. All four
candidates are implemented behind one interface
(`gwpinn/models/backbones.py`) and driven by the same residual, so a run
changes only the field representation. This is what the measurement said, and
what it implies.

## The measurement

Identical data, seed, collocation schedule, loss terms and iteration budget
(900 Adam iterations, no L-BFGS); width and depth held equal across
architectures, so parameter counts differ and are reported. "Head field" and
"log10 T" are cell-by-cell against the reference finite-difference solution
(n = 28,010); "held-out well" is the 12 wells excluded from calibration.

| architecture | params | wall time (s) | calibration RMSE (m) | held-out RMSE (m) | held-out R² | head field RMSE (m) | head field R² | log10 T RMSE |
|---|---|---|---|---|---|---|---|---|
| mlp | 52,878 | 596 | 0.058 | 1.307 | 0.863 | 1.331 | 0.803 | 0.616 |
| resnet | 62,190 | 667 | 0.056 | 1.128 | 0.898 | 1.368 | 0.792 | 0.657 |
| modified_mlp | 85,198 | 1084 | 0.071 | 1.500 | 0.820 | 1.989 | 0.560 | 0.900 |
| **cnn** | 405,518 | 1490 | **0.164** | **1.014** | **0.918** | **1.028** | **0.883** | **0.521** |

Reproduce with:

```bash
python scripts/benchmark_architectures.py sample_data/config.yaml --iters 900
```

## What the numbers say

**The CNN fits the calibration wells worst and everything else best.** Its
calibration RMSE is 0.164 m against 0.056–0.071 m for the coordinate networks —
roughly three times worse — yet it wins on held-out wells, on the head field,
and on transmissivity. That inversion is the whole result, and it is a
regularisation story, not a capacity story: the CNN has five to eight times more
parameters than any of the MLPs.

A coordinate network with Fourier features can put a narrow bump exactly at each
of the 47 calibration wells. With 47 wells over 80 km² that is the cheapest way
to reduce the data loss, and it costs accuracy everywhere in between. The grid
plus cubic-B-spline representation cannot do it: the field is stored at a fixed
pixel pitch and read through a smooth interpolant, so its attainable frequency
content is capped. It is forced to explain the wells with a field that is
coherent at the scale of the aquifer, which is the field we actually want.

That the effect shows up most strongly in the *property* field
(log10 T RMSE 0.521 vs 0.616–0.900) is consistent: transmissivity is the worst-
determined quantity in the inversion, so it is where an implicit prior earns the
most.

**Held-out RMSE alone would not have supported this conclusion** — it rests on
12 wells, and the spread across architectures (1.01–1.50 m) is not resolvable
with that sample. The cell-by-cell metrics (n = 28,010) are what make the
ranking trustworthy, and they agree with it.

**`modified_mlp` is last here**, which is not what its usual motivation would
predict. The gating of Wang, Teng & Perdikaris (2021) is designed to relieve
stiff gradient interactions between the data and residual losses, but that
pathology is already handled in this pipeline by the adaptive weighting, the
warm-up ramp and the non-dimensionalisation. What remains is a heavier network
(1084 s against 596 s) that converges more slowly — at 900 iterations it has not
caught up. Given a longer budget it does: the production run in
`docs/example_run/` uses `modified_mlp` for 4000 Adam iterations plus L-BFGS and
reaches 0.81 m on held-out wells, better than any row above. So the table ranks
convergence *at this budget*, not asymptotic capability.

## Why the CNN is still not the default

The default remains a coordinate network, because the CNN's advantage here is
contingent and its limitations are structural:

1. **Resolution is fixed by the grid.** The decoder emits a 128 × 128 field over
   a 10 × 8 km domain — about 78 m per pixel. The fault barriers in this case are
   60 m wide, i.e. *below* the pixel pitch, so the CNN cannot represent a barrier
   any sharper than it already fails to. A case with narrower barriers, or one
   needing local refinement, would need a larger grid everywhere.
2. **Derivatives are the interpolant's, not the field's.** Bilinear sampling has
   an identically zero second derivative and cannot feed a second-order PDE at
   all; this implementation uses a cubic B-spline, which is C² and does work, but
   the residual is being enforced on a spline reconstruction rather than on the
   network output directly.
3. **The domain has to be masked.** Coordinate networks never represent the
   outline — points are simply sampled inside it. A grid spends capacity and
   gradient on cells outside the aquifer.
4. **Cost.** 2.5× the wall time and 6× the parameters of the residual MLP.

Among the coordinate networks, `resnet` is the best accuracy-per-second: it
matches `mlp` on the head field, beats it on held-out wells, and costs 12% more
time.

## Recommendation

- **Sparse data, modest compute, smooth aquifer** → `cnn`. Its implicit
  smoothness is worth more than the flexibility it gives up, and it is the most
  robust to overfitting a small well network.
- **Sharp internal structure (narrow faults), irregular domains, or a long
  training budget** → a coordinate network; `resnet` for the best
  accuracy-per-second, `modified_mlp` if the budget is generous.
- Either way, the loss formulation mattered far more than the architecture in
  this study. Adding a noise floor to the data terms, redrawing the variogram
  pairs every iteration and sampling inside the fault zones moved held-out RMSE
  from 2.45 m to 0.81 m — a larger effect than any difference in the table above.

## Caveats

- One seed per architecture, one budget, one synthetic case.
- The table was produced at commit `446ac2e`, before two sampling fixes
  (per-iteration variogram pairs, fault-zone collocation) that apply equally to
  all four architectures. See `runs/benchmark2` for a confirmation run of the two
  front-runners on the final code.
- Ranking at fixed *wall clock* rather than fixed iterations would favour the
  cheaper backbones and penalise the CNN further.
