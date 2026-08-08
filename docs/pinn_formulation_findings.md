# What actually won, and what that means

Narrative conclusions from the screening study. The raw tables are in
[`pinn_formulation_results.md`](pinn_formulation_results.md) (generated from
`runs/pinn_formulation_study/results.csv`); the design and the SOTA survey are in
[`pinn_formulation_design.md`](pinn_formulation_design.md).

> **Scope.** 28+ training runs at 240 s each on 4 CPU threads, single seed,
> against a 100×100-cell, 48-stress-period single-layer MODFLOW reference. This
> is an equal-compute screening study, not a convergence study. Read section 6
> before quoting any of it.

---

## 1. The result that reframes everything: the null model

The initial condition is a hard constraint (`h = h_init + s(t)·hs·NN`), which
was a deliberate design choice — it removes a loss term and guarantees every
variant starts identically. The consequence only became clear when the null
model was scored:

| | persistence (learns nothing) |
|---|---|
| head RMSE | **0.912 m** |
| head NSE | **0.9899** |
| fault jump RMSE | **0.558 m** |
| fault jump "recovery" | **0.928** |

A model that does nothing at all already reproduces 93% of the fault jump,
because the true initial field — discontinuity included — is handed to it.

**Of 19 forward screening runs, exactly four beat persistence on head. All four
are control-volume runs.** Every strong-form and every mixed-form variant,
including the baseline at 1.486 m, is *worse than doing nothing*.

**Nothing beat persistence on the fault criterion.** Best skill: +0.002.

Two lessons. First, always score the null model — without it this study would
have reported a tidy ranking of methods that mostly could not beat a constant.
Second, `head_nse` ≈ 0.99 is meaningless here: the spatial head range is 32 m
while the transient signal is ~1 m, so NSE is dominated by static structure. The
metrics to read are RMSE and skill-vs-persistence.

## 2. Formulation beats architecture, decisively

| axis | best variant | head RMSE | vs persistence |
|---|---|---|---|
| **form** | `fv` (control volume) | **0.772** | **+0.15** |
| arch | `mlp+rwf` | 0.940 | −0.03 |
| balance | `ntk` | 1.292 | −0.42 |
| fault | `none` | 1.200 | −0.32 |
| temporal | (none helped) | 2.022 | −1.22 |

The control-volume formulation wins on head, fault jump *and* mass balance
simultaneously — while completing the **fewest iterations** of any cheap variant
(1,070–1,794 against up to 4,172 for a plain MLP). It is not winning on compute;
it is winning on structure. Enforcing the balance over control volumes, where
face fluxes telescope, is worth more than three times as many collocation steps.

Architecture changes moved head RMSE by ±30%. Changing the formulation moved it
by a factor of two and was the only thing that crossed the persistence line.
That ordering is the single most useful thing this study says.

**PirateNet lost at equal wall clock** (2.14 m, worst of the architectures) —
not because it is a bad architecture but because it is ~2× the cost per step and
got 1,300 iterations against 4,172. At equal *iterations* it would likely look
very different. This is the equal-compute framing doing its job: it answers "what
should I run for the next ten minutes", not "what is best asymptotically".

## 3. The mixed form is a specialist, not a generalist

Once its flux output was correctly scaled (the screening run asked the network to
emit ~0.02 instead of ~1 — a fairness bug I fixed and re-ran), the mixed
formulation was:

- **best of all runs on river exchange**: 0.120 relative error, against 0.161 for
  the best strong-form and 0.212 for the best control-volume run;
- still poor on head (1.61 m, worse than persistence).

That is a coherent story rather than noise. The mixed form carries the Darcy flux
as an explicit network output, and river leakage *is* a flux. If the deliverable
is an aquifer–river exchange budget — the number a surface-water licence is
written against — this is the formulation to use, and it is not the one that wins
on head.

## 4. Negative results, and one that was my own bug

**Fault representation did not help.** `fault:none` (1.200 m) *beat* the smeared
anisotropy baseline (1.486 m), with identical jump recovery (0.931 vs 0.931).
`sidefeat` and `faultcoord` were both worse on the jump criterion (0.661, 0.660
vs 0.580). Given the null-model finding, the honest reading is that the fault
criterion could not resolve these options at all: the hard IC supplied the jump
and no variant improved on it. This is a **benchmark design flaw**, now
addressed by a `fault_*_evolution_rmse_m` metric that subtracts each model's own
t₀ jump, leaving only the 0.29 m of jump evolution the model is responsible for.

**Causal weighting "hurting" was a units bug, not a finding.** The screen and a
sweep over ε = 1, 0.1, 0.01 all reported large degradation. All three were
measuring the same degenerate behaviour: `w_i = exp(−ε·Σ Lⱼ)` used the raw
cumulative loss, and with per-bin losses starting near 10⁴, *any* ε above ~10⁻⁴
sends every bin after the first to exactly zero weight. The scheme had silently
become "train on the first time bin only". Normalising the cumulative loss makes
ε dimensionless; the corrected re-test is reported in the results tables. Had I
not checked, this would have been a confident wrong negative about a
well-established method.

**Loss balancing barely mattered** (1.29–1.54 across fixed / gradnorm / NTK /
RBA). On a problem where the formulation gap is 2×, arguing about weighting
schemes is rearranging deck chairs.

## 5. Nobody conserved mass, and nobody solved the inverse problem

**Mass balance**: every variant scored 0.67–0.88 local imbalance against a metric
floor of **0.079**. The control-volume form was best (0.672) — as it must be,
being trained on exactly that residual — but not close to the floor. Conservation
is an *asymptotic* property here, not something bought cheaply.

**Inverse**: no parameterisation recovered the spatial pattern of K (pattern
correlation 0.04–0.22 across all five inverse runs). What *is* interpretable is
the failure mode:

| parameterisation | log₁₀ RMSE | log₁₀ bias | pattern corr |
|---|---|---|---|
| `fv+kl` | 0.853 | −0.52 | 0.11 |
| `grid` | 0.863 | −0.26 | 0.04 |
| `fv+grid` | 0.869 | −0.51 | 0.22 |
| `kl` | 0.975 | −0.59 | 0.10 |
| `net` | **2.647** | **−2.46** | 0.05 |

The network parameterisation drifts two and a half orders of magnitude in level,
while the grid and KL parameterisations stay within 0.3–0.6. That is exactly what
theory predicts: a Fourier-featured MLP's implicit prior is not a statement about
aquifer structure, so nothing anchors the level of a field that the head data
constrain only through a divergence. Use an explicit prior — grid + TV, or a KL
expansion built from the variogram.

But the headline is the negative one: **240 s of CPU does not solve a 10,000-cell
inverse problem**, and no amount of choosing between parameterisations changes
that.

## 6. What this study does not establish

- **Not a convergence comparison.** Everything is far from converged. The
  budget-sensitivity runs (strong vs control-volume at 4× compute) are the only
  evidence here about whether the ranking survives more compute; see the results
  tables.
- **Single seed** for the screening runs.
- **Single layer.** Nothing tests vertical discretisation or the quasi-3D leakage
  of the full five-layer benchmark.
- **The fault criterion was weak** for the reason in §4; the evolution metric was
  added afterwards and the screening rows predate it.
- **No uncertainty quantification** at all.
- **Neural operators were not tested** — deliberately, since they answer a
  different question (amortising over many solves). Given that the formulation
  axis dominated everything else, an FNO or DeepONet trained on the benchmark's
  exported multi-band head stacks is the obvious next experiment.

## 7. If you have to pick one thing

For this class of problem — heterogeneous, faulted, tidally forced, and judged on
water balance:

1. **Use a control-volume residual.** It was the only formulation that beat
   persistence, and it did so from the fewest iterations.
2. **Add an explicit flux output if the river budget is the deliverable.** The
   mixed form owned that criterion and nothing else came close.
3. **Use an explicit geostatistical prior for K** (grid+TV or KL), never a bare
   coordinate network.
4. **Spend your effort on the formulation, not the architecture or the loss
   weights.** Those moved things by tens of percent; the formulation moved them
   by a factor of two.
5. **Score the null model first.** It is the cheapest experiment in this whole
   study and it invalidated more claims than anything else.
