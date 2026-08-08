# What actually won, and what that means

Narrative conclusions from the formulation study. Raw tables are in
[`pinn_formulation_results.md`](pinn_formulation_results.md) (generated from the
`results.csv` files under `runs/`); the design and the SOTA survey are in
[`pinn_formulation_design.md`](pinn_formulation_design.md).

> **Scope.** 33 training runs at 240 s each on 4 CPU threads, plus two at 960 s
> and a seed replication, against a 100×100-cell, 48-stress-period single-layer
> MODFLOW reference. This is an equal-compute screening study, not a convergence
> study. Read §8 before quoting any of it.

---

## 1. The result that reframes everything: the null model

The initial condition is a hard constraint (`h = h_init + s(t)·hs·NN`) — a
deliberate choice, since it removes a loss term and makes every variant start
identically. The consequence only became clear when the *null model* was scored:

| persistence — a model that learns nothing | |
|---|---|
| head RMSE | **0.912 m** |
| head NSE | **0.9899** |
| fault jump RMSE | **0.558 m** |
| fault jump "recovery" | **0.928** |

Doing nothing already reproduces 93% of the fault jump, because the true initial
field — discontinuity included — is handed over.

**Of 19 forward screening runs, exactly four beat persistence on head. All four
are control-volume runs.** Every strong-form and mixed-form variant, including
the baseline at 1.486 m, is *worse than doing nothing*.

**Nothing beat persistence on the fault criterion.** Best skill: +0.002.

A three-seed replication (§6) puts error bars on that and forces one claim to be
weakened. What survives:

| claim | evidence |
|---|---|
| strong form is **worse** than persistence | t = −5.4, n=3 — solid |
| mixed form is **worse** than persistence | t = −7.8, n=3 — solid |
| control-volume is **better than the strong form** | t = +5.2, n=3 — solid |
| control-volume is **better than persistence** | t = +1.0, n=3 — **not established** |

So the defensible statement is *not* "the control-volume form beats doing
nothing" — one of its three seeds (0.974 m) does not. It is: **the control-volume
form is the only one that is not decisively worse than doing nothing, and it is
decisively better than the alternatives.**

Two lessons. Always score the null model — without it this study would have
reported a tidy ranking of methods that mostly could not beat a constant. And
`head_nse ≈ 0.99` is meaningless here: the spatial head range is 32 m while the
transient signal is ~1 m, so NSE is dominated by static structure. Read RMSE and
skill-vs-persistence instead.

## 2. Formulation beats architecture, decisively

Best variant on each axis, head RMSE, against persistence at 0.912 m:

| axis | best variant | head RMSE | skill vs null |
|---|---|---|---|
| **form** | `fv` (control volume) | **0.772** | **+0.15** |
| arch | `mlp+rwf` | 0.940 | −0.03 |
| fault | `none` | 1.200 | −0.32 |
| balance | `ntk` | 1.292 | −0.42 |
| temporal | `causal` (ε scale-free) | 1.330 | −0.46 |

Overall ranking, head:

| rank | variant | head RMSE | iters |
|---|---|---|---|
| 1 | `fv + sidefeat` | **0.703** | 1,743 |
| 2 | `fv + causal (ε scale-free)` | 0.754 | 1,742 |
| 3 | `fv` | 0.772 | 1,794 |
| 4 | `fv + pirate` | 0.775 | 1,070 |
| — | *persistence null* | *0.912* | — |
| 5 | `mlp + rwf` | 0.940 | 4,034 |
| … | `baseline` (strong, modified MLP) | 1.486 | 2,028 |

The control-volume formulation wins on head, fault jump and mass balance
simultaneously — while completing the **fewest iterations** (1,070–1,794 against
up to 4,172). It is not winning on compute; it is winning on structure.
Enforcing the balance over control volumes, where face fluxes telescope, is worth
more than three times as many collocation steps.

Seed-to-seed variation is ~13% (§6), so **the differences *within* the
control-volume family — `fv+sidefeat` 0.703, `fv+causal` 0.754, `fv` 0.772,
`fv+pirate` 0.775 — are not resolvable.** Treat rows 1–4 as one result. The gap
between that family and everything else (0.844 ± 0.117 against 1.512 ± 0.191) is
what is real.

Architecture changes moved head RMSE by ±30%, which is barely outside the ±13%
seed noise. Changing the formulation moved it by a factor of two.
**That ordering is the most useful thing this study says.**

**PirateNet lost at equal wall clock** (2.14 m, worst architecture) — not because
it is bad but because it costs ~2× per step and got 1,300 iterations against
4,172. At equal *iterations* it would likely look very different. This is the
equal-compute framing doing its job: it answers "what should I run for the next
ten minutes", not "what is best asymptotically".

## 3. The mixed form is a specialist, not a generalist

Once its flux output was correctly scaled (the screening run asked the network to
emit ~0.02 instead of ~1 — a fairness bug I fixed and re-ran), the mixed form was:

- **best of all 33 runs on river exchange**: 0.120 relative error, against 0.152
  for the best strong-form and 0.205 for the best control-volume run;
- still poor on head (1.61 m, worse than persistence).

Coherent rather than noisy: the mixed form carries the Darcy flux as an explicit
network output, and river leakage *is* a flux. If the deliverable is an
aquifer–river exchange budget — the number a surface-water licence is written
against — this is the formulation to use, and it is not the one that wins on head.

## 4. Two apparent findings that were my own bugs

**Causal weighting "hurting" was a units bug.** The screen and a sweep over
ε = 1, 0.1, 0.01 all reported large degradation. All three measured the same
degenerate behaviour: `w_i = exp(−ε·Σⱼ<ᵢ Lⱼ)` used the *raw* cumulative loss, and
with per-bin losses starting near 10⁴, *any* ε above ~10⁻⁴ sends every bin after
the first to exactly zero weight. The scheme had silently become "train on the
first time bin only". After normalising the cumulative loss so ε is
dimensionless:

| | head RMSE |
|---|---|
| `time:causal` (ε with units) | 2.527 |
| `causal-norm:eps1` | **1.330** — better than the 1.486 baseline |
| `causal-norm:eps5` | 1.656 |
| `fv + causal-norm:eps2` | **0.754** — better than plain `fv` at 0.772 |

So causal weighting **helps**, modestly, on both the strong and control-volume
forms. Had I not re-tested, this study would have published a confident wrong
negative about a well-established method.

**The fault criterion was too weak to resolve anything.** `fault:none` (1.200 m)
*beat* the smeared-anisotropy baseline (1.486 m) with identical jump recovery
(0.931 vs 0.931); `sidefeat` and `faultcoord` were worse on the jump criterion.
Given §1 and §6, the honest reading is that the criterion could not distinguish these
options at all — the hard IC supplied the jump and nothing improved on it. That
is a **benchmark design flaw**, now addressed by a `fault_*_evolution_rmse_m`
metric that subtracts each model's own t₀ jump, leaving the 0.29 m of jump
evolution the model is actually responsible for. The screening rows predate that
metric, so the fault column in this study should be treated as uninformative.

**Loss balancing barely mattered** (1.29–1.54 across fixed / gradnorm / NTK /
RBA). Where the formulation gap is 2×, arguing about weighting schemes is
rearranging deck chairs.

## 5. What more compute does

At 4× budget (960 s):

| | 240 s | 960 s |
|---|---|---|
| strong: head | 1.486 (2,028 it) | 1.262 (8,419 it) |
| strong: mass | 0.822 | 0.764 |
| strong: river | 0.197 | 0.133 |
| **fv: head** | **0.772** (1,794 it) | **0.826** (7,015 it) |
| fv: mass | 0.672 | **0.503** |

The strong form improves steadily but **still does not beat persistence at four
times the compute**. Fitting `err ∝ iters^−a` gives a ≈ 0.115, implying roughly
17× more iterations — about **4.5 CPU-hours** — merely to match a constant. The
control-volume form beat persistence in four minutes.

The anomaly: **FV got 7% worse on head while its mass balance improved 25%.**
That pattern is the signature of converging to a *different* solution, so I
tested the obvious candidate — that the reference's one backward-Euler step per
2 h period damps the 12.4 h tide, while the surrogates solve the continuous-time
problem. **That hypothesis is refuted:** re-solving with 16× finer stepping
changes the head field by only **0.018 m RMSE** (0.94% amplitude damping), two
orders of magnitude below the effect. The time-discretisation floor is
negligible.

**Resolved: it is noise.** Across three seeds at 240 s, `form:fv` scores
0.844 ± 0.117 m. The 960 s value of 0.826 m sits **−0.2σ** from that mean — i.e.
four times the compute produced no detectable change at all, in either direction.
The apparent degradation was a single-seed artefact and there is no plateau to
explain.

What *did* improve with compute is the mass balance (0.672 → 0.503), which is a
real effect and much larger than the seed spread.

## 6. How much of this is noise?

Three seeds each, 240 s, head RMSE:

| variant | mean | sd | spread | CV |
|---|---|---|---|---|
| `form:fv` | **0.844** | 0.117 | 0.745 – 0.974 | 13.9% |
| `baseline` (strong) | 1.512 | 0.191 | 1.317 – 1.699 | 12.6% |
| `form:mixed` | 1.680 | 0.171 | 1.546 – 1.872 | 10.2% |
| *persistence* | *0.912* | — | — | — |

**Seed noise is ~13%.** That is the resolution limit of every single-seed number
in this study, and it means:

- differences below ~25% between single-seed runs are not interpretable — which
  covers the whole architecture axis, the whole balance axis, and every ordering
  within the control-volume family;
- the formulation gap (0.844 vs 1.512, t = +5.2) is far outside it and is real;
- `form:fv` vs persistence (t = +1.0) is *inside* it and is not established.

Re-running seed 0 under different CPU load also drifted results by 0.2–3.5%
(baseline 1.486 → 1.520), which is the cost of an equal-wall-clock protocol and
is small next to the seed effect.

## 7. Nobody conserved mass, and nobody solved the inverse problem

**Mass balance**: every variant scored 0.50–0.88 local imbalance against a metric
floor of **0.079**. The control-volume form was best — as it must be, being
trained on exactly that residual — and improved most with compute (0.672 →
0.503), but never approached the floor. Local conservation is an *asymptotic*
property here, not something bought cheaply.

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

The network parameterisation drifts two and a half orders of magnitude in level;
grid and KL stay within 0.3–0.6. Exactly what theory predicts: a Fourier-featured
MLP's implicit prior is not a statement about aquifer structure, so nothing
anchors the level of a field the head data constrain only through a divergence.
Use an explicit prior.

But the headline is negative: **240 s of CPU does not solve a 10,000-cell inverse
problem**, and choosing between parameterisations does not change that.

## 8. What this study does not establish

- **Not a convergence comparison.** Everything is far from converged. §5 is the
  only evidence about what more compute does, and it covers two variants.
- **Seed replication covers three variants at three seeds.** Everything else is
  single-seed, and with ~13% seed noise no single-seed difference below ~25%
  should be believed — which includes every ranking within the control-volume
  family and most of the architecture axis.
- **n = 3 is small.** "Control-volume beats persistence" is suggestive at
  t = +1.0 and would need more seeds to settle.
- **Single layer.** Nothing tests vertical discretisation or the quasi-3D leakage
  of the full five-layer benchmark.
- **The fault criterion is uninformative** for the reason in §4. The evolution
  metric that fixes it was added after the screen ran.
- **The inverse task is unresolved**, not won by KL — no parameterisation
  recovered the pattern.
- **No uncertainty quantification** at all.
- **Neural operators were not tested** — deliberately, since they answer a
  different question (amortising over many solves). Given that the formulation
  axis dominated everything else, an FNO or DeepONet trained on the benchmark's
  exported multi-band head stacks is the obvious next experiment.

## 9. If you have to pick one thing

For this class of problem — heterogeneous, faulted, tidally forced, judged on
water balance:

1. **Use a control-volume residual.** The only formulation that is not decisively
   worse than doing nothing, decisively better than the alternatives (t = +5.2),
   achieved from the fewest iterations, and the only one whose mass balance
   improved materially with compute.
2. **Add causal weighting with a scale-free ε.** Free, and it helped both forms.
   Check that your ε is dimensionless before concluding anything about it.
3. **Add an explicit flux output if the river budget is the deliverable.** The
   mixed form owned that criterion and nothing else came close.
4. **Use an explicit geostatistical prior for K** (grid+TV or KL), never a bare
   coordinate network — though expect none of them to recover the pattern at
   realistic compute.
5. **Spend effort on the formulation, not the architecture or the loss weights.**
   Those moved things by tens of percent; the formulation moved them by 2×.
6. **Score the null model first.** The cheapest experiment here, and it
   invalidated more claims than anything else.
