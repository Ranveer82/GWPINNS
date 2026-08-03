# Method notes

Detail that would clutter the README: why the equations are written the way they
are, and what each numerical choice is defending against.

---

## 1. Why the aquifer is confined

The target equation is

$$S_s \frac{\partial h}{\partial t} = \nabla\cdot(K\nabla h) + W$$

which is the **confined** form: storage is elastic, transmissivity does not
depend on head, and the problem is linear in `h` for fixed `K`.

MODFLOW's unconfined option (`icelltype /= 0`) would replace `K` by a saturated
thickness that itself depends on `h`, so the forward model would no longer be
solving the equation the PINN residual implements. Any mismatch between the two
would then be indistinguishable from inversion error, which defeats the purpose
of a benchmark. Layer 0 is best read as a linearised water-table layer.

## 2. The source term is known, not fitted

`W` splits into three pieces, all reproduced exactly as the discrete forward
model applied them:

| Piece | Form | Support |
|---|---|---|
| Wells | `Q / V_cell` | the screened cell, during pumping stress periods |
| Recharge | `R(x,y) / Δz₀` | layer 0 |
| Evapotranspiration | `−ET(h) / Δz₀` | layer 0 |

Recharge and ET are applied as *cell sources in the top layer*, not as a flux
boundary condition at `z = top`. This mirrors MODFLOW exactly: the top face of
layer 0 is a no-flow boundary, and RCH/EVT enter the cell balance as sources.
Applying them as a Neumann condition instead would be equivalent in the
continuum limit but not cell-for-cell, and the discrepancy would show up as
apparent inversion error near the water table.

ET follows the MODFLOW linear ramp and is evaluated on the model's **own
predicted head**, so it stays differentiable and the PINN genuinely solves a
head-dependent sink problem. The inverse problem is never handed the true ET
field.

## 3. Non-dimensionalisation and why `K₀` cancels

Write `x = x_c + a_x X` (and similarly for `y`, `z`, `t`), and
`h = h_c + Δh·H`. Substituting and dividing by `S_s Δh / a_t`:

```
d_T H  =  α · Σ_i μ_i² d_i( K d_i H )  +  β · W
```

with `μ_i = L0/a_i`, `α = a_t/(S_s L0²)`, `β = a_t/(S_s Δh)`, and `L0 = a_x` so
that `μ_x = 1`.

`K` appears in physical units. There is **no reference conductivity in the
residual** — a scaling constant chosen near the true value would otherwise be a
quiet leak of the answer. `K₀` reappears only in the mixed formulation as the
unit of the flux output, where it cancels between the Darcy and continuity
residuals and is therefore a pure unit choice (`K₀ = 1 m/d`).

For this benchmark `α ≈ 0.29`, `β ≈ 3.1 × 10⁵`, and `μ = (1, 1.67, 83.3)`.

## 4. The aspect-ratio problem

`μ_z² ≈ 6.9 × 10³`. The vertical diffusion term therefore carries a coefficient
four orders of magnitude larger than the horizontal one. This is not a modelling
artefact — it is the physics of a 5 km × 60 m aquifer, whose vertical
equilibration time is ~0.2 d against ~300 d horizontally. Heads really are
nearly hydrostatic in the vertical.

The consequences are real and are reported rather than hidden:

1. At initialisation the PDE residual is ~10⁵ times the data misfit, essentially
   all of it in the `z` term. Gradient-norm loss balancing is what makes the
   first few thousand iterations spend their budget on something other than
   flattening `∂H/∂z`.
2. Because heads barely vary vertically, `K(z)` is weakly identifiable. The
   partially penetrating wells (screened in layers 0, 1 and 2 respectively) are
   the only strong vertical signal in the design.

## 5. Why K collapses, and the three things that stop it

Where the head field is locally smooth and `W ≈ 0`, the residual
`d_T H − α Σ μ_i² d_i(K d_i H)` is satisfied by driving `K → 0`, for *any*
smooth head field. The inverse problem is genuinely degenerate away from the
wells and the observation points, and gradient descent finds that solution
reliably and fast.

Three defences, each necessary:

- **Bounded `log₁₀ K` via tanh.** Keeps K physical. Note the failure mode this
  *creates*: at saturation the tanh gradient vanishes, so a field that has
  already collapsed to the lower bound can never climb back. Hence the next two.
- **Initialisation mid-range** (`log₁₀ K = 0`, i.e. 1 m/d) rather than at the
  centre of the bounds (`10^-0.5`).
- **Physics curriculum.** Data-only warm-up, then a linear ramp of the physics
  weights. Enabling the PDE at iteration zero against an untrained head field
  collapses K before the head has any information in it.
- **Weak Tikhonov prior**, faded out as the physics ramps in. Full-strength
  throughout, the prior simply pins K to the prior value; see the prior ablation.

The prior's bulk value (1 m/d) is deliberately *not* the benchmark's true
geometric mean (1.93 m/d), and the ablation sweeps it over two orders of
magnitude to quantify how much of the answer it is buying.

## 6. Source-weighted residual

Inside a pumping cell `β·W ≈ 10³`, against `O(1)` in the bulk. Squared, the well
cells outweigh everything else by six orders of magnitude, and an unweighted
mean-square residual is effectively a well-cell-only loss.

Dividing the residual by `1 + |β W|` targets comparable *relative* accuracy
everywhere without moving the residual's zero. `residual_weighting="none"`
disables it, and the weighting ablation reports the difference.

## 7. The cPINN interface, and why plain continuity is wrong

Standard conservative PINNs impose continuity of the solution and of its flux
across a subdomain interface. Here, flux continuity is correct and
unconditional:

```
Q_n^west = Q_n^east
```

Head continuity is **not**. A barrier fault exists precisely to sustain a head
jump — the benchmark's barrier holds 8.6 m across a 150 m zone. Imposing
`H_west = H_east` forces the model to explain that jump by distorting the
conductivity field on both sides.

The correct condition for a thin, low-permeability feature is a leaky wall
obtained by homogenising the zone away:

```
Q_n = Γ (H_west − H_east),    Γ = C L0 / K0,    C = K_fault / width
```

imposed as `[Γ(H_w − H_e) − Q_n] / (1 + Γ)`, which degenerates gracefully to
`Q_n = 0` as `Γ → 0` and to `H_w = H_e` as `Γ → ∞`.

`Γ` is a learnable field over the fault plane (or a single scalar), so the
architecture *reports* the fault's hydraulic behaviour as a number rather than
requiring it to be read off a recovered K image.

### The identifiability caveat

For a strong barrier the cross-fault flux is near zero, and the condition
`Q_n = Γ ΔH` is then satisfied by *any* sufficiently small `Γ`. The data
constrain an **upper bound** on leakance, not a point value. The barrier/conduit
classification is robust; the fitted magnitude, in the barrier direction, is not.
This is a property of the physics, not of the optimiser, and it is reported as
such in the results.

## 8. Sampling

The residual is dominated by two thin features that together occupy well under
1% of the domain: the pumping cells and the fault zone. Uniform sampling puts
almost no points in either.

The sampler draws a configurable fraction directly inside each, using exact
inverse geometry — sample the signed fault distance and solve
`d = (x−x₀)n_x + (y−y₀)n_y` for `x` — rather than rejection sampling, which
would be prohibitive at a 150 m zone in a 5000 m domain.

Interface points are placed on the fault mid-plane and then stepped **along the
plane normal** to the two walls, so the paired points at `d = ∓w/2` are genuinely
opposite each other. Stepping in `x` alone would pair points offset along the
fault, and the flux-continuity residual would be comparing different locations.
