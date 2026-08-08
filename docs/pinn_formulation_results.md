# Which physics-informed formulation wins on this benchmark?

Results of the controlled screening study defined in [`pinn_formulation_design.md`](pinn_formulation_design.md) and run by `scripts/compare_pinn_formulations.py`.

- Reference: MODFLOW 6 reduced case, 100x100 cells, 48 stress periods of 2 h
- Budget: **240 s of wall clock per run** on 4 CPU threads (equal compute, not equal iterations)
- Runs: 42 completed
- Generated: 2026-08-08 01:51

## Metric floors

Every metric applied to the MODFLOW reference itself, so the numbers below are interpretable:

- `mass_local_err_frac` = **0.07875**
- `mass_global_err_frac` = **0.02119**
- `mass_global_err_m3d` = **1.752e+05**
- `river_total_rel_err` = **6.025e-07**
- `river_cell_rmse_m3d` = **0.01309**
- `river_nse` = **1**
- `river_sign_agreement` = **1**

## The persistence baseline

The initial condition is a hard constraint, so a model that learns nothing still reproduces the true head field at t0 - including the 2.25 m head jump already present across the barriers. Every number below must be read against this:

- `null_head_rmse_m` = **0.9125**
- `null_head_nse` = **0.9899**
- `null_fault_jump_rmse_m` = **0.5584**
- `null_fault_jump_recovery` = **0.9277**
- `ref_transient_amplitude_m` = **1.089**

## head simulation

_head RMSE (m), lower is better_

| name                     |   score_head |   skill_vs_null |   iters |   head_nse |   fault_all_jump_recovery |   k_pattern_corr |
|:-------------------------|-------------:|----------------:|--------:|-----------:|--------------------------:|-----------------:|
| combo:fv+sidefeat        |       0.7027 |         0.2299  |    1743 |     0.994  |                    0.9244 |              nan |
| form:fv                  |       0.7455 |         0.183   |    1804 |     0.9933 |                    0.9332 |              nan |
| fv+causal-norm:eps2      |       0.7543 |         0.1734  |    1742 |     0.9931 |                    0.9337 |              nan |
| form:fv                  |       0.7724 |         0.1535  |    1794 |     0.9928 |                    0.9335 |              nan |
| combo:fv+pirate          |       0.775  |         0.1507  |    1070 |     0.9927 |                    0.9299 |              nan |
| combo:fv+causal          |       0.7934 |         0.1305  |    1333 |     0.9924 |                    0.9343 |              nan |
| form:fv                  |       0.8133 |         0.1088  |    1790 |     0.992  |                    0.9252 |              nan |
| budget:fv                |       0.8264 |         0.09434 |    7015 |     0.9917 |                    0.9279 |              nan |
| -- persistence (null) -- |       0.9125 |         0       |     nan |     0.9899 |                  nan      |              nan |

## head across faults

_RMSE of the head jump (m), lower is better_

| name                     |   score_fault |   skill_vs_null |   iters |   head_nse |   fault_all_jump_recovery |   k_pattern_corr |
|:-------------------------|--------------:|----------------:|--------:|-----------:|--------------------------:|-----------------:|
| form:fv                  |        0.557  |       0.002464  |    1794 |     0.9928 |                    0.9335 |              nan |
| combo:fv+causal          |        0.557  |       0.002389  |    1333 |     0.9924 |                    0.9343 |              nan |
| form:fv                  |        0.5571 |       0.002217  |    1804 |     0.9933 |                    0.9332 |              nan |
| budget:fv                |        0.5583 |       0.0001821 |    7015 |     0.9917 |                    0.9279 |              nan |
| -- persistence (null) -- |        0.5584 |       0         |     nan |   nan      |                    0.9277 |              nan |
| arch:mlp+rwf             |        0.5594 |      -0.001917  |    4034 |     0.9893 |                    0.928  |              nan |
| fv+causal-norm:eps2      |        0.5602 |      -0.003192  |    1742 |     0.9931 |                    0.9337 |              nan |
| form:mixed               |        0.5608 |      -0.004323  |    4150 |     0.9711 |                    0.9361 |              nan |
| form:mixed               |        0.5611 |      -0.004897  |    3344 |     0.972  |                    0.9338 |              nan |

## mass balance

_local imbalance fraction, lower is better_

| name                |   score_mass |   iters |   head_nse |   fault_all_jump_recovery |   k_pattern_corr |
|:--------------------|-------------:|--------:|-----------:|--------------------------:|-----------------:|
| budget:fv           |       0.5028 |    7015 |     0.9917 |                    0.9279 |              nan |
| form:fv             |       0.6341 |    1790 |     0.992  |                    0.9252 |              nan |
| fv+causal-norm:eps2 |       0.6663 |    1742 |     0.9931 |                    0.9337 |              nan |
| form:fv             |       0.6723 |    1794 |     0.9928 |                    0.9335 |              nan |
| form:fv             |       0.679  |    1804 |     0.9933 |                    0.9332 |              nan |
| combo:fv+causal     |       0.6942 |    1333 |     0.9924 |                    0.9343 |              nan |
| combo:fv+sidefeat   |       0.7002 |    1743 |     0.994  |                    0.9244 |              nan |
| combo:fv+pirate     |       0.7271 |    1070 |     0.9927 |                    0.9299 |              nan |

## river exchange

_relative error in total exchange, lower is better_

| name                    |   score_river |   iters |   head_nse |   fault_all_jump_recovery |   k_pattern_corr |
|:------------------------|--------------:|--------:|-----------:|--------------------------:|-----------------:|
| mixed+sidefeat:rescaled |        0.1203 |    4085 |     0.9688 |                    0.9304 |              nan |
| baseline                |        0.1307 |    2049 |     0.979  |                    0.9179 |              nan |
| budget:strong           |        0.1327 |    8419 |     0.9807 |                    0.926  |              nan |
| causal-norm:eps1        |        0.1519 |    2073 |     0.9786 |                    0.9344 |              nan |
| arch:mlp+rwf            |        0.1609 |    4034 |     0.9893 |                    0.928  |              nan |
| bal:fixed               |        0.1612 |    2197 |     0.9781 |                    0.934  |              nan |
| form:mixed              |        0.1641 |    4150 |     0.9711 |                    0.9361 |              nan |
| mixed:rescaled          |        0.1707 |    4348 |     0.9684 |                    0.9387 |              nan |

## inverse (K recovery)

_log10 RMSE of K, lower is better_

| name        |   score_inverse |   iters |   head_nse |   fault_all_jump_recovery |   k_pattern_corr |
|:------------|----------------:|--------:|-----------:|--------------------------:|-----------------:|
| inv:fv+kl   |          0.8534 |    1114 |     0.9945 |                    0.9362 |          0.1055  |
| inv:grid    |          0.8633 |    2110 |     0.9895 |                    0.9542 |          0.04202 |
| inv:fv+grid |          0.8691 |    1726 |     0.9954 |                    0.9367 |          0.2175  |
| inv:kl      |          0.9751 |    1820 |     0.9904 |                    0.9509 |          0.1022  |
| inv:net     |          2.647  |    1900 |     0.9883 |                    0.9548 |          0.04562 |

## Effect of each axis relative to the baseline

_percent change; negative is an improvement_

| name                    | group    |   head |   fault |   mass |   river |
|:------------------------|:---------|-------:|--------:|-------:|--------:|
| form:mixed              | form     |    2.4 |    -3.3 |   -0.7 |    99.6 |
| form:fv                 | form     |  -48.0 |    -4.0 |  -18.2 |    12.8 |
| arch:mlp                | arch     |  -13.9 |    -2.6 |   -3.4 |    -7.5 |
| arch:pirate             | arch     |   44.3 |     2.2 |    2.0 |    38.7 |
| arch:spinn              | arch     |   -3.8 |    -3.1 |    4.9 |    45.0 |
| arch:mlp+rwf            | arch     |  -36.7 |    -3.6 |   -2.5 |   -18.1 |
| fault:none              | fault    |  -19.3 |    -0.8 |   -0.6 |    20.3 |
| fault:sidefeat          | fault    |   -5.8 |    13.9 |   -1.1 |    16.7 |
| fault:faultcoord        | fault    |   54.8 |    13.7 |    0.9 |    38.6 |
| time:causal             | temporal |   70.1 |    55.3 |    6.7 |   300.8 |
| time:march              | temporal |   36.1 |    24.0 |    2.2 |   333.1 |
| bal:fixed               | balance  |   -9.5 |    -0.0 |   -1.0 |   -18.0 |
| bal:ntk                 | balance  |  -13.1 |    -0.6 |   -1.1 |     5.7 |
| bal:rba                 | balance  |    3.5 |     0.8 |   -0.5 |    28.3 |
| combo:fv+causal         | combo    |  -46.6 |    -4.0 |  -15.6 |   158.3 |
| combo:fv+sidefeat       | combo    |  -52.7 |    -1.6 |  -14.8 |     8.1 |
| combo:mixed+sidefeat    | combo    |   -1.1 |    -1.9 |   -1.6 |    82.9 |
| combo:fv+pirate         | combo    |  -47.8 |    -1.1 |  -11.6 |    19.2 |
| mixed:rescaled          | followup |    8.8 |    -0.8 |   -3.5 |   -13.2 |
| mixed+sidefeat:rescaled | followup |    8.1 |     3.0 |   -3.2 |   -38.8 |
| causal:eps0.1           | followup |   86.8 |    25.2 |    5.1 |   445.2 |
| causal:eps0.01          | followup |   36.5 |     0.5 |    1.4 |    27.6 |
| budget:strong           | budget   |  -15.1 |    -1.4 |   -7.1 |   -32.5 |
| budget:fv               | budget   |  -44.4 |    -3.8 |  -38.9 |     4.1 |
| causal-norm:eps1        | causal   |  -10.5 |     0.0 |   -1.1 |   -22.7 |
| causal-norm:eps5        | causal   |   11.5 |    -0.6 |    0.1 |    -4.3 |
| fv+causal-norm:eps2     | causal   |  -49.2 |    -3.5 |  -19.0 |     8.1 |
| form:mixed              | form     |    9.1 |    -0.9 |   -3.7 |   -11.0 |
| form:fv                 | form     |  -49.8 |    -4.0 |  -17.4 |    18.0 |
| form:mixed              | form     |   26.0 |    -3.0 |    0.3 |    29.4 |
| form:fv                 | form     |  -34.5 |    -3.1 |   -3.6 |    55.5 |
| form:mixed              | form     |    4.1 |    -3.4 |   -3.8 |   -16.6 |
| form:fv                 | form     |  -45.3 |    -3.0 |  -22.9 |    27.6 |
