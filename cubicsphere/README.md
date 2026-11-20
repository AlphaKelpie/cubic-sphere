# cubicsphere: JAX/GPU port of `mainGrid4.cpp`

Same params file, same flow, same output names as the C++ executable
(`<path>_init`, `<path>_evolving_NNNN`, `<path>_evolved`).

```bash
python -m venv ~/venvs/cubic-jax && ~/venvs/cubic-jax/bin/pip install -e ".[cuda]"
~/venvs/cubic-jax/bin/python -m cubicsphere params.dat --estimate-spectrum   # check dt first
~/venvs/cubic-jax/bin/python -m cubicsphere params.dat --max-steps 20        # pilot
~/venvs/cubic-jax/bin/python -m cubicsphere params.dat                       # full run
```

## Differences from the C++

| | C++ | JAX default | C++ behaviour |
| --- | --- | --- | --- |
| coefficients `D_i`, `D_ij` and drift | raw grid coordinates `p ≈ (2,2,2,2) ± 1` | `x = p − center` | `--raw-coords` |
| linear solve of CN | fixed point `x ← rhs + αLx` | Jacobi `x ← x + (rhs + αLx − x)/(1 − αL_diag)` | `--solver fixed-point` |
| stopping test | `max\|Δx\| < 1e-3` (absolute) | `max\|Δx\| ≤ 1e-5 · max\|x\|` | `--atol 1e-3 --rtol 0` |
| not converged / NaN | runs all 1000 iterations, keeps going | stops with an error | `--on-fail continue` |
| centre sink `rho[(n−1)/2] = 0` | always (arbitrary point in the crown model; uninitialised index when the grid is created) | hypercube model only | `--center-sink on` |
| storage | point list + 33 neighbours + 16 projection entries per point | dense 8⁴ bricks, neighbours and projection computed on the fly | – |
| output | text `_r.dat` | float32 `_r.npy`, same point order, plus `<path>_params.dat` (copy of the params; `test/test_rho.py` rebuilds the point radii from it) | `--format text` |
| `_s/_n/_p.dat` | written on creation | not needed, not written | – |
| precision | float storage, double arithmetic | float32 (`--dtype float64` for checks) | – |

**Why raw coordinates are wrong.** The operator
`12ρ + 3Σ x_i ∂_iρ + Σ D_i ∂_iiρ + Σ_{i≠j} ∂_i∂_j(D_ij ρ)`, with `D_i = |x|² − x_i²`,
`D_ij = −x_i x_j`, is `P:∇²ρ − 3x·∇ρ` with `P = |x|²I − xxᵀ`: the Fokker–Planck
(Laplace–Beltrami) operator of Brownian motion on the sphere **centred at the
origin**. The grid sphere is centred at `(cw, cx, cy, cz) = (2,2,2,2)`, so `x` must be
`p − center`. With raw `p`, `|p|² ≈ 16` instead of 1: the coefficients are ~16×
too large and describe a different sphere.

**Why the C++ produced NaN.** The fixed-point iteration converges only if the
spectral radius of `αL` is < 1, and it grows like `α/h² · |x|²`. Measured with
`--estimate-spectrum` on the h = 0.05 grid (`sim_2`):

| deltaT | fixed point, raw (C++) | fixed point, centred | Jacobi, centred |
| --- | --- | --- | --- |
| 1e-5 | 0.55 | 0.026 | 0.013 |
| 1e-4 | **5.5** | 0.26 | 0.11 |
| 1e-3 | 55 | **2.6** | 0.54 |
| 1e-1 | – | 264 | 0.97 |

At fixed `α/h²` the numbers are the same for any h, so at h = 0.01 the C++
diverges from deltaT ≈ 7e-7 (raw), the centred fixed point from ≈ 1.5e-5,
while Jacobi stays below 1 (converging more slowly as dt grows). The C++
`deltaT 1e-4` run on `sim_2` took 143 s for one step and turned all 636,844
values to NaN; the JAX default does the same step in 5 iterations.

Large dt is stable but not necessarily accurate: CN damps the shortest modes
poorly (negative undershoots appear). Check a result by halving deltaT.

## Validation (CPU, against this repository's C++)

- `python -m cubicsphere.validate PARAMS data/sim_2` (crown) and `data/base2`
  (hypercube): same point set and order as `_s.dat`; αL equals a NumPy
  re-implementation that uses the C++ `_n.dat` neighbour table to 1e-16 (both
  coordinate modes); `project()` equals the C++ `_p.dat` table to the 6 printed digits.
- End to end against the compiled C++ (loading mode, `sim_2`, deltaT 1e-5, 3
  steps, C++ flags): every snapshot equal to ≤ 1.5e-7 relative, same iteration
  counts and mass. Hypercube (`base2`): ≤ 5e-6 (text print precision).
- Not tested on a GPU here; run a pilot (`--max-steps`) on the T4 first.

## Memory (h = 0.01, deltaC = 0.1: 398,718,028 points)

About 157k bricks, 2.6 GB per float32 field. A Jacobi step holds 3–4 fields
plus the mask (~11 GB): it should fit a 16 GB T4 (estimate, not measured). `bicgstab` needs ~8 fields and
only fits coarser grids. Lower `--chunk` if XLA runs out of memory. Host RAM:
~10 GB peak.

## Inherited from the C++ scheme (not changed)

- Dirichlet zero just outside the crown: mass leaks through the crown
  boundary (the "shell" initial condition loses ~0.35 % per step at h = 0.05,
  deltaT 1e-4), while a point source gains mass slightly (~0.1 %/step), probably
  because `12ρ + 3x·∇ρ` is not discretised in conservative form (not verified).
- `project()` is only applied at the end, so the density also spreads radially
  during the whole run.
- `total` in the params file means `total + 2`, as in the C++ `Params`.
