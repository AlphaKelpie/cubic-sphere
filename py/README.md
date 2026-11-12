# cubic-sphere on the GPU (JAX port of `mainGrid4.cpp` / `corKDTf0.out`)

A JAX port of trondini's `cubic-sphere` Crank–Nicolson solver (C++ source in
`/home/trondini/code/cubic-sphere`, copy in `/home/gregb/corKDT_src`). It runs
the same `params*.dat` files on one GPU.

```bash
source /home/gregb/venvs/jax/bin/activate
python main.py /home/gregb/corKDT_src/params01.dat            # full run
python main.py params01.dat --max-steps 20 --out-prefix /data01/gregb/cubic-sphere/pilot/rad5_01   # pilot
```

## Differences from the C++ (read this before comparing results)

| | C++ | JAX default | flag for C++ behaviour |
|---|---|---|---|
| coefficients `D_ij`, drift | raw grid coordinates `p` (≈ (2,2,2,2) ± 1) | `p - center` | `--raw-coords` |
| `rho[(N-1)/2] = 0` every step | yes (an arbitrary shell point in the crown model) | no | `--keep-center-sink` |
| output | text `_r.dat`, ~2 GB each | float32 `.npy`, same point order | `--text` |
| neighbour / projection tables | stored (~105 GB RAM, 250 GB text) | derived from the grid on the fly | – |

**Why the defaults differ.** With raw coordinates the coefficients are about
16× too large. The fixed-point (Jacobi) solve of the CN system then has
spectral radius ≈ 1.44 > 1 and diverges, so every step runs all 1000
iterations and rho becomes NaN. That is what happened to the `rad5_01` run
started on Nov 6: every value in `rad5_01_evolving_0010_r.dat` and
`_0030_r.dat` is NaN. With centered coordinates the spectral radius is ≈ 0.07
and Jacobi converges in 2–3 iterations.

Everything else is the same scheme: CN with Jacobi (tol 1e-3, max 1000
iterations), the same 33-point stencil (`der1`, `der2`, `derij` with the ±h
shifted `D_ij`), Dirichlet-zero outside the crown, the final `project()`, and
the same step/save loop.

## Layout

The 398,718,028 crown points (h = 0.01) are stored in 157,103 dense 8⁴
"bricks", 2.6 GB per float32 field. Only bricks that contain points are
kept. Neighbours come from the grid structure plus a small brick-neighbour
table; a missing neighbour (C++ index −1) reads 0 from an all-zero brick.

- `params.py`: `params*.dat` parser (same `stof` rounding, `total + 2`)
- `geometry.py`: regenerates the C++ point set and order (`createVol`) and builds the bricks; cached in `/data01/gregb/cubic-sphere/cache`
- `ops.py`: `alpha*L(rho)` and the jitted CN step
- `project.py`: recomputes the closest-point stencil (`createMaps`) and `project()`
- `rhoio.py`: C++ text reader (cached as .npy), point-order conversion, async writer
- `validate.py`: checks against the C++ formulas and trondini's data files

## Validation

- `validate.py small`: on an h = 0.05 grid, `alpha*L` matches a NumPy reference written like the C++ (with an independently built C++-style neighbour table) to 1e-16 in float64 and 3e-7 in float32, in both coordinate modes. One CN step matches exactly, including the Jacobi iteration count.
- `validate.py files`: the regenerated point set equals `sph01_s.dat` (all 398,718,028 rows); sampled rows of `sph01_n.dat` and `sph01_p.dat` match.
- `validate.py spectral` / `physics`: Jacobi spectral radius in raw vs centered mode; mode decay.

## Performance (Tesla T4, params01.dat, 398,718,028 points)

| | C++ (87 cores) | JAX (1 × T4) |
|---|---|---|
| one Jacobi iteration | ~13 s | ~0.4 s |
| one CN step | ~3.6 h (1000 diverging iterations) | 1.18 s (2 iterations) |
| block of 10 steps + snapshot | ~1.5 days | ~15.6 s |
| `project()` | – | 24 s |
| full run (10,000 steps, 1000 snapshots) | ~4 years (NaN) | ~4.4 h, ~1.6 TB of .npy |

One-time setup (cached afterwards): geometry ~70 s, parsing the init `_r.dat` ~2 min.

## Note on the discretization (inherited from the C++)

The crown has zero values just outside |r − 1| = deltaC and is never
re-extended from the sphere during the evolution. So mass leaks out through
the crown boundary. On the h = 0.05 test grid (deltaC = 0.15), a constant
initial rho loses mass at a rate of 33.5 per unit time. Relative to that, a
degree-1 harmonic decays at 3.24, close to the continuum value of 3 for
Brownian motion on S³. So the angular dynamics are right, but the boundary
leak dominates the total mass. This is a property of the C++ scheme, not of the port; it's worth
discussing with trondini.
