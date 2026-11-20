# cubicsphere: notes on the GPU port of `mainGrid4.cpp`

Working notes of how the JAX/GPU implementation (`cubicsphere/`) was reached: the
problem, the theory behind the operator, every test and what it proved, and the open points.
They are written in the order the work happened. Statuses are marked:

- **derived**: follows from the code or from algebra shown here;
- **measured**: obtained by running something (where and with what is stated);
- **estimate / not verified**: reasoned or quoted from someone else, not checked by us.

> **Maintenance rule.** Update this file whenever the implementation, a result or an open
> point changes (add a dated line to the *Changelog* at the end and fix the sections it
> affects). Do not let it describe code that no longer exists.

Branch: `gpu`. Last updated: 2026-11-20.

---

## 0. Summary

| Question | Answer |
|---|---|
| Why did the C++ run produce NaN? | It evaluates the operator coefficients at **raw grid coordinates** `p ≈ (2,2,2,2)+x` instead of `x = p − center`. Coefficients are ~16× too large, the fixed-point (Jacobi-like) iteration of the CN system has spectral radius > 1 and diverges (§3, §4). |
| Is the JAX port equivalent to the C++? | Yes, when run with the C++ settings: operator equal to 1e-16 (float64), snapshots equal to ≤ 1.5e-7 relative, projection to the printed digits of `_p.dat` (§6). Defaults differ on purpose (§5.2). |
| Why are C++ and default-JAX results different? | Raw coordinates also make the shell spread radially (the correct operator has no radial evolution at all). Reproduced numerically (§9). |
| Why are there negative ρ values? | The spatial stencil is not sign-preserving: the mixed-derivative corner weights (and, near the axes, the central-difference drift) are negative. Not a CN or solver effect (§10). |
| Speed? | See §7. On the T4 only numbers quoted from `mod.md` exist; nothing was run on a GPU by us. |

---

## 1. Problem

`mainGrid4.cpp` simulates a Fokker–Planck equation for random orientation (gyroscope noise,
see `README.md`) as diffusion on the hypersphere S³ ⊂ ℝ⁴ (radius 1, centre (2,2,2,2)).
The sphere is embedded in a regular 4D grid (step `h`) and only a **crown** `| |x|−R | ≤ deltaC`
is stored (`createVol`), or the whole hypercube if `deltaC = 0` (`createVolNeigh`).
Time stepping: Crank–Nicolson (CN), the linear system solved by fixed-point iteration;
at the end `project()` moves the density onto the sphere.

Observed (user report + `mod.md`, the report of a sibling implementation in `py/`):

- `params.dat` with `h = 0.01`, `deltaC = 0.1`: **398,718,028** points (`mod.md`). Memory for the
  stored tables ~105 GB (`mod.md`).
- ~13 s per Jacobi iteration on 87 cores, ~3.6 h per CN step (1000 non-converging iterations),
  all ρ become NaN (`mod.md`, these timings are quoted, not measured by us).

## 2. The C++ scheme, as read from the code (derived)

Grid4 (`grid4.cpp`):

- 33-point stencil: the point, 8 axis neighbours, 24 corner neighbours (pairs of axes, ±h ±h).
  Missing neighbour = index −1 = value 0 (Dirichlet zero outside the crown).
- `Lrho` in `evolveCN`, with `x` = the coordinate `p` stored in `_volume`:

  ```
  L ρ = 12 ρ
      + Σ_i 3 p_i (ρ(+e_i) − ρ(−e_i)) / (2h)                          [der1]
      + Σ_i D_i(p) (ρ(+e_i) + ρ(−e_i) − 2ρ) / h²                      [der2]
      + Σ_{i≠j} [ ρ(++) D_ij(+h,+h) − ρ(+−) D_ij(+h,−h)
                  − ρ(−+) D_ij(−h,+h) + ρ(−−) D_ij(−h,−h) ] / (4h²)    [derij]
  D_i(p) = Σ_{j≠i} p_j² ,   D_ij(a,b) = −(p_i + a)(p_j + b)
  ```
  The loop runs over ordered pairs, so every unordered pair counts twice.
- CN: `rhs = ρⁿ + α L ρⁿ`, solve `(I − αL) ρⁿ⁺¹ = rhs` by `x ← rhs + α L x`, stop when
  `max|Δx| < 1e-3` (absolute) or after 1000 iterations; `α = (T/6)·dt/2`,
  `dt = deltaT`, or `0.1 h⁴` when `deltaT = 0`.
- `rho[(N−1)/2] = 0` after every step (an arbitrary point in the crown model).
- `project()`: each point's density is spread on the 16 grid vertices around its
  *closest point on the sphere* with multilinear weights; the closest point uses
  `c = x − ∇ₕφ · φ/|∇ₕφ|` with the central-difference gradient (step `h`) of `φ = |x−c₀| − R`.

## 3. Theory: what the continuous operator is (derived)

Write `P(x) = |x|² I − x xᵀ` (so `P_ii = D_i`, `P_ij = D_ij`).

**Claim 1.** `L = P:∇² − 3 x·∇`.

*Proof.* For `i ≠ j`: `∂_i∂_j(−x_i x_j ρ) = −ρ − x_i ∂_iρ − x_j ∂_jρ − x_i x_j ∂_ijρ`.
Summing over the 12 ordered pairs: `−12ρ − 6 x·∇ρ − Σ_{i≠j} x_i x_j ∂_ijρ`
(each `x_i∂_i` appears 3 times as first and 3 times as second index).
Adding `12ρ + 3 x·∇ρ + Σ D_i ∂_iiρ`:
`L = Σ_i D_i ∂_ii − Σ_{i≠j} x_i x_j ∂_ij − 3 x·∇ = P:∇² − 3 x·∇`. ∎

**Claim 2.** For `f(x) = u(r) Y(ω)` (`r = |x|`, `ω = x/r`): `L f = u(r) Δ_{S³} Y(ω)`.

*Proof.* `∇²f = u''Y + 3u'Y/r + uΔ_SY/r²` and `x̂ᵀ∇²f x̂ = u''Y`. Hence
`P:∇²f = r²(∇²f − u''Y) = 3 r u' Y + u Δ_S Y`, and `−3x·∇f = −3 r u' Y`. Sum: `uΔ_SY`. ∎

Consequences:

1. At `r = 1`, `L` is exactly the Laplace–Beltrami operator of S³; eigenvalues `−l(l+2)`
   (−3 for degree 1). `mod.md` reports a measured decay 3.24 for a degree-1 harmonic on a
   coarse grid, consistent with this (quoted, not repeated by us).
2. `P x = 0`: **there is no radial diffusion and no radial transport**. Every shell `r` evolves
   independently; any radial spreading of the density is a numerical artefact or an effect of
   wrong coefficients.
3. The constant `T/6` and `dt` only rescale time.

**Claim 3 (raw coordinates).** The discrete code uses `p = c₀ + x` in `P` and in the drift, so what is
integrated is `P(p):∇² − 3 p·∇` = the same operator but tangential to spheres
centred at the **box corner** `p = 0`, at radius `|p| ≈ 4` (Claim 2 with `x → p`). Seen from the
true sphere around `c₀`, the radial direction `x̂` is not annihilated by `P(p)`:

`D_rr = |p|² − (p·x̂)²`, average over the shell `= (16+1) − (4+1) = 12` instead of 0
(`⟨(c₀·x̂)²⟩ = |c₀|²/4 = 4`, cross term averages to 0).

Coefficients are also larger by `|p|²/|x|² ≈ 16` (up to ~26 at the far side of the crown).

## 4. Why the C++ iteration diverges (derived + measured)

The fixed-point iteration `x ← rhs + αLx` converges iff `ρ(αL) < 1`.

- Diagonal of the discrete `L`: `12 − 2Σ_i D_i/h² = 12 − 6|x|²/h²` (since `Σ_i D_i = 3|x|²`;
  the drift and the corner terms have no diagonal entry).
- The largest-magnitude eigenvalue of the axis second differences is the checkerboard mode,
  `−4 D_i/h²` per axis, so `ρ(αL) ≈ 12 α |x|²/h²`. Convergence condition (heuristic):
  `dt < h² / (T |x|²_max)`.

Measured with `--estimate-spectrum` (power iteration), `sim_2` grid, h = 0.05, deltaC = 0.1:

| deltaT | fixed point, raw (C++) | fixed point, centred | Jacobi, centred |
|---|---|---|---|
| 6.25e-7 (default 0.1h⁴) | 0.0347 | 0.00165 | 0.00080 |
| 1e-5 | 0.554 | 0.0264 | 0.0127 |
| 1e-4 | **5.54** | 0.264 | 0.113 |
| 4e-4 | 22.2 | **1.055** | 0.332 |
| 1e-3 | 55.4 | 2.64 | 0.544 |
| 1e-2 | – | 26.4 | 0.903 |
| 1e-1 | – | 264 | 0.970 |

(Jacobi raw: 0.0177 / 0.222 / 0.721 / 0.885 / 0.926 for the first five rows.)

The heuristic threshold matches the measured crossing within ~10–20% (centred: 3.4e-4 predicted
vs ~3.8e-4 measured at h = 0.05; raw: predicted 1.6e-5 vs ~1.8e-5, with `|p|²_max ≈ 26`).
Radii scale with `α/h²`, so at **h = 0.01** the C++ iteration diverges for `deltaT ≳ 7e-7` (raw) and the
centred fixed point for `≳ 1.5e-5`, in line with the ratio 1.44 quoted in `mod.md`.

Direct confirmation (measured): the compiled C++ on `sim_2`, `deltaT = 1e-4`: **one step took 143 s
and all 636,844 values became NaN**; the JAX default does the same step in 5 iterations, 1.6 s (CPU).

Jacobi (diagonal scaling `x ← x + (rhs + αLx − x)/(1 − α L_diag)`) stayed below 1 in every case
tried, up to `dt = 0.1`, but slows down as `dt` grows. This is **empirical, no proof**: the corner and drift
terms break strict diagonal dominance. Large `dt` is stable but not necessarily accurate (undershoots
appear); halve `deltaT` to check a result.

## 5. Design of `cubicsphere/`

### 5.1 Data layout

The active points are stored in dense **8⁴ bricks**; only bricks containing points are kept, plus an
all-zero brick that missing neighbours point to. A brick's neighbours come from the brick grid
(32 neighbouring bricks cover the 33-point stencil), so the C++ `_n.dat`/`_p.dat` tables
(≈ 105 GB RAM) are not needed. Per-brick temporaries are built chunk by chunk (`lax.map`).
`mod.md` gives for h = 0.01: 157,103 bricks, 2.6 GB per float32 field (quoted). A Jacobi step holds 3–4 fields
plus the mask: ~11 GB, **estimate** for the 16 GB T4 (not measured).

Modules: `params.py` (same `stof` rounding, `total + 2`), `geometry.py` (point set and order of
`createVol`, bricks, cache), `solver.py` (operator, CN step, projection, spectrum estimate),
`rhoio.py` (C++ order ↔ bricks, `.npy`/text I/O, background writer), `validate.py`, `__main__.py`.

### 5.2 Deliberate differences from the C++

| | C++ | cubicsphere default | C++ behaviour |
|---|---|---|---|
| coefficients | raw `p` | `x = p − center` | `--raw-coords` |
| CN solve | fixed point | Jacobi | `--solver fixed-point` (also `bicgstab`) |
| stopping | `max|Δx| < 1e-3` | `max|Δx| ≤ 1e-5 max|x|` | `--atol 1e-3 --rtol 0` |
| not converged / NaN | continues | stops with a message | `--on-fail continue` |
| centre sink | always | hypercube model only | `--center-sink on` |
| default dt | `0.1h⁴` if `deltaT = 0` | same | – |
| output | text `_r.dat` | float32 `_r.npy` (+ `<path>_params.dat`) | `--format text` |
| precision | float storage, double arithmetic | float32 (`--dtype float64`) | – |

Full C++ reproduction: `--raw-coords --solver fixed-point --atol 1e-3 --rtol 0 --center-sink on --on-fail continue --format text`.

### 5.3 Fix found during the comparison with `py/` (§7)

All device arrays (mask, neighbour table, coordinate tables…) were captured by the `jit`-compiled
functions, so XLA embedded them as **constants**. At h = 0.01 the mask alone is ~650 MB; this would blow up
compile time and duplicate it on the GPU. They are now passed as an argument (`Simulation._jit`).
Measured at h = 0.05: compiled program 8.8 MB → 148 KB. Validation unchanged afterwards.

### 5.4 Bugs fixed in my own first versions

- Jacobi loop started with `diff = inf` and the `isfinite` guard stopped it before the first sweep
  (found by the C++ comparison: no evolution). Now starts from the largest finite value.
- First `project()` scaled points radially onto the sphere; the C++ uses the central-difference
  gradient. Projection error against `_p.dat` went 8.4e-5 → 4.2e-6 (crown) and 3.9e-3 → 1.6e-7 (hypercube).
- A validation reference was built from the 6-digit `_s.dat` coordinates (error 3e-8 appeared as a
  "mismatch"); now the exact coordinates are used.

## 6. Validation against the C++ (measured, CPU, WSL)

| Test | Data | Result |
|---|---|---|
| point set and order = `_s.dat` | `sim_2` (636,844 points, h = 0.05, crown) | all points, max coordinate diff 4.5e-8 |
| αL vs NumPy rewrite using the C++ `_n.dat` table | `sim_2` and `base2` (hypercube h = 0.2) | relative error ~1e-16 (float64), raw and centred |
| `project()` vs C++ `_p.dat` table | same | 4.2e-6 (crown), 1.6e-7 (cube); `_p.dat` prints 6 digits |
| end to end vs compiled C++ (loading mode) | `sim_2`, deltaT 1e-5, 3 steps, C++ flags | evolving snapshots ≤ 1.5e-7 relative, same iteration counts, same mass; evolved 5e-6 |
| hypercube end to end | `base2`, deltaT 1e-4 | ≤ 5e-6 (text print precision) |

Run: `python -m cubicsphere.validate PARAMS data/<set>` (point set, αL, projection),
`python -m cubicsphere.validate --compare A B` (two rho files).

## 7. Comparison with the other implementation (`py/`)

Both use the same brick layout, stencil and centred coordinates by default. Differences (read from the code):

- `py/` ignores the C++ default `dt = 0.1h⁴` (`alpha = T/6·deltaT/2`): with `deltaT 0` nothing evolves.
  (Read in code, not run.)
- `py/` implements only the C++ fixed-point iteration with absolute tol 1e-3, runs all 1000 iterations on
  divergence, and can only stop on NaN at the next snapshot.
- `py/` computes `project()` in float64 reproducing the C++ weight rounding; ours is float32 (3× faster on
  CPU; float64 is much slower on a T4).
- `py/` copies snapshots to host and reorders them in the writer thread; ours reorders on the main thread
  (the GPU waits). Worth adopting.
- Default paths in `py/` point under `/data01/gregb`.

Benchmark (CPU of the development machine, no GPU; same seed/start point; fixed-point, abs tol 1e-3,
centred, deltaT 1e-5):

| | h = 0.05 (0.64 M pts) | | h = 0.025 (10.2 M pts) | |
|---|---|---|---|---|
| | `py/` | ours | `py/` | ours |
| CN step (2 iterations) | 0.18 s | 0.10 s | 0.90 s | 0.73 s |
| `project()` | 3 s | 0.4 s | 8 s | 2.5 s |
| total wall | 6.2 s | 4.6 s | 20.0 s | 15.2 s |
| peak RAM | 1.10 GB | 0.65 GB | 1.52 GB | 1.24 GB |

Snapshots of the two agree to ~1e-5 relative (same iteration counts, same mass to 7 digits).
With ours at its defaults (Jacobi, rtol 1e-5) a step takes 3–4 iterations. **T4 numbers exist only in `mod.md`**
for `py/` (≈ 0.4 s/iteration, 1.18 s/step, 24 s `project()`, ≈ 4.4 h for 10,000 steps, ~1.6 TB of `.npy`;
quoted, not reproduced). Ours does the same work per iteration, so similar speed is expected, **not verified**.

## 8. Initial condition `spheric_rho.py` ⇒ `--init shell`

`spheric_rho.py` puts ρ = 1 on every grid point that is a vertex of the 16-point projection stencil of
some sphere point (reading the `_p.dat`, 250 GB at h = 0.01). `--init shell` computes the same set on
the GPU as `project(mask) > 0`.

Measured on the hypercube grid h = 0.16: **17,792 points, identical** to `data/base16_init_r.dat`.
Not tested: the crown path against a script-made file (none available); a stencil vertex with weight
exactly 0 would be counted by the script but not by us. Values are not normalised (total mass = number of
shell points). The shell is only 1–2 cells thick (0.01–0.02 at h = 0.01) inside a crown of ±0.1.

## 9. Analysis script `test/test_rho.py` and the C++ vs `cubicsphere` comparison

### 9.1 The script

`python test_rho.py ../data/<prefix> [--variance] [--negatives clip|abs|keep] [--volume X_s.dat | --params file.dat]`.

- Reads `<prefix>_init_r`, `_evolving_NNNN_r`, `_evolved_r` (`.npy` or C++ text; `-nan` is parsed) and writes
  a 3-panel PNG per snapshot, `<prefix>_radial_stats.csv` and, with `--variance`, `<prefix>_variance.pdf`.
- The radius of each point comes from (in order) `--volume`, `--params`, `<prefix>_s.dat`,
  `<prefix>_params.dat` (copied by `cubicsphere` next to its outputs because it writes no `_s.dat`).
- Negative ρ (§10) used to enter the radial distribution as negative weights, biasing the mean radius and its
  variance (and making the variance possibly negative). `--negatives` chooses the weight: `clip` = max(ρ,0)
  (default), `abs` = |ρ|, `keep` = signed (old behaviour). Per-radius mean/std of ρ stay raw. The signed mass
  (conserved) and the negative mass are printed and written to the CSV (`mass`, `negative_mass`). Synthetic check
  (base2 shell, 1.5 % negative mass added): variance 0.0679 keep / 0.0792 clip / 0.0899 abs, i.e. the
  choice matters; always compare runs with the same option. Clipping here is only for the diagnostic;
  the simulation itself is untouched (the "do not clip" note in §10 concerns the solver).
- Radii are quantised to 1e-6 and accumulated with `bincount` (no per-radius loop). NaNs count as 0 with a
  warning; an all-NaN snapshot skips its plot. The old `IndexError` on crown grids (no radius > 3.5) is fixed.
- Bug found in testing: regenerating coordinates from the params gave 179 distinct radii, the true count
  is 161 (distinct `i²+j²+k²+l²`). Cause: the C++ parses `h` as float and stores coordinates as float
  (0.05 → 0.0500000007); `_s.dat` prints 6 digits and hides ~1e-7 noise. Regenerated coordinates are now rounded
  the same way; both routes then give identical CSVs.

### 9.2 Why the two plots differ (measured, h = 0.1 hypercube, `base1` shell start, deltaT = 1e-5, 400 steps)

| run | mean radius | variance |
|---|---|---|
| initial shell | 1.02367 | 0.009303 |
| cubicsphere **with C++ settings** (raw coords) | 1.16022 | 0.091319 |
| user's C++ plot `radial1_evolving_0400` | 1.16020 | 0.09133 |
| cubicsphere default (centred) | 1.02349 | 0.009189 |
| user's `cubc5_1_evolving_0400` | 1.02354 | 0.00902 |
| centred + C++ stopping rule | 1.02349 | 0.009189 (identical to default) |

Reading: the C++ plot is reproduced to 4 digits by changing only the coordinates; solver and tolerance
are not the cause. In the centred run the variance stays at the shell's initial thickness (Claim 2:
no radial evolution). In the raw run it grows by 0.082; the estimate of §3, Claim 3, gives
`2·12·(T/6)·400·1e-5 ≈ 0.096` (same order). Open: the user's `cubc5_1` variance is 2% below ours
(0.00902 vs 0.00919); params/command of that run not known.

Corollary: the radial over-diffusion described in `README.md` as a discretisation effect is, at least at
h = 0.1 over 400 steps, mostly due to the raw coefficients. Not checked at other `h` or longer times.

## 10. Negative values of ρ (measured + derived)

Setup: hypercube, h = 0.1, shell start, centred coordinates.

**(a) Weights of the discrete operator** (positivity of `dρ/dt = Lρ` needs every neighbour weight ≥ 0).
Corner weights are `−(x_a+h)(x_b+h)/(2h²)`, `+(x_a+h)(x_b−h)/(2h²)`, `+(x_a−h)(x_b+h)/(2h²)`,
`−(x_a−h)(x_b−h)/(2h²)`; axis weights `±3x_a/(2h) + D_a/h²`, negative on one side when `D_a < 3|x_a|h/2`
(near the axes' poles). On the 39,736 shell points:

| neighbours | negative weights | points with ≥ 1 | most negative |
|---|---|---|---|
| 8 axis | 1.1% | 9.1% | −15.0 |
| 24 corner | **44.5%** | **100%** | −36.0 |

**(b) Not the time integration.** Same physical time, different `dt` (min ρ): t = 2e-3: −0.061 for both
dt = 2e-5 and 1e-5; t = 4e-3: −0.0679 for both. (A third run with dt = 5e-6 was stopped, not needed.)

**(c) Term toggling** (dt = 1e-5, min ρ at step 100 → 400):

| operator | step 100 | step 400 |
|---|---|---|
| full | −0.042 | −0.068 |
| no mixed derivative | −0.016 | −0.061 |
| no drift (`12ρ + 3x·∇ρ`) | −0.044 | −0.090 |
| axis second derivatives only | 0 | 0 |

Conclusion: the spatial stencil is not monotone. The mixed-derivative stencil dominates early; the
central-difference drift (no physical radial diffusion to dominate it) contributes later; they partly cancel.
Total mass stays constant (67,374) while max grows to 1.08: redistribution, not loss. The raw-coordinate C++
run has smaller negatives (−0.036) because its strong radial spreading smooths them (consequence of §9.2,
not separately tested).

Options (not implemented): finer `h` and/or a smoother initial condition (expected to help, untested); a
sign-preserving discretisation (positive-weight corner pairs, which need a diagonal-dominance condition
that fails near the poles unless the stencil is widened; upwinding near the poles). Do **not** clip to 0
(breaks mass conservation, hides the artefact).

## 11. Properties inherited from the C++ scheme (not changed)

- Dirichlet zero just outside the crown: mass leaks through the crown edge; `mod.md` reports a constant
  initial ρ losing mass at 33.5 per unit time (h = 0.05, deltaC = 0.15), while a degree-1 harmonic decays at
  3.24 (continuum 3) (quoted). We saw a point source gain ~0.1%/step (h = 0.05, deltaT 1e-4), probably from the
  non-conservative form of `12ρ + 3x·∇ρ` (not verified).
- `project()` only runs at the end, so ρ spreads radially during the whole run (for the raw coefficients; with
  centred ones the numerical radial spread is small, §9.2).
- `total` in the params file means `total + 2`.

Other C++ bugs noticed while reading the code (not fixed): `_centerPoint` is never set when the grid is
created from scratch; `fillFirstRho` can pick index `n` (`uniform_int_distribution(0, _nPoints)` is inclusive);
the unused `evolve()` has the sign of the `12ρ` term flipped compared with `evolveCN`.

## 12. How to run (operational notes)

1. `params.dat`: remove `pathdata` (or give a file made for **this** grid; `data/base4` is h = 0.4 and fails),
   set `deltaT` explicitly (`0` ⇒ 0.1h⁴ = 1e-9 at h = 0.01: nothing moves), unique `path`, sensible `step`
   (each snapshot ~1.6 GB at h = 0.01).
2. `python -m cubicsphere params.dat --estimate-spectrum` (use a coarse copy of the params; Jacobi radius must be < 1).
3. Pilot: `--max-steps 20 --cache-dir <big disk>`; look at "GiB per field", NaN, iterations per step.
   On out-of-memory lower `--chunk` (256, 128).
4. Full run with `nohup ... > run.log 2>&1 &`. Add `--init shell` for the `spheric_rho.py` start.
5. `sudo python` fails (SyntaxError at an f-string): `sudo` resets `PATH`, so the system Python runs instead of
   the venv (probably an old one: f-strings need 3.6+; the version was not checked). Fix: `chown` the cache/output directory to your user, or `sudo "$(which python)" ...`.
6. Development machine has no GPU: tests run on CPU JAX in a WSL venv (`~/venvs/cubic-jax`); the C++ is built with
   `wsl g++ -std=c++20 -O2 ... -ltbb`. Everything in §6–§10 was measured there.

## 13. Open points

- Run on the real T4 at h = 0.01 (compile time, memory, speed; the §5.1 memory figure is an estimate).
- Compare `cubc5_1` (2% variance gap, §9.2) once its params/command are known.
- Crown-model `--init shell` against a script-generated file; optional normalisation flag.
- Monotone discretisation of the mixed derivative and drift (§10) and/or periodic projection during the run.
- Adopt background host-side reordering of snapshots from `py/`; optional float64 projection.
- `test_rho.py` at 398 M points: designed for it (≈ 1.6 GB of radii), not run.

## Changelog

- 2026-11-20: first version (sections 0–13), covering the work up to the analysis of negative ρ values.
- 2026-11-20: `test_rho.py` `--negatives` option (default `clip`) so negative ρ no longer biases the radial mean/variance; CSV gains `negative_mass` (§9.1).
