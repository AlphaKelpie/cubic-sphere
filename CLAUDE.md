# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Numerical simulation of reaction-diffusion equations on a hypersphere (S^3 embedded in R^4), applied to zebra pattern formation. The core algorithm discretises a 4D volume on a regular grid, solves a PDE (diffusion or the full Laplace-Beltrami operator) using explicit Euler or Crank-Nicolson time-stepping with fixed-point iteration, then projects the density back onto the sphere surface.

## Build

There is no Makefile or CMakeLists. Compilation is done manually with g++ (MinGW). Requires C++20 (`std::format`, `std::filesystem`).

Build the main 4D simulation (current active work on `c-n-bazz` branch):

```bash
g++ -std=c++20 -o cnb.out mainGrid4.cpp grid4.cpp function.cpp
```

Build the simple diffusion program:

```bash
g++ -std=c++20 -o diff.out diffusion.cpp grid4.cpp function.cpp
```

Build the 3D variant:

```bash
g++ -std=c++20 -o main.out mainGrid.cpp grid3.cpp function.cpp
```

## Running

The executable reads `params.dat` (or a file given as argv[1]). The params file uses `key value` pairs (space-separated, one per line). Key parameters: `path` (output prefix), `pathdata` (input data prefix), `T`, `h`, grid bounds (`w_min`/`w_max`/...), sphere `radius` and center (`cw`/`cx`/`cy`/`cz`), `total` (simulation steps), `step` (save interval).

Output goes to `data/` as `*_s.dat` (surface/volume), `*_p.dat` (projection), `*_n.dat` (neighbour), `*_r.dat` (density rho). On first run Grid4 creates these from scratch (slow); subsequent runs load them and skip to evolution.

## Tests

Python scripts in `test/` validate output data files. They require numpy (and matplotlib/scipy for `test_diffusion.py`). Run from the `test/` directory:

```bash
python test_projection.py [simulation_name] [data_path]
python test_neighbour.py [simulation_name] [data_path]
python test_rho.py [simulation_name] [data_path]
python test_diffusion.py [simulation_name] [data_path]
```

Defaults: simulation name `test4_1`, data path `../data`.

## Architecture

- **`utils.hpp`** — `Interval`, `Point` (3D), `Quaternion` (4D) structs. `Point::D(i)` and `Quaternion::D(i)` compute diffusion coefficients (diagonal: sum of other components squared; off-diagonal: negative product of two components). These are used by the Laplace-Beltrami discretisation.
- **`Surface<T>`** (header-only template, `.cpp` included from `.h`) — owning array wrapper with copy/move semantics. Stores the volume grid points.
- **`Function`** — owning `double[]` wrapper for the density field `rho`. `operator[](negative_index)` returns 0.0 (boundary convention for neighbours at domain edges, stored as -1).
- **`Grid3`** / **`Grid4`** — the 3D (sphere in R^3) and 4D (hypersphere in R^4) grid classes. Each manages the volume, neighbour stencil, projection map, and density. Key methods:
  - Constructor builds the grid, neighbour connectivity (33 points for Grid4: self + 8 axis-aligned + 24 diagonal pairs), and projection map (closest-point projection onto the sphere via signed distance `phi` and gradient descent).
  - `evolve()` — explicit Euler step with full Laplace-Beltrami operator.
  - `evolveCN()` — Crank-Nicolson with Laplace-Beltrami (fixed-point iteration).
  - `evolveCNDiff()` — Crank-Nicolson for pure diffusion (flat Laplacian only).
  - `project()` — interpolates density from volume grid to sphere surface using precomputed 16-point (Grid4) or 8-point (Grid3) weights.
  - `der1`, `der2`, `derij` — finite difference stencils for first, second, and mixed second derivatives. The overloads taking a `Function const&` parameter are used by the Crank-Nicolson implicit step.
- **`Params`** — singleton loaded from `params.dat`. Must call `Params::load()` before `Params::get()`.
- **`mainGrid4.cpp`** — main entry point for the hypersphere simulation. Creates or loads grid data, runs evolution loop, saves snapshots.
- **`mainGrid.cpp`** — same for the 3D sphere variant.
- **`diffusion.cpp`** — standalone simple-diffusion simulation using `evolveCNDiff()`.

## Branches

- `main` — 3D sphere simulation (Grid3).
- `c-n-bazz` — active development: 4D hypersphere with Crank-Nicolson Laplace-Beltrami (Grid4).
- `par` — parallel execution experiment.

## Info for Claude

If you need to execute codes, you must use WSL commands, e.g.:

```ps
wsl g++ -std=c++20 -o cnb.out mainGrid4.cpp grid4.cpp function.cpp
```

For python you need to use an WSL environment, e.g.:

```pw
wsl /mnt/e/env/Accel/python test_neighbour.py
```
