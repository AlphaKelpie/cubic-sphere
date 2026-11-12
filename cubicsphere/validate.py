"""Check the JAX port against data files written by the C++ code.

    python -m cubicsphere.validate PARAMS PREFIX [--rho FILE_PREFIX]
    python -m cubicsphere.validate --compare A_PREFIX B_PREFIX

PREFIX is a C++ data set (``PREFIX_s.dat``, ``_n.dat``, ``_p.dat``) created with
the same h / deltaC / radius / bounds as PARAMS. The script checks that:

1. the point set and its order equal ``_s.dat``;
2. alpha*L(rho) computed with the C++ neighbour table (NumPy, float64, the
   formulas of ``Grid4::evolveCN``) equals the brick-layout JAX operator, in
   both coordinate modes;
3. ``project()`` with the C++ ``_p.dat`` table equals the JAX projection.

``--compare`` prints the differences between two rho files (text or npy).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _read_table(filename: str, ncols: int, dtype=np.float64) -> np.ndarray:
    with open(filename, "rb") as fin:
        header = fin.readline().split()
        n = int(header[0])
        data = np.fromfile(fin, dtype=dtype, sep=" ")
    return data.reshape(n, ncols)


def cpp_alpha_L(vol, nbr, rho, alpha, h, coords):
    """alpha * L(rho) with the C++ neighbour table, like Grid4::evolveCN (float64)."""
    f = np.concatenate([rho.astype(np.float64), [0.0]])  # index -1 -> 0
    F = f[nbr]  # [n, 33]
    p = coords
    L = 12.0 * F[:, 0]
    r2 = (p**2).sum(axis=1)
    for i in range(4):
        L += 3.0 * p[:, i] * (F[:, 2 + 2 * i] - F[:, 1 + 2 * i]) / (2 * h)
        D = r2 - p[:, i] ** 2
        L += (D * (F[:, 2 + 2 * i] - F[:, 0]) + D * (F[:, 1 + 2 * i] - F[:, 0])) / (h * h)
    # derij neighbour slots (++, +-, -+, --) per pair, from Grid4::derij
    slots = {(0, 1): (16, 15, 10, 9), (0, 2): (18, 17, 12, 11), (0, 3): (20, 19, 14, 13),
             (1, 2): (26, 25, 22, 21), (1, 3): (28, 27, 24, 23), (2, 3): (32, 31, 30, 29)}
    for (i, j), (pp, pm, mp, mm) in slots.items():
        def D(di, dj):
            return -(p[:, i] + di) * (p[:, j] + dj)
        res = F[:, pp] * D(h, h) - F[:, pm] * D(h, -h) - F[:, mp] * D(-h, h) + F[:, mm] * D(-h, -h)
        L += 2.0 * res / (h * h * 4)  # (i,j) and (j,i)
    return alpha * L


def validate(params_file: str, prefix: str, rho_prefix: str | None):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from .geometry import get_geometry
    from .params import load_params
    from .rhoio import CppOrder, read_rho
    from .solver import Simulation, SolverOptions

    p = load_params(params_file)
    geo = get_geometry(p, None, verbose=False)
    order = CppOrder(geo)
    ok = True

    vol = _read_table(prefix + "_s.dat", 4)
    n = len(vol)
    print(f"[1] point set: C++ {n:,}, JAX {geo.npoints:,}", end=" ")
    if n != geo.npoints:
        print("-> MISMATCH")
        return False
    # coordinates of the JAX points in C++ order
    coords = []
    for a in range(4):
        tab = p.mins[a] + p.h * np.arange(int(geo.lo[a] + 8 * geo.nbricks_axis[a]))
        shape = [1, 1, 1, 1]
        shape[a] = 8
        local = np.arange(8).reshape(shape)
        grid = tab[geo.origins[:, a][:, None, None, None, None] + local[None]]  # [nb, ...] broadcast
        flat = np.broadcast_to(grid, (geo.nb, 8, 8, 8, 8)).reshape(-1)
        coords.append(np.concatenate([flat[s * 4096 + perm] for s, perm in
                                      zip(geo.slab_start[:-1], order.perms)]))
    coords = np.stack(coords, axis=1)
    err = np.abs(coords - vol).max()
    print(f"max coordinate difference {err:.2e}", "-> OK" if err < 1e-5 else "-> MISMATCH")
    ok &= err < 1e-5

    nbr = _read_table(prefix + "_n.dat", 33, np.int64).copy()
    nbr[nbr < 0] = n  # -> the appended 0
    if rho_prefix:
        rho = np.asarray(read_rho(rho_prefix), dtype=np.float64)
    else:
        rho = np.random.default_rng(1).random(n)
    for raw in (True, False):
        opts = SolverOptions(raw_coords=raw, dtype="float64", chunk=64)
        sim = Simulation(geo, p, opts)
        field = sim.to_device(order.from_cpp(rho.astype(np.float32)).astype(np.float64))
        rho32 = rho.astype(np.float32).astype(np.float64)
        # exact coordinates (``_s.dat`` only has 6 printed digits)
        pc = np.float32(coords).astype(np.float64) if raw else coords - np.array(p.center)
        ref = cpp_alpha_L(vol, nbr, rho32, p.alpha, p.h, pc)
        got = _to_cpp64(order, sim._apply_aL(field))
        scale = np.abs(ref).max()
        e = np.abs(got - ref).max() / scale
        name = "raw (C++)" if raw else "centred"
        print(f"[2] alpha*L, {name} coordinates: max |alpha L| {scale:.3e}, relative max error {e:.2e}",
              "-> OK" if e < 1e-9 else "-> MISMATCH")
        ok &= e < 1e-9

    proj = _read_table(prefix + "_p.dat", 32)
    pidx = proj[:, 0::2].astype(np.int64)
    pw = proj[:, 1::2]
    ref = np.zeros(n)
    np.add.at(ref, pidx.ravel(), (rho32[:, None] * pw).ravel())
    got = _to_cpp64(order, sim.project(field))
    e = np.abs(got - ref).max() / np.abs(ref).max()
    # _p.dat weights are printed with 6 significant digits
    print(f"[3] project(): relative max error {e:.2e}", "-> OK" if e < 1e-4 else "-> MISMATCH")
    ok &= e < 1e-4
    mass = (ref.sum(), got.sum(), rho32.sum())
    print(f"    mass before {mass[2]:.10g}, after C++ {mass[0]:.10g}, after JAX {mass[1]:.10g}")
    return bool(ok)


def _to_cpp64(order, field):
    """Like CppOrder.to_cpp but keeps float64."""
    geo = order.geo
    out = np.empty(geo.npoints)
    for bw, perm in enumerate(order.perms):
        s, e = int(geo.slab_start[bw]), int(geo.slab_start[bw + 1])
        if e > s:
            out[order.offsets[bw]:order.offsets[bw + 1]] = np.asarray(field[s:e]).reshape(-1)[perm]
    return out


def compare(a: str, b: str):
    from .rhoio import read_rho

    x, y = np.asarray(read_rho(a), np.float64), np.asarray(read_rho(b), np.float64)
    if len(x) != len(y):
        sys.exit(f"different lengths {len(x)} vs {len(y)}")
    d = np.abs(x - y)
    i = int(np.argmax(d))
    print(f"n = {len(x):,}; NaN: {np.isnan(x).sum()} / {np.isnan(y).sum()}")
    print(f"sum {x.sum():.10g} vs {y.sum():.10g}; max {np.nanmax(x):.6g} vs {np.nanmax(y):.6g}")
    print(f"max |diff| {d.max():.3e} at {i} ({x[i]:.6g} vs {y[i]:.6g}); relative to max {d.max() / np.abs(x).max():.3e}")
    print(f"points with |diff| > 1e-3 * max: {(d > 1e-3 * np.abs(x).max()).sum()}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cubicsphere.validate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("args", nargs="+")
    ap.add_argument("--rho", default=None, help="rho file prefix to test L with (default: random)")
    ap.add_argument("--compare", action="store_true")
    o = ap.parse_args(argv)
    if o.compare:
        compare(*o.args)
        return 0
    return 0 if validate(o.args[0], o.args[1], o.rho) else 1


if __name__ == "__main__":
    sys.exit(main())
