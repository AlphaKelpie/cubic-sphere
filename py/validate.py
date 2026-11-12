"""Validation of the JAX port against the C++ formulas and data files.

    python validate.py small  PARAMS          # operator vs NumPy reference (x64), physics checks
    python validate.py files  PARAMS          # against trondini's sph01_{s,n,p}.dat files
    python validate.py spectral PARAMS        # power iteration, raw vs centered coefficients
"""

import os
import sys
import time

import numpy as np
import os as _os
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")  # T4: ~10 GB for 3 fields + mask; leave room for CUDA modules

import geometry
from geometry import NOFF
from params import Params

CACHE = "/data01/gregb/cubic-sphere/cache"
# derij table entries (C++ point[] indices) for (lo, hi) direction pairs:
# f(++), f(+-), f(-+), f(--)
DERIJ = {(0, 1): (16, 15, 10, 9), (0, 2): (18, 17, 12, 11), (0, 3): (20, 19, 14, 13),
         (1, 2): (26, 25, 22, 21), (1, 3): (28, 27, 24, 23), (2, 3): (32, 31, 30, 29)}


def point_gidx(geo):
    return geo.grid_index_of_slot(geo.slot)


def lex_keys(g, shape):
    g = g.astype(np.int64)
    return ((g[..., 0] * shape[1] + g[..., 1]) * shape[2] + g[..., 2]) * shape[3] + g[..., 3]


def ref_neighbours(gidx_all, rows, shape):
    """C++-style 33-entry neighbour rows (self + NOFF order, -1 if absent),
    computed by key lookup in the lexicographically sorted point list."""
    keys = lex_keys(gidx_all, shape)
    assert np.all(np.diff(keys) > 0), "points not in lexicographic order"
    g = gidx_all[rows].astype(np.int64)
    out = np.full((len(rows), 33), -1, dtype=np.int64)
    out[:, 0] = rows
    for k, off in enumerate(NOFF):
        q = g + off
        ok = np.all((q >= 0) & (q < np.array(shape)), axis=1)
        qk = lex_keys(np.where(ok[:, None], q, 0), shape)
        pos = np.searchsorted(keys, qk)
        pos = np.minimum(pos, len(keys) - 1)
        hit = ok & (keys[pos] == qk)
        out[:, k + 1] = np.where(hit, pos, -1)
    return out


def ref_L(f, nb, P, h):
    """Lrho of Grid4::evolveCN, structured like the C++ (float64)."""
    def F(k):
        idx = nb[:, k]
        return np.where(idx >= 0, f[np.maximum(idx, 0)], 0.0)
    f0 = F(0)
    L = -f0 * (1 - 4) * 4
    for i in range(4):
        der1 = (F(2 + 2 * i) - F(1 + 2 * i)) / (h * 2)
        L -= der1 * P[:, i] * (1 - 4)
        for j in range(4):
            if i == j:
                Di = sum(P[:, k] * P[:, k] for k in range(4) if k != i)
                L += (Di * (F(2 + 2 * i) - f0) + Di * (F(1 + 2 * i) - f0)) / (h * h)
            else:
                a, b = min(i, j), max(i, j)
                pp, pm, mp, mm = DERIJ[(a, b)]
                D = lambda da, db: -(P[:, a] + da) * (P[:, b] + db)
                L += (F(pp) * D(h, h) - F(pm) * D(h, -h) - F(mp) * D(-h, h)
                      + F(mm) * D(-h, -h)) / (h * h * 4)
    return L


def test_function(X, center):
    r = X - np.array(center)
    return np.exp(-np.sum((r - 0.3) ** 2, axis=1)) * (1 + 0.5 * r[:, 0] * r[:, 1] - 0.3 * r[:, 3])


# ---------------------------------------------------------------------------
def cmd_small(pfile):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import rhoio
    from ops import Operator

    p = Params.load(pfile)
    geo = geometry.build(p, CACHE)
    g = point_gidx(geo)
    X = geo.coords(g, p.h)
    print(f"small grid: {geo.n_points} points, {geo.n_bricks} bricks")

    nb = ref_neighbours(g, np.arange(geo.n_points), geo.shape)
    f = test_function(X, p.center)
    alpha = (p.T / 6) * p.deltaT * 0.5
    ok = True
    for centered in (False, True):
        P = X - (np.array(p.center) if centered else 0.0)
        Lref = ref_L(f, nb, P, p.h)
        for dt, tol in ((jnp.float64, 1e-11), (jnp.float32, 2e-3)):
            op = Operator(geo, p.h, alpha, p.center, p.bounds, centered=centered,
                          chunk=64, dtype=dt)
            fld = rhoio.from_points(f, geo.slot, op)
            Ljax = rhoio.to_points(op.alphaL(fld), geo.slot).astype(np.float64) / alpha
            err = np.max(np.abs(Ljax - Lref)) / np.max(np.abs(Lref))
            # float32: L is a sum of large cancelling terms; compare in alpha*L units
            good = err < tol
            ok &= good
            print(f"  L operator {'centered' if centered else 'raw     '} {np.dtype(dt).name:8s}: "
                  f"max rel err {err:.2e}  {'OK' if good else 'FAIL'}")

    # CN step semantics: compare one step against a NumPy CN/Jacobi reference
    P = X - np.array(p.center)
    rhs = f + alpha * ref_L(f, nb, P, p.h)
    A = f.copy()
    for it in range(1000):
        Bv = rhs + alpha * ref_L(A, nb, P, p.h)
        d = np.max(np.abs(Bv - A))
        A = Bv
        if d < 1e-3:
            break
    op = Operator(geo, p.h, alpha, p.center, p.bounds, centered=True, chunk=64, dtype=jnp.float64)
    rho, itj, dj = op.cn_step(rhoio.from_points(f, geo.slot, op))
    out = rhoio.to_points(rho, geo.slot)
    err = np.max(np.abs(out - A))
    good = err < 1e-12 and int(itj) == it + 1
    ok &= good
    print(f"  CN step: iterations jax {int(itj)} / ref {it + 1}, max abs err {err:.2e}  {'OK' if good else 'FAIL'}")
    # project(): GPU implementation vs NumPy reference (C++ arithmetic)
    import project as proj
    op = Operator(geo, p.h, alpha, p.center, p.bounds, centered=True, chunk=64, dtype=jnp.float32)
    vals = np.where(np.random.default_rng(1).random(geo.n_points) < 0.3, 0.0, f).astype(np.float32)
    fld = rhoio.from_points(vals, geo.slot, op)
    ref = rhoio.to_points(proj.project_reference(geo, p, op, vals), geo.slot)
    gpu = rhoio.to_points(proj.project_gpu(geo, p, op, fld, bricks_per_chunk=64), geo.slot)
    err = np.max(np.abs(gpu - ref)) / np.max(np.abs(ref))
    good = err < 1e-5
    ok &= good
    print(f"  project(): GPU vs NumPy reference max rel err {err:.1e}  {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return ok


def cmd_physics(pfile, steps=200):
    import jax.numpy as jnp
    import rhoio
    from ops import Operator

    p = Params.load(pfile)
    geo = geometry.build(p, CACHE)
    g = point_gidx(geo)
    X = geo.coords(g, p.h)
    r = X - np.array(p.center)
    phi1 = r[:, 0] / np.linalg.norm(r, axis=1)  # degree-1 harmonic, 0-homogeneous
    alpha = (p.T / 6) * p.deltaT * 0.5
    t_per_step = p.deltaT  # time unit of the PDE d rho/dt = (T/6) L rho
    for centered in (True, False):
        op = Operator(geo, p.h, alpha, p.center, p.bounds, centered=centered, chunk=64)
        lam = op.spectral_radius(iters=80)
        rho = rhoio.from_points((1.0 + 0.5 * phi1).astype(np.float32), geo.slot, op)
        a0 = None
        its = []
        for s in range(steps):
            rho, it, d = op.cn_step(rho)
            its.append(int(it))
        v = rhoio.to_points(rho, geo.slot).astype(np.float64)
        fin = np.isfinite(v).all()
        print(f"{'centered' if centered else 'raw     '}: spectral radius of alpha*L = {lam:.3f}; "
              f"jacobi iters/step {min(its)}-{max(its)}; finite={fin}")
        if fin:
            tt = steps * t_per_step * p.T / 6
            a1 = np.dot(v, phi1) / np.dot(phi1, phi1)
            a0 = np.mean(v)
            r1 = -np.log(a1 / 0.5) / tt
            r0 = -np.log(a0 / 1.0) / tt
            print(f"   constant mode 1 -> {a0:.5f} (decay rate {r0:.2f}: mass lost through the "
                  f"zero boundary of the crown)")
            print(f"   degree-1 mode 0.5 -> {a1:.5f} (rate {r1:.2f}); rate relative to constant "
                  f"mode {r1 - r0:.2f}  (continuum Brownian motion on S^3: 3.0)")


def cmd_spectral(pfile):
    from ops import Operator
    p = Params.load(pfile)
    geo = geometry.build(p, CACHE)
    alpha = (p.T / 6) * p.deltaT * 0.5
    for centered in (False, True):
        op = Operator(geo, p.h, alpha, p.center, p.bounds, centered=centered)
        t0 = time.time()
        lam = op.spectral_radius(iters=60)
        print(f"{'centered' if centered else 'raw (C++)'}: spectral radius of Jacobi iteration "
              f"alpha*L ~ {lam:.4f} -> {'DIVERGES' if lam > 1 else 'converges'}  ({time.time() - t0:.0f}s)", flush=True)
        del op


# ---------------------------------------------------------------------------
def _read_rows(path, start_byte, n_lines, skip_partial=True):
    with open(path, "rb") as fh:
        fh.seek(start_byte)
        if skip_partial and start_byte > 0:
            fh.readline()
        lines = [fh.readline() for _ in range(n_lines)]
    return [l.split() for l in lines if l.strip()]


def cmd_files(pfile, datadir="/data01/trondini/cubic-sphere", stem="sph01", sample=200_000):
    import pandas as pd
    import project as proj

    p = Params.load(pfile)
    geo = geometry.build(p, CACHE)
    g = point_gidx(geo)
    ok = True

    # 1. surface file: full streaming compare of coordinates
    sfile = f"{datadir}/{stem}_s.dat"
    n_file = int(open(sfile).readline().split()[0])
    print(f"[s.dat] points: file {n_file}, regenerated {geo.n_points}", flush=True)
    ok &= n_file == geo.n_points
    t0 = time.time()
    worst = 0.0
    start = 0
    for chunk in pd.read_csv(sfile, sep=" ", skiprows=2, header=None, dtype=np.float64,
                             chunksize=20_000_000, engine="c"):
        a = chunk.to_numpy()
        mine = geo.coords(g[start:start + len(a)], p.h).astype(np.float32).astype(np.float64)
        worst = max(worst, float(np.max(np.abs(a - mine))))
        start += len(a)
    good = start == geo.n_points and worst < 1e-5
    ok &= good
    print(f"[s.dat] compared {start} rows, max |coord diff| {worst:.2e} (6-digit text)  "
          f"{'OK' if good else 'FAIL'}  ({time.time() - t0:.0f}s)", flush=True)

    # 2. neighbour file: sampled rows (first column is the point's own index)
    nfile = f"{datadir}/{stem}_n.dat"
    size = os.path.getsize(nfile)
    n_rows = n_bad = 0
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 0.97):
        rows = _read_rows(nfile, int(size * frac), sample // 6, skip_partial=frac > 0)
        rows = np.array([r for r in rows if len(r) == 33], dtype=np.int64)
        if frac == 0.0:
            rows = rows[1:] if len(rows) and rows[0, 0] == geo.n_points else rows
        ref = ref_neighbours(g, rows[:, 0], geo.shape)
        n_bad += int(np.sum(np.any(ref != rows, axis=1)))
        n_rows += len(rows)
    good = n_bad == 0
    ok &= good
    print(f"[n.dat] {n_rows} sampled rows, {n_bad} mismatching  {'OK' if good else 'FAIL'}", flush=True)

    # 3. projection file: head rows (row i = point i)
    pfile_ = f"{datadir}/{stem}_p.dat"
    rows = _read_rows(pfile_, 0, 100_003, skip_partial=False)[1:]
    arr = np.array([r for r in rows if len(r) == 32], dtype=np.float64)
    idx_f, w_f = arr[:, 0::2].astype(np.int64), arr[:, 1::2]
    vert, w, bad = proj.stencil(geo, p, g[: len(arr)])
    keys = lex_keys(g, geo.shape)
    vidx = np.searchsorted(keys, lex_keys(vert, geo.shape))
    idx_ok = np.mean(np.all(vidx == idx_f, axis=1))
    w_err = np.max(np.abs(w - w_f))
    good = idx_ok == 1.0 and w_err < 1e-5
    ok &= good
    print(f"[p.dat] {len(arr)} rows: vertex indices match {idx_ok * 100:.4f}%, "
          f"max weight diff {w_err:.1e} (6-digit text)  {'OK' if good else 'FAIL'}", flush=True)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    cmd, pf = sys.argv[1], sys.argv[2]
    {"small": cmd_small, "files": cmd_files, "spectral": cmd_spectral, "physics": cmd_physics}[cmd](pf)
