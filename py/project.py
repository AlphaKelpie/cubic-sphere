"""Closest-point projection stencil and ``Grid4::project()``.

Recomputes the C++ ``_projection`` table (16 trilinear vertices + weights per
point) instead of reading the 127 GB ``*_p.dat`` text file:

* crown model: ``Grid4::createMaps`` (grid4.cpp:733-830)
* hypercube model: ``Grid4::createProjection`` (grid4.cpp:569-668)

Arithmetic is float64 NumPy with the same operation order as the C++; weights
are rounded like ``(Tf)(1 - wd) * (1 - xd) * ...`` (only the first factor is
cast, the product is stored as float).
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from geometry import B, BB

CHUNK = 5_000_000


def _phi(q, center, radius):
    d2 = (q[:, 0] - center[0]) * (q[:, 0] - center[0]) + (q[:, 1] - center[1]) * (q[:, 1] - center[1])
    d2 = d2 + (q[:, 2] - center[2]) * (q[:, 2] - center[2])
    d2 = d2 + (q[:, 3] - center[3]) * (q[:, 3] - center[3])
    return np.sqrt(d2) - radius


def closest(p, h, center, radius):
    """Grid4::closest for an array of points p (float64[n, 4])."""
    grad = np.empty_like(p)
    for k in range(4):
        e = np.zeros(4)
        e[k] = h
        grad[:, k] = (_phi(p + e, center, radius) - _phi(p - e, center, radius)) / (2 * h)
    mod = np.sqrt(grad[:, 0] * grad[:, 0] + grad[:, 1] * grad[:, 1]
                  + grad[:, 2] * grad[:, 2] + grad[:, 3] * grad[:, 3])
    dist = _phi(p, center, radius)
    zero = np.all(grad == 0, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = p - grad * (dist / mod)[:, None]
    r[zero] = p[zero]
    return r


def stencil(geo, params, gidx):
    """(vertex grid indices int64[n,16,4], weights float32[n,16], n_bad)."""
    h = params.h
    mins = np.array([b[0] for b in params.bounds])
    # _volume holds float coordinates (static_cast<Tf>(m) / parsed text)
    p = (mins + h * gidx.astype(np.float64)).astype(np.float32).astype(np.float64)
    c = closest(p, h, params.center, params.radius)
    if params.crown:
        fl = np.floor(c / h)
        d = (c - fl * h) / h
        base = np.rint(fl - mins / h).astype(np.int64)
    else:
        f = c / h
        hi = np.array(geo.shape) - 2
        base = np.clip(np.floor(f).astype(np.int64), 0, hi)
        d = f - base
    bad = np.any((d > 1.0) | (d < 0.0), axis=1)
    one_minus = 1.0 - d
    w = np.empty((len(p), 16), dtype=np.float32)
    vert = np.empty((len(p), 16, 4), dtype=np.int64)
    for j in range(16):
        bits = ((j >> 3) & 1, (j >> 2) & 1, (j >> 1) & 1, j & 1)  # (w, x, y, z)
        f = [d[:, k] if b else one_minus[:, k] for k, b in enumerate(bits)]
        first = f[0].astype(np.float32).astype(np.float64)
        w[:, j] = (first * f[1] * f[2] * f[3]).astype(np.float32)
        vert[:, j, :] = base + np.array(bits)
    return vert, w, int(bad.sum())


def project_reference(geo, params, op, rho_points, log=print):
    """Grid4::project() in NumPy (slow reference; see project_gpu).

    rho_points: float32[N] in C++ point order. Returns the new brick field.
    """
    flat = jnp.zeros(op.NBt * op.mask[0].size, op.dtype)
    nz = np.nonzero(rho_points != 0)[0]  # C++ skips rho == 0
    n_bad = n_missing = 0
    for s in range(0, len(nz), CHUNK):
        idx = nz[s:s + CHUNK]
        gidx = geo.grid_index_of_slot(geo.slot[idx])
        vert, w, bad = stencil(geo, params, gidx)
        n_bad += bad
        vslot = geo.slot_of_grid_index(vert)
        miss = vslot < 0
        n_missing += int(miss.sum())
        vslot = np.where(miss, 0, vslot).astype(np.int32)
        contrib = (rho_points[idx][:, None] * w).astype(np.float32)
        contrib[miss] = 0
        flat = flat.at[jnp.asarray(vslot.ravel())].add(jnp.asarray(contrib.ravel()))
    out = flat.reshape(op.field_shape)
    lost = float(jnp.sum(jnp.where(op.mask, 0, out)))
    if n_bad or n_missing or lost:
        log(f"WARNING project(): {n_bad} points with weights outside [0,1], "
            f"{n_missing} stencil vertices outside the bricks, mass {lost:g} on "
            f"vertices outside the point set (C++ would throw)")
    return jnp.where(op.mask, out, 0)


def project_gpu(geo, params, op, rho, bricks_per_chunk=256, log=print):
    """Grid4::project() on the GPU, iterating over the brick layout.

    Same arithmetic as ``stencil``/``project_reference`` (float64 stencil,
    float32 scatter). ``rho`` is the brick field; returns the projected field.
    """
    Cb = bricks_per_chunk
    n_chunks = op.NBt // Cb
    assert n_chunks * Cb == op.NBt
    h = float(params.h)
    mins = np.array([b[0] for b in params.bounds], dtype=np.float64)
    center = np.array(params.center, dtype=np.float64)
    radius = float(params.radius)
    shape_m2 = np.array(geo.shape, dtype=np.int64) - 2
    lo = geo.lo.astype(np.int64)
    nbg = geo.nbg.astype(np.int64)
    zero_slot = geo.n_bricks * BB  # all-zero padding brick

    with jax.enable_x64(True):
        bok = jnp.asarray(geo.brick_of_key)
        origin = op.origin.astype(jnp.int64)
        loc = np.stack(np.unravel_index(np.arange(BB), (B, B, B, B)), axis=1)
        loc = jnp.asarray(loc, dtype=jnp.int64)
        bits = jnp.asarray([((j >> 3) & 1, (j >> 2) & 1, (j >> 1) & 1, j & 1) for j in range(16)],
                           dtype=jnp.int64)

        def phi(q):
            d = q - center
            d2 = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]
            d2 = d2 + d[..., 2] * d[..., 2]
            d2 = d2 + d[..., 3] * d[..., 3]
            return jnp.sqrt(d2) - radius

        def run(r0, rho_, mask_, origin_, bok_):
          def chunk(j, carry):
              out, n_bad, n_missing = carry
              org = lax.dynamic_slice_in_dim(origin_, j * Cb, Cb)                 # (Cb,4)
              g = (org[:, None, :] + loc[None]).reshape(-1, 4)                   # (n,4)
              vals = lax.dynamic_slice_in_dim(rho_, j * Cb, Cb).reshape(-1)
              valid = lax.dynamic_slice_in_dim(mask_, j * Cb, Cb).reshape(-1) & (vals != 0)
              p = (mins + h * g.astype(jnp.float64)).astype(jnp.float32).astype(jnp.float64)
              grad = jnp.stack([(phi(p + h * e) - phi(p - h * e)) / (2 * h)
                                for e in np.eye(4)], axis=1)
              mod = jnp.sqrt(grad[:, 0] * grad[:, 0] + grad[:, 1] * grad[:, 1]
                             + grad[:, 2] * grad[:, 2] + grad[:, 3] * grad[:, 3])
              zero = jnp.all(grad == 0, axis=1)
              c = jnp.where(zero[:, None], p, p - grad * (phi(p) / jnp.where(zero, 1.0, mod))[:, None])
              if params.crown:
                  fl = jnp.floor(c / h)
                  d = (c - fl * h) / h
                  base = jnp.rint(fl - mins / h).astype(jnp.int64)
              else:
                  f = c / h
                  base = jnp.clip(jnp.floor(f).astype(jnp.int64), 0, shape_m2)
                  d = f - base
              n_bad = n_bad + jnp.sum(valid & jnp.any((d > 1.0) | (d < 0.0), axis=1))
              om = 1.0 - d
              for jv in range(16):
                  b = bits[jv]
                  fac = jnp.where(b[None, :] == 1, d, om)                        # (n,4)
                  first = fac[:, 0].astype(jnp.float32).astype(jnp.float64)
                  w = (first * fac[:, 1] * fac[:, 2] * fac[:, 3]).astype(jnp.float32)
                  vert = base + b
                  rel = vert - lo
                  bc = rel // B
                  inside = jnp.all((rel >= 0) & (bc < nbg), axis=1)
                  bc = jnp.where(inside[:, None], bc, 0)
                  key = ((bc[:, 0] * nbg[1] + bc[:, 1]) * nbg[2] + bc[:, 2]) * nbg[3] + bc[:, 3]
                  brick = jnp.where(inside, bok_[key], -1)
                  l = rel % B
                  local = ((l[:, 0] * B + l[:, 1]) * B + l[:, 2]) * B + l[:, 3]
                  ok = valid & (brick >= 0)
                  n_missing = n_missing + jnp.sum(valid & (brick < 0))
                  slot = jnp.where(ok, brick * BB + local, zero_slot)
                  out = out.at[slot].add(jnp.where(ok, vals * w, jnp.float32(0)))
              return out, n_bad, n_missing
          return lax.fori_loop(0, n_chunks, chunk, (r0, jnp.int64(0), jnp.int64(0)))

        out0 = jnp.zeros(op.NBt * BB, op.dtype)
        out, n_bad, n_missing = jax.jit(run)(out0, rho, op.mask, origin, bok)
        out = out.reshape(op.field_shape)
        lost = float(jnp.sum(jnp.where(op.mask, 0, out)))
        out = jnp.where(op.mask, out, jnp.zeros((), op.dtype))
    if int(n_bad) or int(n_missing) or lost:
        log(f"WARNING project(): {int(n_bad)} points with weights outside [0,1], "
            f"{int(n_missing)} stencil vertices outside the bricks, mass {lost:g} on "
            f"vertices outside the point set (C++ would throw)")
    return out
