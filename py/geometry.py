"""Point set and block-sparse ("brick") layout of the 4D grid.

Point set and ordering replicate ``Grid4::createVol`` (spherical crown) and
``Grid4::createVolNeigh`` (whole hypercube) in grid4.cpp: lexicographic order
over (w, x, y, z), coordinates ``min + h*i`` in double precision, crown points
kept when ``|phi| <= deltaC``.

Instead of the C++ per-point neighbour/projection tables (~105 GB at h=0.01),
points live in dense B^4 bricks; only bricks containing points are stored.
Neighbours are derived from the grid structure plus a small brick-neighbour
table. A missing neighbour (C++ index -1) reads from an all-zero brick.
"""

import hashlib
import json
import os
import time

import numpy as np

B = 8  # brick edge length (cells)
BB = B**4

# Neighbour offsets in the exact order of the C++ Neighbours struct
# (point[k + 1] = n[k], see Grid4::createMaps).
NOFF = np.array(
    [
        (-1, 0, 0, 0), (1, 0, 0, 0), (0, -1, 0, 0), (0, 1, 0, 0),
        (0, 0, -1, 0), (0, 0, 1, 0), (0, 0, 0, -1), (0, 0, 0, 1),
        (-1, -1, 0, 0), (-1, 1, 0, 0), (-1, 0, -1, 0), (-1, 0, 1, 0),
        (-1, 0, 0, -1), (-1, 0, 0, 1), (1, -1, 0, 0), (1, 1, 0, 0),
        (1, 0, -1, 0), (1, 0, 1, 0), (1, 0, 0, -1), (1, 0, 0, 1),
        (0, -1, -1, 0), (0, -1, 1, 0), (0, -1, 0, -1), (0, -1, 0, 1),
        (0, 1, -1, 0), (0, 1, 1, 0), (0, 1, 0, -1), (0, 1, 0, 1),
        (0, 0, -1, -1), (0, 0, -1, 1), (0, 0, 1, -1), (0, 0, 1, 1),
    ],
    dtype=np.int64,
)
NOFF_INDEX = {tuple(int(v) for v in o): k for k, o in enumerate(NOFF)}


def grid_shape(params):
    """Number of grid points per axis, as in the Grid4 constructor."""
    return tuple(
        int(np.floor((hi - lo) / params.h + 0.5)) + 1 for lo, hi in params.bounds
    )


class Geometry:
    """Brick layout of the point set.

    Attributes
    ----------
    n_points : number of grid points (C++ ``_nPoints``)
    n_bricks : number of stored bricks (excluding the zero brick)
    slot     : int32[n_points], flat index ``brick*B^4 + local`` of each point,
               in C++ point order
    origin   : int32[n_bricks, 4], grid index of each brick's first cell
    nbr      : int32[n_bricks, 32], brick index of the neighbour brick in each
               NOFF direction, ``n_bricks`` (zero brick) when absent
    lo, nbg  : brick-grid origin (grid index) and brick-grid shape per axis
    brick_of_key : int32[prod(nbg)], dense lookup brick-grid key -> brick index
    """

    def __init__(self, meta, slot, origin, nbr, brick_of_key):
        self.meta = meta
        self.n_points = int(meta["n_points"])
        self.n_bricks = int(meta["n_bricks"])
        self.lo = np.array(meta["lo"], dtype=np.int64)
        self.nbg = np.array(meta["nbg"], dtype=np.int64)
        self.shape = tuple(meta["shape"])
        self.slot = slot
        self.origin = origin
        self.nbr = nbr
        self.brick_of_key = brick_of_key

    # ---- helpers -----------------------------------------------------------
    def grid_index_of_slot(self, slot):
        """Grid indices (int64[..., 4]) of flat brick slots."""
        slot = np.asarray(slot, dtype=np.int64)
        brick = slot // BB
        local = slot % BB
        loc = np.stack(
            [local // (B**3), (local // (B**2)) % B, (local // B) % B, local % B],
            axis=-1,
        )
        return self.origin[brick].astype(np.int64) + loc

    def slot_of_grid_index(self, gidx):
        """Flat slot of grid indices (int64[..., 4]); -1 where no brick exists."""
        gidx = np.asarray(gidx, dtype=np.int64)
        rel = gidx - self.lo
        bc = rel // B
        inside = np.all((rel >= 0) & (bc < self.nbg), axis=-1)
        bc = np.where(inside[..., None], bc, 0)
        key = ((bc[..., 0] * self.nbg[1] + bc[..., 1]) * self.nbg[2] + bc[..., 2]) * self.nbg[3] + bc[..., 3]
        brick = np.where(inside, self.brick_of_key[key], -1)
        loc = rel % B
        local = ((loc[..., 0] * B + loc[..., 1]) * B + loc[..., 2]) * B + loc[..., 3]
        return np.where(brick >= 0, brick * BB + local, -1)

    def coords(self, gidx, h):
        """Float64 coordinates of grid indices (min + h*i, as in the C++)."""
        mins = np.array([b[0] for b in self.meta["bounds"]])
        return mins + h * np.asarray(gidx, dtype=np.float64)


def _cache_key(params):
    blob = json.dumps(
        {
            "bounds": params.bounds,
            "h": params.h,
            "deltaC": params.deltaC,
            "radius": params.radius,
            "center": params.center,
            "B": B,
            "v": 1,
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _generate_points(params, log=print):
    """Grid indices (int16[N,4]) of all points, in C++ order."""
    h = params.h
    shape = grid_shape(params)
    axes = [lo + h * np.arange(n, dtype=np.float64) for (lo, _), n in zip(params.bounds, shape)]

    if params.crown:
        # Only indices that can possibly be within deltaC of the sphere.
        reach = params.radius + params.deltaC + 2 * h
        ranges = []
        for ax, c in zip(axes, params.center):
            idx = np.nonzero(np.abs(ax - c) <= reach)[0]
            ranges.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0))
    else:
        ranges = [(0, n) for n in shape]

    (w0, w1), (x0, x1), (y0, y1), (z0, z1) = ranges
    xs, ys, zs = axes[1][x0:x1], axes[2][y0:y1], axes[3][z0:z1]
    cw, cx, cy, cz = params.center
    dx2 = ((xs - cx) * (xs - cx))[:, None, None]
    dy2 = ((ys - cy) * (ys - cy))[None, :, None]
    dz2 = ((zs - cz) * (zs - cz))[None, None, :]

    chunks = []
    t0 = time.time()
    for mi in range(w0, w1):
        if params.crown:
            m = axes[0][mi]
            # Same association order as phi(): ((dw2 + dx2) + dy2) + dz2
            d2 = (m - cw) * (m - cw) + dx2
            d2 = d2 + dy2
            d2 = d2 + dz2
            keep = ~(np.abs(np.sqrt(d2) - params.radius) > params.deltaC)
            ii, jj, kk = np.nonzero(keep)
        else:
            ii, jj, kk = [a.ravel() for a in np.indices((x1 - x0, y1 - y0, z1 - z0))]
        if len(ii) == 0:
            continue
        g = np.empty((len(ii), 4), dtype=np.int16)
        g[:, 0] = mi
        g[:, 1] = ii + x0
        g[:, 2] = jj + y0
        g[:, 3] = kk + z0
        chunks.append(g)
        if (mi - w0) % 20 == 0:
            log(f"  points: w-slice {mi - w0 + 1}/{w1 - w0} ({time.time() - t0:.0f}s)")
    return np.concatenate(chunks), shape


def build(params, cache_dir, log=print):
    """Build (or load from cache) the brick geometry for these params."""
    key = _cache_key(params)
    cdir = os.path.join(cache_dir, f"geom_{key}")
    meta_f = os.path.join(cdir, "meta.json")
    if os.path.exists(meta_f):
        log(f"Loading cached geometry {cdir}")
        with open(meta_f) as f:
            meta = json.load(f)
        arrs = {n: np.load(os.path.join(cdir, f"{n}.npy"), mmap_mode=None)
                for n in ("slot", "origin", "nbr", "brick_of_key")}
        return Geometry(meta, **arrs)

    log("Generating point set (replicating createVol/createVolNeigh)...")
    t0 = time.time()
    gidx, shape = _generate_points(params, log)
    n_points = len(gidx)
    log(f"  {n_points} points in {time.time() - t0:.0f}s")
    if n_points > np.iinfo(np.int32).max:
        raise RuntimeError("too many points for int32 indices (C++ limit too)")

    lo = gidx.min(axis=0).astype(np.int64)
    hi = gidx.max(axis=0).astype(np.int64)
    nbg = (hi - lo) // B + 1
    n_keys = int(np.prod(nbg))

    occupied = np.zeros(n_keys, dtype=bool)
    step = 50_000_000
    for s in range(0, n_points, step):
        rel = gidx[s:s + step].astype(np.int64) - lo
        bc = rel // B
        k = ((bc[:, 0] * nbg[1] + bc[:, 1]) * nbg[2] + bc[:, 2]) * nbg[3] + bc[:, 3]
        occupied[k] = True
    n_bricks = int(occupied.sum())
    brick_of_key = np.full(n_keys, -1, dtype=np.int32)
    brick_of_key[occupied] = np.arange(n_bricks, dtype=np.int32)
    if (n_bricks + 1) * BB > np.iinfo(np.int32).max:
        raise RuntimeError("brick storage exceeds int32 slot range; reduce B")
    log(f"  {n_bricks} bricks of {B}^4 -> {n_bricks * BB / n_points:.2f}x storage overhead")

    keys = np.nonzero(occupied)[0].astype(np.int64)
    bc = np.stack(
        [keys // (nbg[1] * nbg[2] * nbg[3]), (keys // (nbg[2] * nbg[3])) % nbg[1],
         (keys // nbg[3]) % nbg[2], keys % nbg[3]], axis=1)
    origin = (lo + B * bc).astype(np.int32)

    nbr = np.full((n_bricks, len(NOFF)), n_bricks, dtype=np.int32)
    for k, off in enumerate(NOFF):
        nb = bc + off
        ok = np.all((nb >= 0) & (nb < nbg), axis=1)
        nk = ((nb[:, 0] * nbg[1] + nb[:, 1]) * nbg[2] + nb[:, 2]) * nbg[3] + nb[:, 3]
        found = np.where(ok, brick_of_key[np.where(ok, nk, 0)], -1)
        nbr[:, k] = np.where(found >= 0, found, n_bricks)

    slot = np.empty(n_points, dtype=np.int32)
    for s in range(0, n_points, step):
        rel = gidx[s:s + step].astype(np.int64) - lo
        bcp = rel // B
        k = ((bcp[:, 0] * nbg[1] + bcp[:, 1]) * nbg[2] + bcp[:, 2]) * nbg[3] + bcp[:, 3]
        loc = rel % B
        local = ((loc[:, 0] * B + loc[:, 1]) * B + loc[:, 2]) * B + loc[:, 3]
        slot[s:s + step] = brick_of_key[k].astype(np.int64) * BB + local
    del gidx

    meta = {
        "n_points": n_points,
        "n_bricks": n_bricks,
        "lo": lo.tolist(),
        "nbg": nbg.tolist(),
        "shape": list(shape),
        "bounds": [list(b) for b in params.bounds],
        "h": params.h,
        "crown": params.crown,
        "B": B,
    }
    os.makedirs(cdir, exist_ok=True)
    for n, a in (("slot", slot), ("origin", origin), ("nbr", nbr), ("brick_of_key", brick_of_key)):
        np.save(os.path.join(cdir, f"{n}.npy"), a)
    with open(meta_f, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"Geometry built in {time.time() - t0:.0f}s, cached in {cdir}")
    return Geometry(meta, slot, origin, nbr, brick_of_key)
