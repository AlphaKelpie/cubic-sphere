"""Grid geometry: which points of the 4D grid are simulated, and how they are stored.

The C++ code keeps a flat list of points plus a 33-entry neighbour table and a
16-entry projection table per point (about 105 GB of RAM at h = 0.01). Here
the active points are stored in dense 8^4 "bricks" instead: only bricks that
contain at least one active point are kept, and a neighbour is found from the
brick structure (same brick, or one of 32 neighbouring bricks). Points of a
brick that are outside the crown are masked to zero, which is the C++
convention for a missing neighbour (index -1 reads 0).

Active points are the same set, in the same lexicographic (w, x, y, z) order,
as ``Grid4::createVol`` (crown model, ``deltaC != 0``) or
``Grid4::createVolNeigh`` (whole hypercube, ``deltaC == 0``).
"""

from __future__ import annotations

import hashlib
import itertools
import os
import time

import numpy as np

from .params import Params

B = 8  # brick edge
B4 = B**4

# The 32 neighbouring bricks the 33-point stencil can reach: 8 along the axes,
# 24 along the 2D diagonals (no 3D/4D diagonals are needed).
OFFSETS: list[tuple[int, int, int, int]] = []
for _a in range(4):
    for _s in (-1, 1):
        _o = [0, 0, 0, 0]
        _o[_a] = _s
        OFFSETS.append(tuple(_o))
for _a, _b in itertools.combinations(range(4), 2):
    for _sa, _sb in itertools.product((-1, 1), repeat=2):
        _o = [0, 0, 0, 0]
        _o[_a], _o[_b] = _sa, _sb
        OFFSETS.append(tuple(_o))
OFFSETS_ARR = np.array(OFFSETS, dtype=np.int64)


def _cpp_round(x: float) -> int:
    """``std::round`` (half away from zero) for the positive values used here."""
    return int(np.floor(x + 0.5))


class Geometry:
    """Host-side description of the active points and the brick layout."""

    def __init__(self, p: Params):
        self.h = p.h
        self.crown = p.crown
        self.radius = p.radius
        self.deltaC = p.deltaC
        self.center = np.array(p.center, dtype=np.float64)
        self.mins = np.array(p.mins, dtype=np.float64)
        self.n = np.array([_cpp_round((mx - mn) / p.h) + 1 for mn, mx in zip(p.mins, p.maxs)])
        # Grid coordinates exactly as the C++ computes them: min + h * i (double).
        self.coords = [self.mins[a] + self.h * np.arange(self.n[a], dtype=np.float64) for a in range(4)]

        # Box of grid indices that can contain active points.
        if self.crown:
            reach = self.radius + abs(self.deltaC) + self.h
            lo, hi = [], []
            for a in range(4):
                inside = np.nonzero(np.abs(self.coords[a] - self.center[a]) <= reach)[0]
                if inside.size == 0:
                    raise ValueError("The sphere does not intersect the grid")
                lo.append(int(inside[0]))
                hi.append(int(inside[-1]) + 1)
            self.lo, self.hi = np.array(lo), np.array(hi)
        else:
            self.lo, self.hi = np.zeros(4, dtype=np.int64), self.n.copy()
        self.nbox = self.hi - self.lo
        self.nbricks_axis = -(-self.nbox // B)  # ceil

    # ------------------------------------------------------------------ masks
    def mask_plane(self, iw: int) -> np.ndarray:
        """Active points of the box plane ``w = lo_w + iw`` as bool [Nx, Ny, Nz].

        Same expression and evaluation order as ``Grid4::phi`` and the test in
        ``Grid4::createVol``: ``|sqrt(dw^2 + dx^2 + dy^2 + dz^2) - R| <= deltaC``.
        """
        if not self.crown:
            return np.ones(tuple(self.nbox[1:]), dtype=bool)
        c = self.center
        lo, hi = self.lo, self.hi
        dw = self.coords[0][lo[0] + iw] - c[0]
        dx = self.coords[1][lo[1] : hi[1]] - c[1]
        dy = self.coords[2][lo[2] : hi[2]] - c[2]
        dz = self.coords[3][lo[3] : hi[3]] - c[3]
        s = dw * dw + dx * dx
        s = s[:, None] + (dy * dy)[None, :]
        s = s[:, :, None] + (dz * dz)[None, None, :]
        return np.abs(np.sqrt(s) - self.radius) <= self.deltaC

    # ------------------------------------------------------------------ build
    def build(self, verbose: bool = True) -> "Geometry":
        nbw, nbx, nby, nbz = (int(v) for v in self.nbricks_axis)
        nx, ny, nz = (int(v) for v in self.nbox[1:])
        coords_list, masks_list = [], []
        self.slab_start = np.zeros(nbw + 1, dtype=np.int64)
        self.slab_count = np.zeros(nbw, dtype=np.int64)  # active points per w-slab of bricks
        t0 = time.time()
        for bw in range(nbw):
            slab = np.zeros((B, nbx * B, nby * B, nbz * B), dtype=bool)
            for k in range(B):
                iw = bw * B + k
                if iw < self.nbox[0]:
                    slab[k, :nx, :ny, :nz] = self.mask_plane(iw)
            bricks = slab.reshape(B, nbx, B, nby, B, nbz, B).transpose(1, 3, 5, 0, 2, 4, 6)
            active = bricks.reshape(nbx, nby, nbz, -1).any(axis=-1)
            idx = np.argwhere(active)  # lexicographic (bx, by, bz)
            coords_list.append(np.column_stack([np.full(len(idx), bw), idx]))
            masks_list.append(bricks[active])
            self.slab_count[bw] = int(slab.sum())
            self.slab_start[bw + 1] = self.slab_start[bw] + len(idx)
            if verbose:
                print(f"\r  geometry: brick slab {bw + 1}/{nbw}", end="", flush=True)
        if verbose:
            print(f"  ({time.time() - t0:.1f} s)")

        self.brick_coords = np.concatenate(coords_list).astype(np.int32)  # [nb, 4]
        self.masks = np.concatenate(masks_list)  # [nb, 8, 8, 8, 8] bool
        self._finish()
        return self

    def _finish(self):
        self.nb = len(self.brick_coords)
        self.npoints = int(self.slab_count.sum())
        self.lookup = np.full(tuple(int(v) for v in self.nbricks_axis), -1, dtype=np.int32)
        bc = self.brick_coords.astype(np.int64)
        self.lookup[tuple(bc.T)] = np.arange(self.nb, dtype=np.int32)
        # neighbour bricks (-1 = none)
        nbr = np.full((self.nb, len(OFFSETS)), -1, dtype=np.int32)
        for k, off in enumerate(OFFSETS_ARR):
            q = bc + off
            ok = np.all((q >= 0) & (q < self.nbricks_axis), axis=1)
            nbr[ok, k] = self.lookup[tuple(q[ok].T)]
        self.neighbours = nbr
        # global grid index of each brick's first point
        self.origins = (self.lo + B * bc).astype(np.int32)

    # ------------------------------------------------------------------ cache
    def cache_key(self) -> str:
        items = (self.h, *self.mins, *self.n, *self.center, self.radius, self.deltaC, B)
        return hashlib.sha1(repr(tuple(float(v) for v in items)).encode()).hexdigest()[:16]

    def save(self, filename: str):
        np.savez(
            filename,
            brick_coords=self.brick_coords,
            masks=np.packbits(self.masks.reshape(self.nb, -1), axis=1),
            slab_start=self.slab_start,
            slab_count=self.slab_count,
        )

    def load(self, filename: str) -> "Geometry":
        d = np.load(filename)
        self.brick_coords = d["brick_coords"]
        self.masks = np.unpackbits(d["masks"], axis=1, count=B4).astype(bool).reshape(-1, B, B, B, B)
        self.slab_start = d["slab_start"]
        self.slab_count = d["slab_count"]
        self._finish()
        return self


def get_geometry(p: Params, cache_dir: str | None, verbose: bool = True) -> Geometry:
    geo = Geometry(p)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.join(cache_dir, f"geometry_{geo.cache_key()}.npz")
        if os.path.exists(fname):
            if verbose:
                print(f"  geometry: loading cache {fname}")
            return geo.load(fname)
        geo.build(verbose)
        geo.save(fname)
        return geo
    return geo.build(verbose)
