"""Conversion between the brick layout and the C++ point order, and rho file I/O.

The C++ stores rho as a flat array over the active points in lexicographic
(w, x, y, z) order and writes it as text (``_r.dat``: ``nPoints``, a blank
line, one ``%g`` value per line). Here the default output is a float32
``.npy`` with the same values in the same order; ``--format text`` writes the
C++ format.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .geometry import B, B4, Geometry


class CppOrder:
    """Permutation between the brick layout and the C++ point order, per w-slab of bricks."""

    def __init__(self, geo: Geometry):
        self.geo = geo
        self._perms: list[np.ndarray] | None = None
        self.offsets = np.concatenate([[0], np.cumsum(geo.slab_count)])

    @property
    def perms(self) -> list[np.ndarray]:
        if self._perms is None:
            self._perms = [self._build(bw) for bw in range(len(self.geo.slab_count))]
        return self._perms

    def _build(self, bw: int) -> np.ndarray:
        geo = self.geo
        s, e = int(geo.slab_start[bw]), int(geo.slab_start[bw + 1])
        nbw, nbx, nby, nbz = (int(v) for v in geo.nbricks_axis)
        nw = min(B, int(geo.nbox[0]) - bw * B)
        nx, ny, nz = (int(v) for v in geo.nbox[1:])
        dense = np.full((nbx, nby, nbz, B, B, B, B), -1, dtype=np.int64)
        local = np.arange((e - s) * B4, dtype=np.int64).reshape(-1, B, B, B, B)
        local[~geo.masks[s:e]] = -1
        bc = geo.brick_coords[s:e].astype(np.int64)
        dense[bc[:, 1], bc[:, 2], bc[:, 3]] = local
        dense = dense.transpose(3, 0, 4, 1, 5, 2, 6).reshape(B, nbx * B, nby * B, nbz * B)
        dense = dense[:nw, :nx, :ny, :nz].ravel()
        perm = dense[dense >= 0]
        if len(perm) != geo.slab_count[bw]:
            raise RuntimeError("Internal error: C++ order permutation mismatch")
        return perm.astype(np.int32 if len(local.ravel()) < 2**31 else np.int64)

    def to_cpp(self, field) -> np.ndarray:
        """Device field [NB, 8,8,8,8] -> host float32 array in C++ point order."""
        geo = self.geo
        out = np.empty(geo.npoints, dtype=np.float32)
        for bw, perm in enumerate(self.perms):
            s, e = int(geo.slab_start[bw]), int(geo.slab_start[bw + 1])
            if e == s:
                continue
            chunk = np.asarray(field[s:e]).reshape(-1)
            out[self.offsets[bw] : self.offsets[bw + 1]] = chunk[perm]
        return out

    def from_cpp(self, values: np.ndarray) -> np.ndarray:
        """Array in C++ point order -> host bricks [nb, 8,8,8,8] (float32)."""
        geo = self.geo
        if len(values) != geo.npoints:
            raise ValueError(
                f"rho has {len(values)} points but the grid from the params file has {geo.npoints}"
                " (different h, deltaC, radius or bounds?)"
            )
        out = np.zeros((geo.nb, B, B, B, B), dtype=np.float32)
        flat = out.reshape(-1)
        for bw, perm in enumerate(self.perms):
            s = int(geo.slab_start[bw])
            flat[s * B4 + perm] = values[self.offsets[bw] : self.offsets[bw + 1]]
        return out

    def cpp_index_to_flat(self, k: int) -> int:
        """Index of C++ point ``k`` in the flattened device field."""
        bw = int(np.searchsorted(self.offsets, k, side="right") - 1)
        return int(self.geo.slab_start[bw]) * B4 + int(self.perms[bw][k - self.offsets[bw]])


# ---------------------------------------------------------------------- files
def read_rho(prefix: str) -> np.ndarray:
    """Read ``<prefix>_r.npy`` or the C++ text ``<prefix>_r.dat``.

    Parsed text is cached as ``<prefix>_r.dat.npy``, used while newer than the text.
    """
    npy, txt = prefix + "_r.npy", prefix + "_r.dat"
    cache = txt + ".npy"
    if os.path.exists(npy):
        return np.load(npy, mmap_mode="r")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(txt):
        return np.load(cache, mmap_mode="r")
    with open(txt, "rb") as fin:
        n = int(fin.readline())
        values = np.fromfile(fin, dtype=np.float32, sep=" ")
    if len(values) != n:
        raise ValueError(f"{txt}: header says {n} points, found {len(values)}")
    try:
        np.save(cache, values)
    except OSError:
        pass  # read-only data directory: just parse again next time
    return values


def has_rho(prefix: str) -> bool:
    return os.path.exists(prefix + "_r.npy") or os.path.exists(prefix + "_r.dat")


def write_rho(prefix: str, values: np.ndarray, fmt: str):
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    if fmt == "npy":
        np.save(prefix + "_r.npy", values)
    else:
        with open(prefix + "_r.dat", "w") as fout:
            fout.write(f"{len(values)}\n\n")
            values.astype(np.float64).tofile(fout, sep="\n", format="%g")
            fout.write("\n")


class Writer:
    """Writes snapshots in a background thread so the GPU keeps computing."""

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def submit(self, prefix: str, values: np.ndarray):
        self.wait()
        self.pending = self.pool.submit(write_rho, prefix, values, self.fmt)

    def wait(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None
