"""Discrete operator and Crank-Nicolson step on the brick layout (JAX/GPU).

``alpha*L(rho)`` is a term-by-term translation of the ``Lrho`` lambda in
``Grid4::evolveCN`` (grid4.cpp), using ``der1``, ``der2``, ``derij`` and
``Quaternion::D`` (utils.hpp)::

    L = 12 rho
      + sum_i 3 p_i (rho(+e_i) - rho(-e_i)) / (2h)                      [der1]
      + sum_i D_i(p) (rho(+e_i) + rho(-e_i) - 2 rho) / h^2              [der2]
      + sum_{i != j} [ f(++) D_ij(+h,+h) - f(+-) D_ij(+h,-h)
                      - f(-+) D_ij(-h,+h) + f(--) D_ij(-h,-h) ] / (4h^2) [derij]

    D_i(p) = sum_{j != i} p_j^2,   D_ij(a, b) = -(p_i + a)(p_j + b)

The C++ loops over ordered pairs (i, j), i != j, and derij is symmetric, so
each unordered pair contributes twice. Missing neighbours (index -1 in the C++)
read as 0; here they come from an all-zero brick or masked cells.

``p`` is the raw grid coordinate (as in the C++) with ``centered=False``, or
the coordinate relative to the sphere center with ``centered=True``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from geometry import B, BB, NOFF_INDEX


class Operator:
    def __init__(self, geo, h, alpha, center, bounds, centered=True,
                 chunk=2048, dtype=jnp.float32, tol=1e-3, max_iter=1000,
                 sink_slot=None):
        self.geo = geo
        self.C = chunk
        self.dtype = dtype
        self.n_chunks = math.ceil((geo.n_bricks + 1) / chunk)
        self.NBt = self.n_chunks * chunk  # >= n_bricks + 1 (zero brick at n_bricks)
        self.field_shape = (self.NBt, B, B, B, B)
        self.tol = tol
        self.max_iter = max_iter
        self.sink_slot = sink_slot

        origin = np.zeros((self.NBt, 4), dtype=np.int32)
        origin[: geo.n_bricks] = geo.origin
        nbr = np.full((self.NBt, 32), geo.n_bricks, dtype=np.int32)
        nbr[: geo.n_bricks] = geo.nbr
        self.origin = jnp.asarray(origin)
        self.nbr = jnp.asarray(nbr)

        mask = np.zeros(self.NBt * BB, dtype=bool)
        mask[geo.slot] = True
        self.mask = jnp.asarray(mask.reshape(self.field_shape))
        del mask

        # p_k = (i_k - off_k) * h  with off_k = (c_k - min_k)/h  (centered)
        #                          or   off_k = -min_k/h         (raw)
        mins = np.array([b[0] for b in bounds], dtype=np.float64)
        c = np.array(center, dtype=np.float64) if centered else np.zeros(4)
        self.off = tuple(float(v) for v in (c - mins) / h)
        f = np.dtype(dtype).type
        self.h = f(h)
        self.k12 = f(12.0 * alpha)
        self.k1 = f(3.0 * alpha / (2.0 * h))
        self.k2 = f(alpha / (h * h))
        self.kx = f(2.0 * alpha / (4.0 * h * h))  # x2: ordered pairs (i,j),(j,i)

        self._alphaL_jit = jax.jit(self._alphaL)
        self._cn_step_jit = jax.jit(self._cn_step, donate_argnums=0)

    # Large device constants are passed as jit arguments (not closed over),
    # so XLA does not embed copies of them in the executables.
    def _consts(self):
        return (self.mask, self.nbr, self.origin)

    def alphaL(self, A):
        return self._alphaL_jit(A, self._consts())

    def cn_step(self, rho):
        return self._cn_step_jit(rho, self._consts())

    # ------------------------------------------------------------------ halo
    def _padded(self, A, a_c, nbr_c):
        """(C, B+2, B+2, B+2, B+2) brick data with face/edge halos."""
        C = a_c.shape[0]
        inner = slice(1, B + 1)
        P = jnp.zeros((C, B + 2, B + 2, B + 2, B + 2), A.dtype)
        P = P.at[:, inner, inner, inner, inner].set(a_c)

        def layer(s):  # layer of the neighbour brick that touches us
            return B - 1 if s < 0 else 0

        def pos(s):  # halo position in the padded brick
            return 0 if s < 0 else B + 1

        for a in range(4):
            for s in (-1, 1):
                off = [0, 0, 0, 0]
                off[a] = s
                nb = nbr_c[:, NOFF_INDEX[tuple(off)]]
                src = [nb, slice(None), slice(None), slice(None), slice(None)]
                src[1 + a] = layer(s)
                dst = [slice(None), inner, inner, inner, inner]
                dst[1 + a] = pos(s)
                P = P.at[tuple(dst)].set(A[tuple(src)])
        for a in range(4):
            for b in range(a + 1, 4):
                for sa in (-1, 1):
                    for sb in (-1, 1):
                        off = [0, 0, 0, 0]
                        off[a], off[b] = sa, sb
                        nb = nbr_c[:, NOFF_INDEX[tuple(off)]]
                        src = [nb, slice(None), slice(None), slice(None), slice(None)]
                        src[1 + a], src[1 + b] = layer(sa), layer(sb)
                        dst = [slice(None), inner, inner, inner, inner]
                        dst[1 + a], dst[1 + b] = pos(sa), pos(sb)
                        P = P.at[tuple(dst)].set(A[tuple(src)])
        return P

    # -------------------------------------------------------------- operator
    def _chunk_alphaL(self, A, j, k):
        """alpha*L(A) on bricks [j*C, (j+1)*C) (unmasked)."""
        C = self.C
        s = j * C
        a_c = lax.dynamic_slice_in_dim(A, s, C)
        _, nbr, origin = k
        nbr_c = lax.dynamic_slice_in_dim(nbr, s, C)
        org_c = lax.dynamic_slice_in_dim(origin, s, C)
        P = self._padded(A, a_c, nbr_c)

        def S(d):
            return P[:, 1 + d[0]: B + 1 + d[0], 1 + d[1]: B + 1 + d[1],
                     1 + d[2]: B + 1 + d[2], 1 + d[3]: B + 1 + d[3]]

        ar = jnp.arange(B, dtype=self.dtype)
        p = []
        for k in range(4):
            shp = [C, 1, 1, 1, 1]
            shp[1 + k] = B
            base = org_c[:, k].astype(self.dtype).reshape(C, 1, 1, 1, 1)
            loc = ar.reshape([1 if i != 1 + k else B for i in range(5)])
            p.append((base + loc - self.off[k]) * self.h)
        sq = [pk * pk for pk in p]

        f0 = a_c
        L = self.k12 * f0
        h = self.h
        for i in range(4):
            e = [0, 0, 0, 0]
            e[i] = 1
            fp = S(e)
            fm = S([-v for v in e])
            Di = sum(sq[j] for j in range(4) if j != i)
            L = L + self.k1 * p[i] * (fp - fm) + self.k2 * Di * (fp + fm - 2 * f0)
        for i in range(4):
            for j in range(i + 1, 4):
                def d(si, sj):
                    v = [0, 0, 0, 0]
                    v[i], v[j] = si, sj
                    return v
                pip, pim = p[i] + h, p[i] - h
                pjp, pjm = p[j] + h, p[j] - h
                t = (-S(d(1, 1)) * pip * pjp + S(d(1, -1)) * pip * pjm
                     + S(d(-1, 1)) * pim * pjp - S(d(-1, -1)) * pim * pjm)
                L = L + self.kx * t
        return a_c, L

    def _map(self, fn):
        out = lax.map(fn, jnp.arange(self.n_chunks, dtype=jnp.int32))
        if isinstance(out, tuple):
            return tuple(o.reshape((self.NBt,) + o.shape[2:]) if o.ndim > 1 else o for o in out)
        return out.reshape(self.field_shape)

    def _mask_c(self, j, k):
        return lax.dynamic_slice_in_dim(k[0], j * self.C, self.C)

    def _alphaL(self, A, k):
        def fn(j):
            _, L = self._chunk_alphaL(A, j, k)
            return jnp.where(self._mask_c(j, k), L, 0)
        return self._map(fn)

    # ---------------------------------------------------------------- CN step
    def _cn_step(self, rho, k):
        """One Grid4::evolveCN step. Returns (rho_new, iterations, last maxdiff)."""
        def rhs_fn(j):
            a_c, L = self._chunk_alphaL(rho, j, k)
            return jnp.where(self._mask_c(j, k), a_c + L, 0)
        rhs = self._map(rhs_fn)
        if self.sink_slot is not None:
            rhs = rhs.reshape(-1).at[self.sink_slot].set(0).reshape(self.field_shape)

        def jac_fn(A, j):
            a_c, L = self._chunk_alphaL(A, j, k)
            r_c = lax.dynamic_slice_in_dim(rhs, j * self.C, self.C)
            new = jnp.where(self._mask_c(j, k), r_c + L, 0)
            return new, jnp.max(jnp.abs(new - a_c))

        def cond(state):
            _, it, d = state
            return (it < self.max_iter) & ~(d < self.tol)

        def body(state):
            A, it, _ = state
            new, dm = lax.map(lambda j: jac_fn(A, j), jnp.arange(self.n_chunks, dtype=jnp.int32))
            return new.reshape(self.field_shape), it + 1, jnp.max(dm)

        init = (rho, jnp.int32(0), jnp.asarray(jnp.inf, self.dtype))
        A, it, d = lax.while_loop(cond, body, init)
        if self.sink_slot is not None and not self.geo.meta["crown"]:
            A = A.reshape(-1).at[self.sink_slot].set(0).reshape(self.field_shape)
        return A, it, d

    # ------------------------------------------------------------- utilities
    def zeros(self):
        return jnp.zeros(self.field_shape, self.dtype)

    def spectral_radius(self, iters=60, seed=0):
        """Power-iteration estimate of rho(alpha*L) (Jacobi converges iff < 1)."""
        v = jax.random.normal(jax.random.PRNGKey(seed), self.field_shape, self.dtype)
        v = jnp.where(self.mask, v, 0)
        v = v / jnp.linalg.norm(v.ravel())
        lam = 0.0
        for _ in range(iters):
            w = self.alphaL(v)
            lam = float(jnp.linalg.norm(w.ravel()))
            v = w / lam
        return lam
