"""JAX implementation of ``Grid4::evolveCN`` and ``Grid4::project`` on the brick layout.

Fields have shape ``[NB, 8, 8, 8, 8]``. Brick ``ZERO`` (index ``geo.nb``) and the
padding bricks after it are always zero: a missing neighbour points there.
Work is done in chunks of ``K`` bricks (``lax.map``) so that the temporary
padded bricks fit in GPU memory next to the 3 full fields a Jacobi solve needs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import B, B4, OFFSETS, Geometry
from .params import Params

P = B + 2  # padded brick edge (one halo point per side)


@dataclass
class SolverOptions:
    solver: str = "jacobi"  # "jacobi" | "fixed-point" | "bicgstab"
    rtol: float = 1e-5
    atol: float = 0.0
    max_iter: int = 1000
    raw_coords: bool = False  # True = C++ coefficients (uncentred coordinates)
    center_sink: str = "auto"  # "auto" | "on" | "off"
    chunk: int = 1024  # bricks per lax.map chunk
    dtype: str = "float32"


def _pow2_at_least(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


class Simulation:
    def __init__(self, geo: Geometry, p: Params, opts: SolverOptions):
        self.geo, self.p, self.opts = geo, p, opts
        self.dtype = jnp.dtype(opts.dtype)
        self.alpha = p.alpha
        h = p.h

        K = min(_pow2_at_least(opts.chunk), _pow2_at_least(geo.nb + 1))
        self.K = K
        self.NB = -(-(geo.nb + 1) // K) * K
        self.ZERO = geo.nb
        self.Kp = min(K, 256)  # chunk for the projection scatter (16x temporaries)

        NB, ZERO = self.NB, self.ZERO
        origins = np.zeros((NB, 4), dtype=np.int32)
        origins[: geo.nb] = geo.origins
        nbr = np.full((NB, len(OFFSETS)), ZERO, dtype=np.int32)
        nbr[: geo.nb] = np.where(geo.neighbours < 0, ZERO, geo.neighbours)
        mask = np.zeros((NB, B, B, B, B), dtype=bool)
        mask[: geo.nb] = geo.masks
        lookup = np.where(geo.lookup < 0, ZERO, geo.lookup).astype(np.int32)

        # Coordinate tables, indexed by global grid index (extended past the grid
        # end so that every brick point has one; those points are masked anyway).
        ext = [int(geo.lo[a] + B * geo.nbricks_axis[a]) for a in range(4)]
        raw = [p.mins[a] + h * np.arange(ext[a], dtype=np.float64) for a in range(4)]
        centred = [raw[a] - p.center[a] for a in range(4)]
        # The C++ stores coordinates as float (Quaternion<float>) before using them.
        coef = [np.float32(r).astype(np.float64) for r in raw] if opts.raw_coords else centred

        # Device arrays used by the kernels. They are passed to every jitted
        # function as an argument: arrays captured by closure would be embedded
        # in the compiled program as constants (the mask alone is ~650 MB at
        # h = 0.01), blowing up compile time and duplicating them on the GPU.
        self._consts = {
            "origins": jnp.asarray(origins),
            "nbr": jnp.asarray(nbr),
            "mask": jnp.asarray(mask),
            "lookup": jnp.asarray(lookup),
            "coef": [jnp.asarray(c, dtype=self.dtype) for c in coef],
            "cent": [jnp.asarray(c, dtype=self.dtype) for c in centred],
            "bids": jnp.arange(NB, dtype=jnp.int32).reshape(NB // K, K),
        }
        self._c = self._consts  # what the kernels read; tracers while jit traces
        self.d_mask = self._consts["mask"]

        # Centre sink (C++: rho[(nPoints - 1) / 2] = 0), see main.py.
        self.sink = None

        self._step = self._jit(self._cn_step, donate_argnums=(0,))
        self._apply_aL = self._jit(self._apply_aL_impl)
        self._project = self._jit(self._project_impl)
        self._sweep = self._jit(self._sweep_impl, static_argnums=(2,))

    def _jit(self, fn, donate_argnums=(), static_argnums=()):
        """jax.jit(fn) with the device constants as an extra leading argument."""

        def traced(consts, *args):
            saved, self._c = self._c, consts
            try:
                return fn(*args)
            finally:
                self._c = saved

        compiled = jax.jit(
            traced,
            donate_argnums=tuple(i + 1 for i in donate_argnums),
            static_argnums=tuple(i + 1 for i in static_argnums),
        )
        def call(*args):
            return compiled(self._consts, *args)

        call.lower = lambda *args: compiled.lower(self._consts, *args)
        return call

    # ================================================================ fields
    def zeros(self):
        return jnp.zeros((self.NB, B, B, B, B), dtype=self.dtype)

    def to_device(self, host_bricks: np.ndarray):
        """Host array [nb or NB, 8,8,8,8] -> device field [NB, ...]."""
        full = np.zeros((self.NB, B, B, B, B), dtype=self.dtype)
        full[: len(host_bricks)] = host_bricks
        full[self.ZERO :] = 0
        return jnp.asarray(full) * self.d_mask

    # ============================================================== stencil
    def _padded(self, field, bids):
        """Bricks ``bids`` with a one-point halo taken from the 32 neighbour bricks."""
        K = bids.shape[0]
        pad = jnp.zeros((K, P, P, P, P), dtype=field.dtype)
        pad = pad.at[:, 1:-1, 1:-1, 1:-1, 1:-1].set(field[bids])
        nb = self._c["nbr"][bids]
        for k, off in enumerate(OFFSETS):
            src = [nb[:, k]]
            dst = [slice(None)]
            for s in off:
                if s == 0:
                    src.append(slice(None))
                    dst.append(slice(1, -1))
                elif s > 0:  # halo after the brick = first layer of the next brick
                    src.append(0)
                    dst.append(P - 1)
                else:
                    src.append(B - 1)
                    dst.append(0)
            pad = pad.at[tuple(dst)].set(field[tuple(src)])
        return pad

    def _coords(self, tables, bids):
        """Per-axis coordinates of the points of bricks ``bids``, broadcastable to [K,8,8,8,8]."""
        org = self._c["origins"][bids]
        out = []
        for a in range(4):
            x = tables[a][org[:, a][:, None] + jnp.arange(B)[None, :]]  # [K, 8]
            shape = [x.shape[0], 1, 1, 1, 1]
            shape[a + 1] = B
            out.append(x.reshape(shape))
        return out

    def _alpha_L(self, field, bids):
        """alpha * L(field) on bricks ``bids`` (masked) and alpha * diag(L).

        L = 12 f + 3 sum_i x_i d_i f + sum_i D_i d_ii f + sum_{i != j} d_i d_j (D_ij f)
        with D_i = |x|^2 - x_i^2 and D_ij = -x_i x_j, discretised exactly like
        ``Grid4::evolveCN`` (der1, der2, derij with the +-h shifted D_ij).
        """
        h = self.p.h
        pad = self._padded(field, bids)

        def S(s):
            return pad[(slice(None),) + tuple(slice(1 + si, 1 + si + B) for si in s)]

        X = self._coords(self._c["coef"], bids)
        r2 = X[0] ** 2 + X[1] ** 2 + X[2] ** 2 + X[3] ** 2
        f0 = S((0, 0, 0, 0))
        L = 12.0 * f0
        for a in range(4):
            e = [0, 0, 0, 0]
            e[a] = 1
            fp = S(e)
            e[a] = -1
            fm = S(e)
            L = L + (3.0 / (2.0 * h)) * X[a] * (fp - fm)  # -(1-4) p_i der1
            L = L + (r2 - X[a] ** 2) * (fp + fm - 2.0 * f0) * (1.0 / (h * h))  # der2
        for a, b in itertools.combinations(range(4), 2):

            def sh(sa, sb):
                s = [0, 0, 0, 0]
                s[a], s[b] = sa, sb
                return S(s)

            ap, am, bp, bm = X[a] + h, X[a] - h, X[b] + h, X[b] - h
            mixed = -(ap * bp) * sh(1, 1) + (ap * bm) * sh(1, -1) + (am * bp) * sh(-1, 1) - (am * bm) * sh(-1, -1)
            L = L + mixed * (2.0 / (4.0 * h * h))  # derij, (i,j) and (j,i)
        mask = self._c["mask"][bids]
        aL = jnp.where(mask, self.alpha * L, 0.0)
        a_diag = self.alpha * (12.0 - 6.0 * r2 / (h * h))
        return aL, a_diag, mask

    def _apply_aL_impl(self, field):
        out = jax.lax.map(lambda bids: self._alpha_L(field, bids)[0], self._c["bids"])
        return out.reshape(field.shape)

    def _sweep_impl(self, x, rhs, precond: bool):
        """One Jacobi (precond) or fixed-point sweep for (I - alpha L) x = rhs.

        fixed point (C++):  x' = rhs + alpha L x
        Jacobi:             x' = x + (rhs + alpha L x - x) / (1 - alpha L_diag)
        Returns x', max|x' - x| and max|x'|.
        """
        K = self.K

        def body(args):
            bids, rhs_c = args
            aL, a_diag, mask = self._alpha_L(x, bids)
            x_c = x[bids]
            r = rhs_c + aL - x_c
            dx = r / (1.0 - a_diag) if precond else r
            x_new = jnp.where(mask, x_c + dx, 0.0)
            return x_new, jnp.max(jnp.abs(x_new - x_c)), jnp.max(jnp.abs(x_new))

        rhs_chunks = rhs.reshape((-1, K) + rhs.shape[1:])
        x_new, dmax, xmax = jax.lax.map(body, (self._c["bids"], rhs_chunks))
        return x_new.reshape(x.shape), jnp.max(dmax), jnp.max(xmax)

    # ========================================================== CN step
    def _set_sink(self, field):
        if self.sink is None:
            return field
        return field.reshape(-1).at[self.sink].set(0.0).reshape(field.shape)

    def _cn_step(self, rho):
        rhs = rho + self._apply_aL_impl(rho)
        rhs = self._set_sink(rhs)
        o = self.opts
        if o.solver == "bicgstab":
            x, iters, diff, converged = self._bicgstab(rhs, rho)
        else:
            precond = o.solver == "jacobi"

            def cond(state):
                _, it, diff, xmax = state
                tol = o.atol + o.rtol * xmax
                return (it < o.max_iter) & (diff > tol) & jnp.isfinite(diff)

            def body(state):
                x, it, _, _ = state
                x_new, diff, xmax = self._sweep_impl(x, rhs, precond)
                return x_new, it + 1, diff, xmax

            big = jnp.array(jnp.finfo(self.dtype).max, dtype=self.dtype)  # forces the first sweep
            x, iters, diff, xmax = jax.lax.while_loop(
                cond, body, (rho, jnp.array(0, jnp.int32), big, jnp.array(0.0, self.dtype))
            )
            converged = jnp.isfinite(diff) & (diff <= o.atol + o.rtol * xmax)
            diff = diff / jnp.maximum(xmax, jnp.finfo(self.dtype).tiny)  # relative, for the report
        if self.sink is not None and not self.p.crown:
            x = self._set_sink(x)  # C++: `if (deltaC == 0) rhoA[_centerPoint] = 0`
        return x, iters, diff, converged

    def _bicgstab(self, rhs, x0):
        def A(v):
            return v - self._apply_aL_impl(v)

        def M(v):  # Jacobi preconditioner
            def body(bids):
                _, a_diag, mask = self._alpha_L(v, bids)
                return jnp.where(mask, v[bids] / (1.0 - a_diag), 0.0)

            return jax.lax.map(body, self._c["bids"]).reshape(v.shape)

        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, rhs, x0=x0, tol=self.opts.rtol, atol=self.opts.atol, maxiter=self.opts.max_iter, M=M
        )
        bnorm = jnp.maximum(jnp.sqrt(jnp.sum(rhs**2)), 1e-30)
        res = jnp.sqrt(jnp.sum((A(x) - rhs) ** 2)) / bnorm
        converged = jnp.isfinite(res) & (res <= 2.0 * self.opts.rtol + self.opts.atol / bnorm)
        return x, jnp.array(-1, jnp.int32), res, converged

    def step(self, rho):
        """One Crank-Nicolson step.

        Returns (rho, iterations, relative update or residual, converged).

        ``rho`` is donated: do not use it after the call.
        """
        return self._step(rho)

    # ============================================================ project
    def _project_impl(self, field):
        """``Grid4::project``: spread each point's density on the 16 grid points
        around its closest point on the sphere, with multilinear weights."""
        geo, p, Kp = self.geo, self.p, self.Kp
        lo = [int(v) for v in geo.lo]
        nbox = [int(v) for v in geo.nbox]
        nmax = [int(v) - 2 for v in geo.n]
        flat_out = jnp.zeros(field.size, dtype=field.dtype)

        def body(c, out):
            bids = c * Kp + jnp.arange(Kp, dtype=jnp.int32)
            rho = field[bids]
            X = self._coords(self._c["cent"], bids)
            X = [jnp.broadcast_to(x, rho.shape) for x in X]
            # Grid4::closest: c = x - grad_h(phi) * phi / |grad_h(phi)|, with the
            # central-difference gradient (step h) of phi = |x| - R.
            r2 = X[0] ** 2 + X[1] ** 2 + X[2] ** 2 + X[3] ** 2
            phi = jnp.sqrt(r2) - p.radius
            grad = [
                (jnp.sqrt(r2 + 2 * p.h * X[a] + p.h**2) - jnp.sqrt(r2 - 2 * p.h * X[a] + p.h**2)) / (2 * p.h)
                for a in range(4)
            ]
            gnorm = jnp.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2 + grad[3] ** 2)
            step = jnp.where(gnorm > 0, phi / jnp.where(gnorm > 0, gnorm, 1.0), 0.0)
            base, frac = [], []
            for a in range(4):
                g = (X[a] - grad[a] * step + (p.center[a] - p.mins[a])) / p.h
                i = jnp.clip(jnp.floor(g).astype(jnp.int32), 0, nmax[a])
                base.append(i)
                frac.append(g - i)
            idx, val = [], []
            for bits in itertools.product((0, 1), repeat=4):
                weight = rho
                bi, valid = [], True
                for a in range(4):
                    weight = weight * (frac[a] if bits[a] else 1.0 - frac[a])
                    v = base[a] + bits[a] - lo[a]
                    valid = valid & (v >= 0) & (v < nbox[a])
                    bi.append(jnp.clip(v, 0, nbox[a] - 1))
                brick = self._c["lookup"][bi[0] // B, bi[1] // B, bi[2] // B, bi[3] // B]
                brick = jnp.where(valid, brick, self.ZERO)
                local = (((bi[0] % B) * B + bi[1] % B) * B + bi[2] % B) * B + bi[3] % B
                idx.append((brick * B4 + local).ravel())
                val.append(weight.ravel())
            return out.at[jnp.concatenate(idx)].add(jnp.concatenate(val))

        flat_out = jax.lax.fori_loop(0, self.NB // Kp, body, flat_out)
        out = flat_out.reshape(field.shape).at[self.ZERO].set(0.0)
        return jnp.where(self._c["mask"], out, 0.0)

    def project(self, field):
        return self._project(field)

    # ========================================================= diagnostics
    def estimate_spectral_radius(self, iters: int = 60, seed: int = 0):
        """Power iteration on the iteration matrices of fixed-point and Jacobi.

        A value >= 1 means that solver diverges at this time step.
        """
        key = jax.random.PRNGKey(seed)
        v0 = jnp.where(self.d_mask, jax.random.normal(key, self.d_mask.shape, dtype=self.dtype), 0.0)
        zero = self.zeros()
        out = {}
        for name, precond in (("fixed-point", False), ("jacobi", True)):
            v = v0 / jnp.linalg.norm(v0)
            ratios = []
            for _ in range(iters):
                w, _, _ = self._sweep(v, zero, precond)  # rhs = 0: w = G v
                nrm = jnp.linalg.norm(w)
                ratios.append(float(nrm))
                v = w / nrm
            tail = np.array(ratios[-20:])
            out[name] = float(np.exp(np.mean(np.log(tail))))
        return out

    def stats(self, field):
        return {
            "sum": float(jnp.sum(field, dtype=jnp.float64 if jax.config.jax_enable_x64 else None)),
            "max": float(jnp.max(field)),
            "min": float(jnp.min(field)),
            "nan": int(jnp.sum(~jnp.isfinite(field))),
        }
