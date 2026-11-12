"""JAX/GPU port of cubic-sphere ``mainGrid4.cpp`` (corKDTf0.out).

    python main.py params01.dat [--out-prefix PREFIX] [--max-steps N] [...]

Default physics differs from the C++ in two deliberate ways (see README.md):
coefficients use coordinates relative to the sphere center, and there is no
absorbing point at index (N-1)/2. ``--raw-coords --keep-center-sink``
reproduces the C++ exactly (which diverges to NaN for params01.dat).
"""

import argparse
import os
import time

import numpy as np
import os as _os
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")  # T4: ~10 GB for 3 fields + mask; leave room for CUDA modules


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="?", default="params.dat")
    ap.add_argument("--out-prefix", help="output prefix (default /data01/gregb/cubic-sphere/<basename of params path>)")
    ap.add_argument("--cache-dir", default="/data01/gregb/cubic-sphere/cache")
    ap.add_argument("--raw-coords", action="store_true", help="C++ behaviour: raw grid coordinates in the coefficients")
    ap.add_argument("--keep-center-sink", action="store_true", help="C++ behaviour: rho[(N-1)/2] = 0 each step")
    ap.add_argument("--text", action="store_true", help="write C++ text _r.dat files instead of .npy")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many CN steps (pilot runs)")
    ap.add_argument("--no-project", action="store_true", help="skip the final project() / _evolved output")
    ap.add_argument("--stop-on-nan", action="store_true", help="abort as soon as rho becomes non-finite")
    ap.add_argument("--chunk", type=int, default=2048, help="bricks per GPU chunk")
    ap.add_argument("--x64", action="store_true", help="float64 fields (slow on T4; validation only)")
    ap.add_argument("--seed", type=int, default=None, help="seed for fillFirstRho (creation mode)")
    args = ap.parse_args()

    import jax
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import geometry
    import project as proj
    import rhoio
    from ops import Operator
    from params import Params

    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:8.1f}s] {msg}", flush=True)

    print(f"\033[1;36mParams from {args.params}\033[0m", flush=True)
    p = Params.load(args.params)
    prefix = args.out_prefix or os.path.join("/data01/gregb/cubic-sphere", os.path.basename(p.path))
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    log(f"device: {jax.devices()[0]}  output prefix: {prefix}")
    log(f"physics: {'RAW coords (C++)' if args.raw_coords else 'centered coords'}, "
        f"{'with' if args.keep_center_sink else 'no'} center sink")

    geo = geometry.build(p, args.cache_dir, log)
    log(f"{geo.n_points} points, {geo.n_bricks} bricks")

    loaded = os.path.exists(p.pathdata + "_s.dat")
    h_op = p.h
    if loaded:
        n_file, h_file = rhoio.read_s_dat_header(p.pathdata + "_s.dat")
        if n_file != geo.n_points:
            raise SystemExit(f"{p.pathdata}_s.dat has {n_file} points, regenerated grid has {geo.n_points}")
        h_op = h_file  # Grid4(surfFile, ...) reads _h from the surface file
        log(f"\033[35mLoading {p.pathdata}\033[0m (h = {h_file})")

    center_idx = (geo.n_points - 1) // 2
    sink_slot = int(geo.slot[center_idx]) if args.keep_center_sink else None
    alpha = (p.T / 6.0) * p.deltaT * 0.5
    dtype = jnp.float64 if args.x64 else jnp.float32
    op = Operator(geo, h_op, alpha, p.center, p.bounds, centered=not args.raw_coords,
                  chunk=args.chunk, dtype=dtype, sink_slot=sink_slot)
    log(f"alpha = {alpha:g}; field {op.field_shape} = {op.NBt * geometry.BB * np.dtype(dtype).itemsize / 2**30:.2f} GiB each")

    if loaded:
        init_f = p.pathdata + "_init_r.dat"
        if os.path.exists(init_f):
            rho_pts = rhoio.read_r_dat(init_f, args.cache_dir, log).astype(np.float32)
        else:
            log("Warning: unable to load initial rho. Recreating (zeros)")
            rho_pts = np.zeros(geo.n_points, np.float32)
        if args.keep_center_sink:
            rho_pts[center_idx] = 0.0
    else:
        log("\033[35mCreating\033[0m (fillFirstRho: random start point)")
        rng = np.random.default_rng(args.seed)
        start = int(rng.integers(0, geo.n_points))
        gidx = geo.grid_index_of_slot(geo.slot[[start]])
        vert, w, _ = proj.stencil(geo, p, gidx)
        rho_pts = np.zeros(geo.n_points, np.float32)
        order = np.argsort(geo.slot)  # slot -> point index lookup
        pos = order[np.searchsorted(geo.slot, geo.slot_of_grid_index(vert[0]), sorter=order)]
        rho_pts[pos] = w[0]
        log(f"  start point {start}")

    rho = rhoio.from_points(rho_pts, geo.slot, op)
    writer = rhoio.AsyncWriter(text=args.text, log=log)
    writer.submit(prefix + "_init", rho_pts)
    del rho_pts

    print(f"\n\033[1;34mEvolving with deltaT = {p.deltaT:g}\033[0m", flush=True)
    done = 0
    nan_seen = False
    for step in range(p.step, p.total, p.step):
        if args.max_steps is not None and done >= args.max_steps:
            break
        t0 = time.time()
        its, ds = [], []
        for _ in range(p.step):
            rho, it, d = op.cn_step(rho)
            its.append(it)
            ds.append(d)
        its = [int(i) for i in its]
        ds = [float(d) for d in ds]
        dt = time.time() - t0
        done += p.step
        mass = float(jnp.sum(rho, dtype=jnp.float32))
        finite = bool(jnp.all(jnp.isfinite(rho)))
        log(f"Step {step}: {dt * 1000:.0f}ms  ({dt / p.step:.2f}s/step)  "
            f"jacobi iters {min(its)}-{max(its)}  last maxdiff {ds[-1]:.2e}  sum(rho) {mass:.6g}")
        if not finite and not nan_seen:
            nan_seen = True
            log("\033[1;31mWARNING: rho contains NaN/Inf\033[0m")
            if args.stop_on_nan:
                writer.wait()
                raise SystemExit(1)
        writer.submit_field(f"{prefix}_evolving_{step:04d}", rho, geo.slot)

    if not args.no_project:
        t0 = time.time()
        rho = proj.project_gpu(geo, p, op, rho, log=log)
        log(f"project() in {time.time() - t0:.0f}s")
        writer.submit_field(prefix + "_evolved", rho, geo.slot)
    writer.wait()
    log("\033[1;32mEnd\033[0m")


if __name__ == "__main__":
    main()
