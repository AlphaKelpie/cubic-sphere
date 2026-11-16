"""GPU (JAX) port of mainGrid4.cpp: Crank-Nicolson Fokker-Planck on S^3.

    python -m cubicsphere params.dat [options]

Same params file, same flow and file names as the C++ executable:
``<path>_init``, ``<path>_evolving_NNNN`` every ``step`` steps, then
``project()`` and ``<path>_evolved``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cubicsphere", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="?", default="params.dat")
    ap.add_argument("--solver", choices=["jacobi", "fixed-point", "bicgstab"], default="jacobi",
                    help="jacobi: diagonally scaled (default); fixed-point: the C++ iteration; "
                         "bicgstab: Krylov, any dt, but needs ~8 fields of GPU memory")
    ap.add_argument("--rtol", type=float, default=1e-5,
                    help="stop when max|x_new - x| <= atol + rtol * max|x| (default 1e-5)")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance (C++: 1e-3, rtol 0)")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--on-fail", choices=["stop", "continue"], default="stop",
                    help="what to do when a step does not converge (C++: continue)")
    ap.add_argument("--raw-coords", action="store_true",
                    help="C++ coefficients: D_ij and drift from raw grid coordinates instead of x - center")
    ap.add_argument("--center-sink", choices=["auto", "on", "off"], default="auto",
                    help="C++ rho[(n-1)/2] = 0; auto = on for the hypercube model, off for the crown")
    ap.add_argument("--init", choices=["auto", "file", "point", "shell"], default="auto",
                    help="auto: <pathdata>_init_r.{npy,dat} if present, else point; "
                         "point: C++ fillFirstRho (random point); shell: rho = 1 around the sphere")
    ap.add_argument("--init-file", default=None, help="prefix of the initial rho file (default <pathdata>_init)")
    ap.add_argument("--seed", type=int, default=None, help="seed for --init point")
    ap.add_argument("--format", choices=["npy", "text"], default="npy", help="output format (text = C++ _r.dat)")
    ap.add_argument("--cache-dir", default=None, help="geometry cache (default <dir of path>/cache)")
    ap.add_argument("--chunk", type=int, default=1024, help="bricks per GPU chunk (lower it if out of memory)")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many CN steps (pilot runs)")
    ap.add_argument("--estimate-spectrum", action="store_true",
                    help="print the spectral radius of the fixed-point and Jacobi iterations at this dt, then exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # JAX reserves 75% of the GPU by default; at h = 0.01 a Jacobi step needs
    # ~3-4 fields of 2.6 GB plus the mask, so let it use almost all of the T4.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    import jax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    from .geometry import B4, get_geometry
    from .params import load_params
    from .rhoio import CppOrder, Writer, has_rho, read_rho
    from .solver import Simulation, SolverOptions

    p = load_params(args.params)
    print(f"\033[1;36mParams from {args.params}\033[0m")
    print(f"  devices: {jax.devices()}")
    model = f"spherical crown, deltaC = {p.deltaC:g}" if p.crown else "whole hypercube"
    print(f"  model: {model}; h = {p.h:g}; dt = {p.dt:g}; alpha = T/6*dt/2 = {p.alpha:g}")

    cache = args.cache_dir or os.path.join(os.path.dirname(p.path) or ".", "cache")
    geo = get_geometry(p, cache)
    opts = SolverOptions(solver=args.solver, rtol=args.rtol, atol=args.atol, max_iter=args.max_iter,
                         raw_coords=args.raw_coords, center_sink=args.center_sink,
                         chunk=args.chunk, dtype=args.dtype)
    sim = Simulation(geo, p, opts)
    order = CppOrder(geo)
    nbytes = sim.NB * B4 * jnp.dtype(args.dtype).itemsize
    print(f"  grid {tuple(int(v) for v in geo.n)}, {geo.npoints:,} active points in {geo.nb:,} bricks "
          f"({geo.npoints / max(1, geo.nb * B4):.0%} filled); {nbytes / 2**30:.2f} GiB per field")

    if args.estimate_spectrum:
        rho = sim.estimate_spectral_radius()
        print(f"  spectral radius at this dt: fixed-point (C++) {rho['fixed-point']:.4g}, "
              f"Jacobi {rho['jacobi']:.4g}  (must be < 1 to converge)")
        return 0

    # ------------------------------------------------------------- initial rho
    init = args.init
    prefix = args.init_file or (p.pathdata + "_init" if p.pathdata else None)
    if init == "auto":
        init = "file" if prefix and has_rho(prefix) else "point"
    if init == "file":
        if not prefix or not has_rho(prefix):
            sys.exit(f"Initial rho file not found: {prefix}_r.npy / _r.dat")
        print(f"\033[35mLoading {prefix}\033[0m")
        rho = sim.to_device(order.from_cpp(read_rho(prefix)))
    elif init == "point":
        k = int(np.random.default_rng(args.seed).integers(geo.npoints))
        print(f"\033[35mCreating\033[0m: unit mass at C++ point {k}, projected on the sphere")
        delta = sim.zeros().reshape(-1).at[order.cpp_index_to_flat(k)].set(1.0).reshape(sim.zeros().shape)
        rho = sim.project(delta)
    else:
        print("\033[35mCreating\033[0m: rho = 1 on the grid points around the sphere")
        rho = jnp.where(sim.project(sim.d_mask.astype(sim.dtype)) > 0, 1.0, 0.0).astype(sim.dtype)

    sink = args.center_sink
    if sink == "on" or (sink == "auto" and not p.crown):
        sim.sink = order.cpp_index_to_flat((geo.npoints - 1) // 2)
        if init == "file":
            rho = sim._set_sink(rho)  # C++ loading constructor
        print(f"  centre sink at C++ point {(geo.npoints - 1) // 2}")

    writer = Writer(args.format)
    path = p.path
    # Keep the params next to the outputs: the grid is not saved (no _s.dat), and
    # test/test_rho.py rebuilds the point radii from this copy.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    shutil.copyfile(args.params, path + "_params.dat")
    writer.submit(path + "_init", order.to_cpp(rho))
    print(f"  init: {sim.stats(rho)}")

    # ------------------------------------------------------------------ evolve
    print(f"\n\033[1;34mEvolving with deltaT = {p.deltaT:g} (dt = {p.dt:g}), solver = {opts.solver}\033[0m")
    done = 0
    stop = False
    for step in range(p.step, p.total, p.step):
        t0 = time.time()
        iters = []
        for _ in range(p.step):
            rho, it, diff, ok = sim.step(rho)
            iters.append(int(it))
            done += 1
            if not bool(ok):
                msg = (f"\n\033[31mCN step {done} did not converge: {int(it)} iterations, "
                       f"relative change {float(diff):.3g}.\033[0m Reduce deltaT, use --solver bicgstab "
                       f"(more memory), or check --estimate-spectrum.")
                if args.on_fail == "stop":
                    writer.wait()
                    sys.exit(msg)
                print(msg)
            if args.max_steps is not None and done >= args.max_steps:
                stop = True
                break
        rho.block_until_ready()
        it_info = f"iterations {min(iters)}-{max(iters)}" if iters and iters[0] >= 0 else "bicgstab"
        print(f"Step {step}: {time.time() - t0:.2f} s, {it_info}\t", end="", flush=True)
        writer.submit(path + f"_evolving_{step:04d}", order.to_cpp(rho))
        s = sim.stats(rho)
        print(f"sum {s['sum']:.6g}, max {s['max']:.3g}, min {s['min']:.3g}" + (f", NaN {s['nan']}" if s["nan"] else ""))
        if stop:
            print(f"--max-steps {args.max_steps} reached")
            break

    t0 = time.time()
    rho = sim.project(rho)
    writer.submit(path + "_evolved", order.to_cpp(rho))
    print(f"project(): {time.time() - t0:.2f} s, {sim.stats(rho)}")
    writer.wait()
    print("\033[1;32mEnd\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
