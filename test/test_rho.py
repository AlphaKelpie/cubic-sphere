"""Radial analysis of the rho snapshots of one simulation.

    python test_rho.py ../data/radial6_08                    # per-snapshot plots
    python test_rho.py ../data/radial6_08 --variance         # + variance-vs-step plot

The argument is the simulation path with its prefix (the ``path`` of the params
file). The script analyses, when present,

    <prefix>_init_r.{npy,dat}
    <prefix>_evolving_NNNN_r.{npy,dat}      (every one, in step order)
    <prefix>_evolved_r.{npy,dat}

and works on the output of both programs: the C++ executable (text ``.dat``) and
``python -m cubicsphere`` (``.npy``, or ``.dat`` with ``--format text``). If both
extensions exist for a snapshot the ``.npy`` is used.

Each snapshot gets ``<snapshot>_r.png`` (radial distribution of rho, std of rho
on each sphere of constant radius, and its ratio to the mean). Besides, the mean
radius and the variance of the radial distribution of every snapshot are
written to ``<prefix>_radial_stats.csv``; with ``--variance`` the variance of
init + evolving snapshots is plotted in ``<prefix>_variance.pdf``.

The radius of every point comes from the point set, which is found in this order:
  1. ``--volume``: a C++ surface file ``X_s.dat`` (or just ``X``);
  2. ``--params``: the params file of the run (the grid is regenerated);
  3. ``<prefix>_s.dat``: written by the C++ code when it creates a grid;
  4. ``<prefix>_params.dat``: copied by ``cubicsphere`` next to its outputs.
A C++ run that loaded its grid (``pathdata``) has no ``<prefix>_s.dat``: pass
``--volume <pathdata>`` or ``--params``.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = 1e6  # radii are compared at 1e-6, the precision of the C++ _s.dat (6 digits)


# ----------------------------------------------------------------------- volume
class Volume:
    """Quantised radius of every point (in C++ point order) and per-radius counts."""

    def __init__(self, key_chunks):
        self.chunks = [k for k in key_chunks if len(k)]
        self.offsets = np.cumsum([0] + [len(k) for k in self.chunks])
        self.n = int(self.offsets[-1])
        self.size = max(int(k.max()) for k in self.chunks) + 1
        counts = np.zeros(self.size, dtype=np.int64)
        for k in self.chunks:
            counts += np.bincount(k, minlength=self.size)
        self.bins = np.nonzero(counts)[0]
        self.counts = counts[self.bins]
        self.radii = self.bins / RES

    def radial(self, rho):
        """Per distinct radius: sum, mean and (population) std of rho. Also the number of
        non-finite values, which are counted as 0."""
        s1 = np.zeros(self.size)
        s2 = np.zeros(self.size)
        bad = 0
        for i, k in enumerate(self.chunks):
            r = np.asarray(rho[self.offsets[i]:self.offsets[i + 1]], dtype=np.float64)
            finite = np.isfinite(r)
            bad += int(r.size - finite.sum())
            r = np.where(finite, r, 0.0)
            s1 += np.bincount(k, weights=r, minlength=self.size)
            s2 += np.bincount(k, weights=r * r, minlength=self.size)
        total = s1[self.bins]
        mean = total / self.counts
        std = np.sqrt(np.clip(s2[self.bins] / self.counts - mean**2, 0.0, None))
        return total, mean, std, bad


def volume_from_surface(path, center):
    """C++ ``_s.dat``: first row ``nPoints h``, then one ``w x y z`` row per point."""
    with open(path, "rb") as f:
        n = int(f.readline().split()[0])
        xyz = np.fromfile(f, dtype=np.float64, sep=" ")
    if xyz.size != 4 * n:
        raise SystemExit(f"{path}: header says {n} points, found {xyz.size / 4:g}")
    xyz = xyz.reshape(n, 4) - np.asarray(center, dtype=np.float64)
    r = np.sqrt((xyz**2).sum(axis=1))
    return Volume([np.rint(r * RES).astype(np.int32)])


def six_digits(x):
    """Round to 6 significant digits, like the default ostream precision used for ``_s.dat``."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    nz = x != 0.0
    scale = 10.0 ** (5 - np.floor(np.log10(np.abs(x[nz]))))
    out[nz] = np.round(x[nz] * scale) / scale
    return out


def volume_from_params(params):
    """Regenerate the point set of a params file (same set and order as the C++
    ``createVol`` / ``createVolNeigh``), one w-plane at a time."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cubicsphere.geometry import Geometry

    geo = Geometry(params)
    c = geo.center
    # The C++ keeps coordinates as float and ``_s.dat`` prints them with 6 digits;
    # do the same, so that points of equal radius get exactly equal radii (h is
    # parsed as a float too, which leaves ~1e-7 noise in min + h * i).
    axes = [six_digits(np.float32(a)) for a in geo.coords]
    chunks = []
    for iw in range(int(geo.nbox[0])):
        ii, jj, kk = np.nonzero(geo.mask_plane(iw))
        dw = axes[0][geo.lo[0] + iw] - c[0]
        dx = axes[1][geo.lo[1] + ii] - c[1]
        dy = axes[2][geo.lo[2] + jj] - c[2]
        dz = axes[3][geo.lo[3] + kk] - c[3]
        r = np.sqrt(dw * dw + dx * dx + dy * dy + dz * dz)
        chunks.append(np.rint(r * RES).astype(np.int32))
    return Volume(chunks)


def load_volume(args, prefix):
    """Returns (Volume, description)."""
    from_params = None
    if args.params:
        from_params = Path(args.params)
    elif Path(str(prefix) + "_params.dat").exists():
        from_params = Path(str(prefix) + "_params.dat")
    params = None
    if from_params is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cubicsphere.params import load_params

        params = load_params(str(from_params))

    center = params.center if params is not None else (args.center,) * 4

    if args.volume:
        s = Path(args.volume)
        s = s if s.suffix == ".dat" else Path(str(s) + "_s.dat")
        return volume_from_surface(s, center), f"surface file {s}"
    if args.params:
        return volume_from_params(params), f"params file {args.params}"
    s = Path(str(prefix) + "_s.dat")
    if s.exists():
        return volume_from_surface(s, center), f"surface file {s}"
    if params is not None:
        return volume_from_params(params), f"params file {from_params}"
    raise SystemExit(
        f"Cannot find the point set of '{prefix}': there is no {s.name} or "
        f"{Path(str(prefix) + '_params.dat').name}. Pass --volume (C++ _s.dat) or --params."
    )


# ----------------------------------------------------------------------- files
def read_rho(path):
    """rho in C++ point order from ``.npy`` or the C++ text format (N, blank line, one value per line)."""
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    with open(path, "rb") as f:
        n = int(f.readline().split()[0])
        pos = f.tell()
        values = np.fromfile(f, dtype=np.float32, sep=" ")
        if values.size != n:  # e.g. an unparsable token: slow but tolerant path
            f.seek(pos)
            values = np.array(f.read().split(), dtype=np.float32)
    if values.size != n:
        raise SystemExit(f"{path}: header says {n} values, found {values.size}")
    return values


def find_snapshots(prefix):
    """[(label, step or None, path)]: init, evolving by step, evolved. ``.npy`` wins over ``.dat``."""
    folder, name = prefix.parent, prefix.name
    found = {}
    for f in folder.iterdir():
        m = re.fullmatch(re.escape(name) + r"_(init|evolved|evolving_(\d+))_r\.(npy|dat)", f.name)
        if m:
            key = m.group(1)
            if key not in found or f.suffix == ".npy":
                found[key] = f
    out = []
    if "init" in found:
        out.append(("init", 0, found["init"]))
    steps = sorted((int(k.split("_")[1]), k) for k in found if k.startswith("evolving_"))
    out += [(f"evolving_{s:04d}", s, found[k]) for s, k in steps]
    if "evolved" in found:
        out.append(("evolved", None, found["evolved"]))
    return out


# ------------------------------------------------------------------- analysis
def analyse(vol, rho, png, title):
    """Prints a summary, saves the 3-panel plot and returns (mean radius, variance, mass)."""
    if len(rho) != vol.n:
        raise SystemExit(
            f"{title}: rho has {len(rho)} points but the point set has {vol.n} "
            "(wrong --volume/--params for this simulation?)"
        )
    r = vol.radii
    total, mean, std, bad = vol.radial(rho)
    finite = np.asarray(rho)[np.isfinite(rho)] if bad else np.asarray(rho)
    mass = float(total.sum())
    print(
        f"{vol.n} points, sum {mass:.8g}, min {finite.min() if finite.size else np.nan:.4g}, "
        f"max {finite.max() if finite.size else np.nan:.4g}, "
        f"{int(np.count_nonzero(finite > 1e-7))} points > 1e-7"
    )
    if bad:
        print(f"  WARNING: {bad} non-finite values (NaN/inf), counted as 0")
    if bad == vol.n or mass == 0.0:
        print("  no mass: plot skipped")
        return np.nan, np.nan, mass

    m = np.sum(r * total) / mass
    v = np.sum((r - m) ** 2 * total) / mass

    # first radius of the all-zero tail of the radial distribution (4 if there is none)
    nonzero = np.nonzero(np.abs(total) > 1e-5)[0]
    last = nonzero[-1] if nonzero.size else -1
    mrho = r[last + 1] if last + 1 < len(r) else 4.0

    mean_abs = np.abs(mean)
    ratios = np.divide(std, mean_abs, out=np.zeros_like(std), where=mean_abs != 0.0)
    count = int(np.count_nonzero(ratios > 100.0))
    far = np.nonzero(r > 3.5)[0]
    stop = far[0] if far.size else len(r)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].bar(r[:stop], total[:stop], width=0.02)
    axs[0].set_ylabel("rho")
    axs[0].set_xlabel("radius")
    axs[0].set_title("Rho distribution")
    axs[0].text(0.95, 0.95, f"{mrho}\n{m}\n{v}", ha="right", va="top", transform=axs[0].transAxes)

    axs[1].hist(std, bins=30, log=True)
    axs[1].set_ylabel("occurences")
    axs[1].set_xlabel(r"$\Delta \rho$")
    axs[1].set_title("Std")

    top = min(ratios.max(), 100.0) or 1.0
    axs[2].hist(ratios, bins=30, range=(0.0, top), log=True)
    axs[2].set_ylabel("occurences")
    axs[2].set_xlabel(r"$\Delta \rho/ \bar{\rho}$")
    axs[2].set_title("Relative Std")
    axs[2].text(0.95, 0.95, count, ha="right", va="top", transform=axs[2].transAxes)
    fig.suptitle(title)
    fig.savefig(png)
    plt.close(fig)
    return m, v, mass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("simulation", help="simulation path with prefix, e.g. ../data/radial6_08")
    ap.add_argument("--variance", action="store_true",
                    help="also plot the variance of the radial distribution vs step (<prefix>_variance.pdf)")
    ap.add_argument("--volume", help="C++ surface file (X_s.dat) or its prefix X, with the point coordinates")
    ap.add_argument("--params", help="params file of the run; the point set is regenerated from it")
    ap.add_argument("--center", type=float, default=2.0,
                    help="sphere centre (all four coordinates) when only a surface file is known (default 2)")
    args = ap.parse_args()

    prefix = Path(args.simulation)
    snaps = find_snapshots(prefix)
    if not snaps:
        raise SystemExit(f"No snapshot found: expected {prefix}_init_r.[npy|dat], {prefix}_evolving_NNNN_r.* ...")
    vol, source = load_volume(args, prefix)
    print(f"Test Rho in {prefix.parent} in the simulation {prefix.name}")
    print(f"Point set: {vol.n} points, {len(vol.radii)} distinct radii, from {source}")

    rows = []
    for label, step, path in snaps:
        print(f"\n\nTesting {path}")
        m, v, mass = analyse(vol, read_rho(path), path.with_suffix(".png"), path.stem[:-2])
        rows.append((label, step, m, v, mass))

    stats = Path(str(prefix) + "_radial_stats.csv")
    with open(stats, "w") as f:
        f.write("snapshot,step,mean_radius,variance,mass\n")
        for label, step, m, v, mass in rows:
            f.write(f"{label},{'' if step is None else step},{m},{v},{mass}\n")
    print(f"\nWrote {stats}")

    if args.variance:
        evol = [(s, v) for _, s, _, v, _ in rows if s is not None]  # as before, the evolved row is left out
        fig, ax = plt.subplots(1, 1)
        ax.plot([s for s, _ in evol], [v for _, v in evol], "b.-", label=prefix.name)
        ax.set_xlabel("step")
        ax.set_ylabel("variance of the radial distribution")
        ax.legend()
        out = Path(str(prefix) + "_variance.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
