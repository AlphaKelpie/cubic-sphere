import numpy as np # pyright: ignore[reportMissingImports]
from pathlib import Path
from sys import argv
import matplotlib.pyplot as plt

def ctor():
    vol = np.loadtxt(VOLUMEFILE, skiprows=2)
    vol -= CENTER
    radii = np.linalg.norm(vol, axis=1)
    uniqr = np.sort(np.unique(radii))
    return radii, uniqr

def diff(rhos, radii, unradii, filename):
    searchsorted = np.searchsorted(unradii, radii)
    # if (filename.stem[-4] == '0' or 'init' in filename.stem):
    #     with open(filename.with_suffix('.txt'), 'w') as f:
    val = np.bincount(searchsorted, rhos)

    stds = []
    ratios = []
    count = 0
    mrho = 4.
    for idx in range(len(unradii)):
        if np.all(np.isclose(val[-idx-1:], [0.], atol=1e-5)):
            mrho = unradii[len(unradii) - idx - 1]
        rrhos = rhos[searchsorted == idx]
        mean = np.absolute(np.mean(rrhos))
        stds.append(np.std(rrhos))
        ratios.append((stds[-1] / mean) if mean != 0. else 0.)
        if ratios[-1] > 100.:
            count += 1

    fig, axs = plt.subplots(1,3, figsize=(12,6))
    axs[0].bar(unradii, val, width=.02)
    # axs[0].set_yscale('log')
    axs[0].set_ylabel("rho")
    axs[0].set_xlabel("radius")
    axs[0].set_title("Rho distribution")
    axs[0].text(.95, .95, mrho, ha='right', va='top', transform=axs[0].transAxes)

    axs[1].hist(stds, bins=30, log=True)
    axs[1].set_ylabel("occurences")
    axs[1].set_xlabel(r"$\Delta \rho$")
    axs[1].set_title("Std")

    axs[2].hist(ratios, bins=30, range=(0., np.min([np.max(ratios),100])), log=True)
    axs[2].set_ylabel("occurences")
    axs[2].set_xlabel(r"$\Delta \rho/ \bar{\rho}$")
    axs[2].set_title("Relative Std")
    axs[2].text(.95, .95, count, ha='right', va='top', transform=axs[2].transAxes)
    fig.suptitle(str(filename.stem[:-2]))
    fig.savefig(filename.with_suffix('.png'))
    plt.close(fig)

def test(filename, radii, unradii):
    print("\n\nTesting ", filename)
    with open(filename) as f:
        nPoints = int(f.readline())
    rhos = np.loadtxt(filename, skiprows=2)
    np.where(rhos == np.nan, 0., None)
    print(nPoints, np.sum(rhos), np.min(rhos), np.max(rhos), np.count_nonzero(rhos > 1e-7))

    diff(rhos, radii, unradii, filename)

def main():
    init_file = path / (name+"_init_r.dat")
    evolving_files = np.sort(list(path.rglob(name+"_evolving_*_r.dat")))
    final_file = path / (name+"_evolved_r.dat")

    radii, unradii = ctor()

    test(init_file, radii, unradii)

    for file in evolving_files:
        test(file, radii, unradii)
    
    test(final_file, radii, unradii)


if __name__ == "__main__":
    path = Path("../data")
    name = "test4_1"
    if len(argv) >= 2:
        name = argv[1]
    if len(argv) >= 3:
        volfile = argv[2]
    print("Test Rho in ", path, " in the simulation ", name)
    VOLUMEFILE = "../data/" + volfile + "_s.dat"
    print(VOLUMEFILE)
    CENTER = 2.
    main()
