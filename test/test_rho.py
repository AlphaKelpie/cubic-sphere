import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def ctor():
    vol = np.loadtxt(volumefile, skiprows=2)
    vol -= CENTER
    radii = np.linalg.norm(vol, axis=1)
    uniqr, uniqc = np.unique(radii, return_counts=True)
    return radii, uniqr, uniqc

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def diff(rhos, filename):
    searchsorted = np.searchsorted(unradii, radii)
    val = np.bincount(searchsorted, rhos)
    stop = np.argwhere(unradii > 3.5)[0][0]
    m = np.average(unradii, weights=val)
    v = np.average((unradii - m)**2, weights=val)

    # for plots
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
    axs[0].bar(unradii[:stop], val[:stop], width=.02)
    axs[0].set_ylabel("rho")
    axs[0].set_xlabel("radius")
    axs[0].set_title("Rho distribution")
    axs[0].text(.95, .95, f"{mrho}\n{m}\n{v}", ha='right',
                va='top', transform=axs[0].transAxes)

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

    return m, v

def test(filename):
    print("\n\nTesting ", filename)
    with open(filename) as f:
        nPoints = int(f.readline())
    rhos = np.loadtxt(filename, skiprows=2)
    np.where(rhos == np.nan, 0., None)
    print(nPoints, np.sum(rhos), np.min(rhos), np.max(rhos), np.count_nonzero(rhos > 1e-7))

    return diff(rhos, filename)

def main():
    times = []
    averages = []
    variances = []
    init_file = path / (name+"_init_r.dat")
    evolving_files = np.sort(list(path.rglob(name+"_evolving_*_r.dat")))
    final_file = path / (name+"_evolved_r.dat")

    times.append(0)
    m, v = test(init_file)
    averages.append(m)
    variances.append(v)

    for epoch in epochs:
        for file in evolving_files:
            if epoch in str(file):
                times.append(int(str(file).split('_')[-2]))
                m, v = test(file)
                averages.append(m)
                variances.append(v)
                break
    
    m, v = test(final_file)
    times.append(times[-1] + 1000)
    averages.append(m)
    variances.append(v)
    return times, averages, variances


if __name__ == "__main__":
    path = Path("../data")
    epochs = list(range(100, 10000, 100)) + list(range(10000, 100001, 1000))
    for i in range(len(epochs)):
        epochs[i] = '_' + str(epochs[i]).zfill(4) + '_'
    p = [1., 1., .2]
    CENTER = 2.

    name = "radial6_08"
    volfile = "base08"
    print("Test Rho in ", path, " in the simulation ", name)
    volumefile = "../data/" + volfile + "_s.dat"
    print(volumefile)
    radii, unradii, unicount = ctor()
    times08, averages08, variances08 = main()
    with open("../data/radial6_08.dat", 'w') as f:
        raws = list(f"{t},{a},{v}" for t, a, v in zip(times08, averages08, variances08))
        f.write('\n'.join(r for r in raws))

    name = "radial6_1"
    volfile = "base1"
    print("Test Rho in ", path, " in the simulation ", name)
    volumefile = "../data/" + volfile + "_s.dat"
    print(volumefile)
    radii, unradii, unicount = ctor()
    times1, averages1, variances1 = main()
    with open("../data/radial6_1.dat", 'w') as f:
        raws = list(f"{t},{a},{v}" for t, a, v in zip(times1, averages1, variances1))
        f.write('\n'.join(r for r in raws))

    name = "radial6_16"
    volfile = "base16"
    print("Test Rho in ", path, " in the simulation ", name)
    volumefile = "../data/" + volfile + "_s.dat"
    print(volumefile)
    radii, unradii, unicount = ctor()
    times16, averages16, variances16 = main()
    with open("../data/radial6_16.dat", 'w') as f:
        raws = list(f"{t},{a},{v}" for t, a, v in zip(times16, averages16, variances16))
        f.write('\n'.join(r for r in raws))

    # plot variances
    LAST_ROW = -1
    dat = np.loadtxt("../data/radial6_08.dat", delimiter=',')
    times08 = dat[:LAST_ROW,0]
    variances08 = dat[:LAST_ROW, 2]
    dat = np.loadtxt("../data/radial6_1.dat", delimiter=',')
    times1 = dat[:LAST_ROW,0]
    variances1 = dat[:LAST_ROW, 2]
    dat = np.loadtxt("../data/radial6_16.dat", delimiter=',')
    times16 = dat[:LAST_ROW,0]
    variances16 = dat[:LAST_ROW, 2]
    fig, ax = plt.subplots(1,1)
    ax.plot(times08, variances08, 'r', label="08")
    ax.plot(times1, variances1, 'b.-', label="1")
    ax.plot(times16, variances16, 'g--', label="16")
    ax.legend()
    fig.savefig("../data/variance6.pdf")
