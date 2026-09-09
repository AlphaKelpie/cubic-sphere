import numpy as np # pyright: ignore[reportMissingImports]
from pathlib import Path
from sys import argv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, a, b):
    return 1 / np.sqrt(2 * np.pi * b) * np.exp(-.5 * (x - a)**2 / b)

def get_volume():
    filename = path / (name+"_s.dat")
    with open(filename) as f:
        s = f.readline().split()
        nPoints = int(s[0])
        h = float(s[1])
    side = np.pow(nPoints, 0.25).astype(int)
    volumeMap = np.arange(nPoints).reshape(side, side, side, side)
    return volumeMap, nPoints, h, side

def plot(rhos, axis, side, filename, h):
    fig, ax = plt.subplots()

    x = np.arange(0, side*h, h)
    ax.plot(x, rhos, color='b', label="Data")

    popt, pcov = curve_fit(gauss, x, rhos)
    ax.plot(x, gauss(x, *popt), color='r',
        label=f"Fit: u={popt[0]:.2f} v={popt[1]:.2f}")
    ax.set_xlabel("ax")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title(f"{filename.stem} on {axis}")
    fig.savefig(str(filename.parent / filename.stem) + f"_{axis}.pdf")
    plt.close()
    print(axis, popt)

def test(filename, vMap, num, side, h):
    print("\n\nTesting ", filename)
    with open(filename) as f:
        nPoints = int(f.readline())
    rhos = np.loadtxt(filename, skiprows=2)
    if nPoints != num:
        print("Err: unmach between num and nPoints")
        return
    if len(rhos) != nPoints:
        print("Err: unmach between rhos len and nPoints")
        return
    np.where(rhos == np.nan, 0., None)
    print(nPoints, np.sum(rhos), np.min(rhos), np.max(rhos), np.count_nonzero(rhos > 1e-7))

    tots = np.zeros(side)
    for idx in range(side):
        tots[idx] = np.sum(rhos[vMap[idx,:,:,:].flatten()])
    plot(tots, 'w', side, filename, h)
    for idx in range(side):
        tots[idx] = np.sum(rhos[vMap[:,idx,:,:].flatten()])
    plot(tots, 'x', side, filename, h)
    for idx in range(side):
        tots[idx] = np.sum(rhos[vMap[:,:,idx,:].flatten()])
    plot(tots, 'y', side, filename, h)
    for idx in range(side):
        tots[idx] = np.sum(rhos[vMap[:,:,:,idx].flatten()])
    plot(tots, 'z', side, filename, h)
    

def main():
    volumeMap, nPoints, h, side = get_volume()

    init_file = path / (name+"_init_r.dat")
    evolving_files = np.sort(list(path.rglob(name+"_evolving_*_r.dat")))
    final_file = path / (name+"_evolved_r.dat")

    test(init_file, volumeMap, nPoints, side, h)

    for file in evolving_files:
        test(file, volumeMap, nPoints, side, h)
    
    test(final_file, volumeMap, nPoints, side, h)


if __name__ == "__main__":
    path = Path("../data")
    name = "test4_1"
    if len(argv) >= 2:
        name = argv[1]
    if len(argv) >= 3:
        path = Path(argv[2])
    print("Test difusion in ", path, " in the simulation ", name)

    main()
