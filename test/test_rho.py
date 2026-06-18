import numpy as np # pyright: ignore[reportMissingImports]
from pathlib import Path
from sys import argv

def test(filename):
    print("\n\nTesting ", filename)
    with open(filename) as f:
        nPoints = int(f.readline())
    rhos = np.loadtxt(filename, skiprows=2)
    np.where(rhos == np.nan, 0., None)
    print(nPoints, np.sum(rhos), np.min(rhos), np.max(rhos), np.count_nonzero(rhos))
    

def main():
    init_file = path / (name+"_init_r.dat")
    evolving_files = np.sort(list(path.rglob(name+"_evolving_*_r.dat")))
    final_file = path / (name+"_evolved_r.dat")

    test(init_file)

    for file in evolving_files:
        test(file)
    
    test(final_file)


if __name__ == "__main__":
    path = Path("../data")
    name = "test4_1"
    if len(argv) >= 2:
        name = argv[1]
    if len(argv) >= 3:
        path = Path(argv[2])
    print("Test Rho in ", path, " in the simulation ", name)

    main()
