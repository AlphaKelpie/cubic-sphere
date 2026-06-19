import numpy as np # pyright: ignore[reportMissingImports]
from pathlib import Path
from sys import argv

CENTER = np.array([2., 2., 2., 2.])
RADIUS = 1.

def phi(p) :
    return np.sqrt(np.sum(np.pow(p - CENTER, 2))) - RADIUS

def test_sum1(weights, nPoints) :
    sums = np.sum(weights, axis=1)
    no_ones = np.arange(nPoints)[~np.isclose(sums, 1.)]
    print(f"No-one-sum indexes: {len(no_ones)}")
    if len(no_ones) > 10:
        print(no_ones[:10])
    elif len(no_ones) > 0:
        print(no_ones)

def repeat(inds, volume, h, half_cubic1, half_cubic2) :
    proj = inds.astype(int)
    points = np.array([volume[p] for p in proj])

    if not np.allclose(points[1] - points[0], [0, 0, 0, h]) :
        return 1
    if not np.allclose(points[2] - points[0], [0, 0, h, 0]) :
        return 2
    if not np.allclose(points[3] - points[0], [0, 0, h, h]) :
        return 3
    if not np.allclose(points[4] - points[0], [0, h, 0, 0]) :
        return 4
    if not np.allclose(points[5] - points[0], [0, h, 0, h]) :
        return 5
    if not np.allclose(points[6] - points[0], [0, h, h, 0]) :
        return 6
    if not np.allclose(points[7] - points[0], [0, h, h, h]) :
        return 7
    if not np.allclose(points[8] - points[0], [h, 0, 0, 0]) :
        return 8
    if not np.allclose(points[9] - points[0], [h, 0, 0, h]) :
        return 9
    if not np.allclose(points[10] - points[0], [h, 0, h, 0]) :
        return 10
    if not np.allclose(points[11] - points[0], [h, 0, h, h]) :
        return 11
    if not np.allclose(points[12] - points[0], [h, h, 0, 0]) :
        return 12
    if not np.allclose(points[13] - points[0], [h, h, 0, h]) :
        return 13
    if not np.allclose(points[14] - points[0], [h, h, h, 0]) :
        return 14
    if not np.allclose(points[15] - points[0], [h, h, h, h]) :
        return 15

    dists = np.apply_along_axis(phi, 1, points)
    dists[np.isclose(dists, 0, atol=1e-7)] = 0.
    if np.all(dists[half_cubic1] * dists[half_cubic2] > 0) :
        return 16
    
    return 0

def test_cubic(inds, nPoints) :
    vol_file = path / (name+"_s.dat")
    with open(vol_file) as f:
        h = float(f.readline().split(' ')[-1])
    print("H: ", h)

    volume = np.loadtxt(vol_file, skiprows=2, dtype=np.float32)
    if nPoints != len(volume) :
        print(f"\nMismach lens projection ({nPoints}) and volume ({len(volume)})")
    
    if volume.shape[1] != 4:
        print("Wrong volume shape: {volume.shape}")

    half_lenght = inds.shape[1] // 2
    half_cubic1 = list(range(0, half_lenght, 1))
    half_cubic2 = list(range(-1, -half_lenght - 1, -1))

    results = np.apply_along_axis(repeat, 1, inds, volume, h, half_cubic1, half_cubic2)
    for i, r in enumerate(results) :
        if r != 0 :
            print(i, r)
    # for idx in range(0, nPoints) :
    #     print(f"\r{idx}/{nPoints}", end='')
    #     proj = inds[idx].astype(int)
    #     points = np.array([volume[p] for p in proj])

    #     if not np.allclose(points[1] - points[0], [0, 0, 0, h]) :
    #         print(f"Not correct cubic: {idx}, {1}")
    #         break
    #     if not np.allclose(points[2] - points[0], [0, 0, h, 0]) :
    #         print(f"Not correct cubic: {idx}, {2}")
    #         break
    #     if not np.allclose(points[3] - points[0], [0, 0, h, h]) :
    #         print(f"Not correct cubic: {idx}, {3}")
    #         break
    #     if not np.allclose(points[4] - points[0], [0, h, 0, 0]) :
    #         print(f"Not correct cubic: {idx}, {4}")
    #         break
    #     if not np.allclose(points[5] - points[0], [0, h, 0, h]) :
    #         print(f"Not correct cubic: {idx}, {5}")
    #         break
    #     if not np.allclose(points[6] - points[0], [0, h, h, 0]) :
    #         print(f"Not correct cubic: {idx}, {6}")
    #         break
    #     if not np.allclose(points[7] - points[0], [0, h, h, h]) :
    #         print(f"Not correct cubic: {idx}, {7}")
    #         break
    #     if not np.allclose(points[8] - points[0], [h, 0, 0, 0]) :
    #         print(f"Not correct cubic: {idx}, {8}")
    #         break
    #     if not np.allclose(points[9] - points[0], [h, 0, 0, h]) :
    #         print(f"Not correct cubic: {idx}, {9}")
    #         break
    #     if not np.allclose(points[10] - points[0], [h, 0, h, 0]) :
    #         print(f"Not correct cubic: {idx}, {10}")
    #         break
    #     if not np.allclose(points[11] - points[0], [h, 0, h, h]) :
    #         print(f"Not correct cubic: {idx}, {11}")
    #         break
    #     if not np.allclose(points[12] - points[0], [h, h, 0, 0]) :
    #         print(f"Not correct cubic: {idx}, {12}")
    #         break
    #     if not np.allclose(points[13] - points[0], [h, h, 0, h]) :
    #         print(f"Not correct cubic: {idx}, {13}")
    #         break
    #     if not np.allclose(points[14] - points[0], [h, h, h, 0]) :
    #         print(f"Not correct cubic: {idx}, {14}")
    #         break
    #     if not np.allclose(points[15] - points[0], [h, h, h, h]) :
    #         print(f"Not correct cubic: {idx}, {15}")
    #         break

    #     dists = np.apply_along_axis(phi, 1, points)
    #     dists[np.isclose(dists, 0, atol=1e-7)] = 0.
    #     if np.all(dists[half_cubic1] * dists[half_cubic2] > 0) :
    #         print(f"Vertexs on the same semispace: {idx}")
    #         print(dists)
    #         break
    # print("")
        
def main():
    file = path / (name+"_p.dat")

    with open(file) as f:
        nPoints = int(f.readline())
    projection = np.loadtxt(file, skiprows=2)

    print("Default tests")
    if nPoints != len(projection) :
        print(f"Number of points mismached: {nPoints} != {len(projection)}")
    if projection.shape[1] != 32:
        print(f"Number of neighbours per points mismached: {projection.shape[1]}")
    
    if np.any(projection[:, index_col] >= nPoints) :
        print("Too big index")
    if np.any(projection[:, index_col] < -1) :
        print("Too low index")
    
    # test somma 1; siano sulla sfera; formino un cubo;
    print("\nTest sum1")
    # test_sum1(projection[:, weight_col], nPoints)

    print("\nTest projection")
    test_cubic(projection[:, index_col], nPoints)

if __name__ == "__main__":
    path = Path("../data")
    name = "test4_1"
    if len(argv) >= 2:
        name = argv[1]
    if len(argv) >= 3:
        path = Path(argv[2])
    print("Test Projection in ", path, " in the simulation ", name)
    index_col = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    weight_col = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

    main()
