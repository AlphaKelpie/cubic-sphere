import numpy as np # pyright: ignore[reportMissingImports]
from pathlib import Path
from sys import argv

def test_surface(neigh, nPoints) :
    vol_file = path / (name+"_s.dat")
    with open(vol_file) as f:
        h = float(f.readline().split(' ')[-1])
    print("H: ", h)

    volume = np.loadtxt(vol_file, skiprows=2, dtype=np.float32)
    if nPoints != len(volume) :
        print(f"\nMismach lens neighbour ({nPoints}) and volume ({len(volume)})")
    
    if volume.shape[1] != 4:
        print("Wrong volume shape: {volume.shape}")
    
    nulPoint = [-1., -1., -1., -1.]
    volume = np.concat([volume, [nulPoint]])
    vmin = volume[0]
    vmax = volume[-2]
    for idx in range(0, nPoints) :
        print(f"\r{idx}/{nPoints}", end='')
        n = neigh[idx]
        points = np.array([volume[p] for p in n])
        if not np.array_equal(points[1], nulPoint) :
            if not np.allclose(points[1] - points[0], [-h, 0, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {1}")
                break
        else:
            if points[0][0] !=  vmin[0] :
                print(f"\nWrong '-1' assignment: {idx}, {1}")
                break

        if not np.array_equal(points[2], nulPoint) :
            if not np.allclose(points[2] - points[0], [h, 0, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {2}")
                break
        else:
            if points[0][0] != vmax[0] :
                print(f"\nWrong '-1' assignment: {idx}, {2}")
                break

        if not np.array_equal(points[3], nulPoint) :
            if not np.allclose(points[3] - points[0], [0, -h, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {3}")
                break
        else:
            if points[0][1] != vmin[1] :
                print(f"\nWrong '-1' assignment: {idx}, {3}")
                break

        if not np.array_equal(points[4], nulPoint) :
            if not np.allclose(points[4] - points[0], [0, h, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {4}")
                break
        else:
            if points[0][1] != vmax[1] :
                print(f"\nWrong '-1' assignment: {idx}, {4}")
                break

        if not np.array_equal(points[5], nulPoint) :
            if not np.allclose(points[5] - points[0], [0, 0, -h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {5}")
                break
        else:
            if points[0][2] != vmin[2] :
                print(f"\nWrong '-1' assignment: {idx}, {5}")
                break

        if not np.array_equal(points[6], nulPoint) :
            if not np.allclose(points[6] - points[0], [0, 0, h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {6}")
                break
        else:
            if points[0][2] != vmax[2] :
                print(f"\nWrong '-1' assignment: {idx}, {6}")
                break

        if not np.array_equal(points[7], nulPoint) :
            if not np.allclose(points[7] - points[0], [0, 0, 0, -h]) :
                print(f"\nMismach volume and neighbour: {idx}, {7}")
                break
        else:
            if points[0][3] != vmin[3] :
                print(f"\nWrong '-1' assignment: {idx}, {7}")
                break

        if not np.array_equal(points[8], nulPoint) :
            if not np.allclose(points[8] - points[0], [0, 0, 0, h]) :
                print(f"\nMismach volume and neighbour: {idx}, {8}")
                break
        else:
            if points[0][3] != vmax[3] :
                print(f"\nWrong '-1' assignment: {idx}, {8}")
                break

        if not np.array_equal(points[9], nulPoint) :
            if not np.allclose(points[9] - points[0], [-h, -h, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {9}")
                break
        else:
            if points[0][0] != vmin[0] and points[0][1] != vmin[1] :
                print(f"\nWrong '-1' assignment: {idx}, {9}")
                break

        if not np.array_equal(points[10], nulPoint) :
            if not np.allclose(points[10] - points[0], [-h, h, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {10}")
                break
        else:
            if points[0][0] != vmin[0] and points[0][1] != vmax[1] :
                print(f"\nWrong '-1' assignment: {idx}, {10}")
                break

        if not np.array_equal(points[11], nulPoint) :
            if not np.allclose(points[11] - points[0], [-h, 0, -h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {11}")
                break
        else:
            if points[0][0] != vmin[0] and points[0][2] != vmin[2] :
                print(f"\nWrong '-1' assignment: {idx}, {11}")
                break

        if not np.array_equal(points[12], nulPoint) :
            if not np.allclose(points[12] - points[0], [-h, 0, h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {12}")
                break
        else:
            if points[0][0] != vmin[0] and points[0][2] != vmax[2] :
                print(f"\nWrong '-1' assignment: {idx}, {12}")
                break

        if not np.array_equal(points[13], nulPoint) :
            if not np.allclose(points[13] - points[0], [-h, 0, 0, -h]) :
                print(f"\nMismach volume and neighbour: {idx}, {13}")
                break
        else:
            if points[0][0] != vmin[0] and points[0][3] != vmin[3] :
                print(f"\nWrong '-1' assignment: {idx}, {13}")
                break

        if not np.array_equal(points[14], nulPoint) :
            if not np.allclose(points[14] - points[0], [-h, 0, 0, h]) :
                print(f"\nMismach volume and neighbour: {idx}, {14}")
                break
        else:
            if points[0][0] != vmin[0] and points[0][3] != vmax[3] :
                print(f"\nWrong '-1' assignment: {idx}, {14}")
                break

        if not np.array_equal(points[15], nulPoint) :
            if not np.allclose(points[15] - points[0], [h, -h, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {15}")
                break
        else:
            if points[0][0] != vmax[0] and points[0][1] != vmin[1] :
                print(f"\nWrong '-1' assignment: {idx}, {15}")
                break

        if not np.array_equal(points[16], nulPoint) :
            if not np.allclose(points[16] - points[0], [h, h, 0, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {16}")
                break
        else:
            if points[0][0] != vmax[0] and points[0][1] != vmax[1] :
                print(f"\nWrong '-1' assignment: {idx}, {16}")
                break

        if not np.array_equal(points[17], nulPoint) :
            if not np.allclose(points[17] - points[0], [h, 0, -h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {17}")
                break
        else:
            if points[0][0] != vmax[0] and points[0][2] != vmin[2] :
                print(f"\nWrong '-1' assignment: {idx}, {17}")
                break

        if not np.array_equal(points[18], nulPoint) :
            if not np.allclose(points[18] - points[0], [h, 0, h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {18}")
                break
        else:
            if points[0][0] != vmax[0] and points[0][2] != vmax[2] :
                print(f"\nWrong '-1' assignment: {idx}, {18}")
                break

        if not np.array_equal(points[19], nulPoint) :
            if not np.allclose(points[19] - points[0], [h, 0, 0, -h]) :
                print(f"\nMismach volume and neighbour: {idx}, {19}")
                break
        else:
            if points[0][0] != vmax[0] and points[0][3] != vmin[3] :
                print(f"\nWrong '-1' assignment: {idx}, {19}")
                break

        if not np.array_equal(points[20], nulPoint) :
            if not np.allclose(points[20] - points[0], [h, 0, 0, h]) :
                print(f"\nMismach volume and neighbour: {idx}, {20}")
                break
        else:
            if points[0][0] != vmax[0] and points[0][3] != vmax[3] :
                print(f"\nWrong '-1' assignment: {idx}, {20}")
                break

        if not np.array_equal(points[21], nulPoint) :
            if not np.allclose(points[21] - points[0], [0, -h, -h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {21}")
                break
        else:
            if points[0][1] != vmin[1] and points[0][2] != vmin[2] :
                print(f"\nWrong '-1' assignment: {idx}, {21}")
                break

        if not np.array_equal(points[22], nulPoint) :
            if not np.allclose(points[22] - points[0], [0, -h, h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {22}")
                break
        else:
            if points[0][1] != vmin[1] and points[0][2] != vmax[2] :
                print(f"\nWrong '-1' assignment: {idx}, {22}")
                break

        if not np.array_equal(points[23], nulPoint) :
            if not np.allclose(points[23] - points[0], [0, -h, 0, -h]) :
                print(f"\nMismach volume and neighbour: {idx}, {23}")
                break
        else:
            if points[0][1] != vmin[1] and points[0][3] != vmin[3] :
                print(f"\nWrong '-1' assignment: {idx}, {23}")
                break

        if not np.array_equal(points[24], nulPoint) :
            if not np.allclose(points[24] - points[0], [0, -h, 0, h]) :
                print(f"\nMismach volume and neighbour: {idx}, {24}")
                break
        else:
            if points[0][1] != vmin[1] and points[0][3] != vmax[3] :
                print(f"\nWrong '-1' assignment: {idx}, {24}")
                break

        if not np.array_equal(points[25], nulPoint) :
            if not np.allclose(points[25] - points[0], [0, h, -h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {25}")
                break
        else:
            if points[0][1] != vmax[1] and points[0][2] != vmin[2] :
                print(f"\nWrong '-1' assignment: {idx}, {25}")
                break

        if not np.array_equal(points[26], nulPoint) :
            if not np.allclose(points[26] - points[0], [0, h, h, 0]) :
                print(f"\nMismach volume and neighbour: {idx}, {26}")
                break
        else:
            if points[0][1] != vmax[1] and points[0][2] != vmax[2] :
                print(f"\nWrong '-1' assignment: {idx}, {26}")
                break

        if not np.array_equal(points[27], nulPoint) :
            if not np.allclose(points[27] - points[0], [0, h, 0, -h]) :
                print(f"\nMismach volume and neighbour: {idx}, {27}")
                break
        else:
            if points[0][1] != vmax[1] and points[0][3] != vmin[3] :
                print(f"\nWrong '-1' assignment: {idx}, {27}")
                break

        if not np.array_equal(points[28], nulPoint) :
            if not np.allclose(points[28] - points[0], [0, h, 0, h]) :
                print(f"\nMismach volume and neighbour: {idx}, {28}")
                break
        else:
            if points[0][1] != vmax[1] and points[0][3] != vmax[3] :
                print(f"\nWrong '-1' assignment: {idx}, {28}")
                break

        if not np.array_equal(points[29], nulPoint) :
            if not np.allclose(points[29] - points[0], [0, 0, -h, -h]) :
                print(f"\nMismach volume and neighbour: {idx}, {29}")
                break
        else:
            if points[0][2] != vmin[2] and points[0][3] != vmin[3] :
                print(f"\nWrong '-1' assignment: {idx}, {29}")
                break

        if not np.array_equal(points[30], nulPoint) :
            if not np.allclose(points[30] - points[0], [0, 0, -h, h]) :
                print(f"\nMismach volume and neighbour: {idx}, {30}")
                break
        else:
            if points[0][2] != vmin[2] and points[0][3] != vmax[3] :
                print(f"\nWrong '-1' assignment: {idx}, {30}")
                break

        if not np.array_equal(points[31], nulPoint) :
            if not np.allclose(points[31] - points[0], [0, 0, h, -h]) :
                print(f"\nMismach volume and neighbour: {idx}, {31}")
                break
        else:
            if points[0][2] != vmax[2] and points[0][3] != vmin[3] :
                print(f"\nWrong '-1' assignment: {idx}, {31}")
                break

        if not np.array_equal(points[32], nulPoint) :
            if not np.allclose(points[32] - points[0], [0, 0, h, h]) :
                print(f"\nMismach volume and neighbour: {idx}, {32}")
                break
        else:
            if points[0][2] != vmax[2] and points[0][3] != vmax[3] :
                print(f"\nWrong '-1' assignment: {idx}, {32}")
                break
        # break
    print(f"\r{nPoints}/{nPoints}")

def main():
    file = path / (name+"_n.dat")

    with open(file) as f:
        nPoints = int(f.readline())
    neighbour = np.loadtxt(file, skiprows=2, dtype=np.int64)

    if nPoints != len(neighbour) :
        print(f"\nWrong number of points: {nPoints} != {len(neighbour)}")
    if neighbour.shape[1] != 33:
        print(f"\nWrong number of neighbours per points: {neighbour.shape[1]}")
    
    if np.any(neighbour >= nPoints) :
        print("Too big index")
    if np.any(neighbour < -1) :
        print("Too low index")

    test_surface(neighbour, nPoints)

if __name__ == "__main__":
    path = Path("../data")
    name = "test4_1"
    if len(argv) >= 2:
        name = argv[1]
    if len(argv) >= 3:
        path = Path(argv[2])
    print("Test Neighbour in ", path, " in the simulation ", name)

    main()
