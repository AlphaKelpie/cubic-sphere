"""
Create an inital rho distribution based on the radius from the center of the 
hypercube.
"""

import numpy as np

EXT = "base08"
VOLF = f"./data/{EXT}_s.dat"
PROF = f"./data/{EXT}_p.dat"
FILE = f"./data/{EXT}_init_r.dat"
print(EXT)

# with open(VOLF) as f:
#     nPoints = int(f.readline().split(' ')[0])
# vol = np.loadtxt(VOLF, skiprows=2)

# vol -= 2.
# rad = np.linalg.norm(vol, axis=1)

# # rho = np.where((rad > .9 and rad < 1.1), 1., 0.)
# # rho = np.zeros_like(rad)
# # rho[rad >= .9] = 1.
# # rho[rad >= 1.1] = 0.
# rho = np.copy(rad)
# rho[rho >= 2.] = 0.

# # rho = np.absolute(np.absolute(rad - 1) - 3)
# # totinv = 1. / np.sum(rho)
# # rho *= totinv

with open(PROF) as f:
    nPoints = int(f.readline().split(' ')[0])
print(nPoints)

proj = np.unique(np.loadtxt(PROF, skiprows=2, usecols=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)).astype(np.int64).flatten())
print(len(proj))

rho = np.zeros(nPoints)
rho[proj] = 1.

with open(FILE, 'w') as f:
    f.write(str(nPoints))
    f.write('\n\n')
    f.write('\n'.join(str(r) for r in rho))
