"""
Create an inital rho distribution based on the radius from the center of the 
hypercube.
"""

import numpy as np

PATH = "./data/base16_s.dat"
FILE = "./data/base16_init_r.dat"

with open(PATH) as f:
    nPoints = int(f.readline().split(' ')[0])
vol = np.loadtxt(PATH, skiprows=2)

vol -= 2.
rad = np.linalg.norm(vol, axis=1)

# rho = np.where((rad > .9 and rad < 1.1), 1., 0.)
# rho = np.zeros_like(rad)
# rho[rad >= .9] = 1.
# rho[rad >= 1.1] = 0.
rho = np.copy(rad)
rho[rho >= 2.] = 0.

# rho = np.absolute(np.absolute(rad - 1) - 3)
# totinv = 1. / np.sum(rho)
# rho *= totinv

with open(FILE, 'w') as f:
    f.write(str(nPoints))
    f.write('\n\n')
    f.write('\n'.join(str(r) for r in rho))
