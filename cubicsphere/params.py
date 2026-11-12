"""Reader for the C++ ``params.dat`` format (``key value`` pairs).

Numbers are parsed like the C++ ``Params`` class: ``std::stof`` (rounded to
float32, then widened to double), so the grid, the crown mask and the default
time step are bit-for-bit the ones the C++ code uses. ``total`` keeps the
C++ quirk of adding 2 to the value in the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _stof(value: str) -> float:
    return float(np.float32(value))


@dataclass
class Params:
    path: str = "data/simulation"
    pathdata: str = ""
    T: float | None = None
    h: float | None = None
    deltaT: float = 0.0
    deltaC: float = 0.0
    mins: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    maxs: tuple[float, float, float, float] = (4.0, 4.0, 4.0, 4.0)
    center: tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0)
    radius: float | None = None
    total: int = 2002
    step: int = 100
    raw: dict = field(default_factory=dict)

    @property
    def crown(self) -> bool:
        """Spherical crown model (``deltaC != 0``) or whole hypercube."""
        return self.deltaC != 0.0

    @property
    def dt(self) -> float:
        """Time step: ``deltaT``, or the C++ default ``0.1 h^4`` when it is 0."""
        return self.deltaT if self.deltaT != 0.0 else 0.1 * self.h**4

    @property
    def alpha(self) -> float:
        """Crank-Nicolson half step factor ``(T/6) * dt / 2`` as in ``evolveCN``."""
        return (self.T / 6.0) * self.dt * 0.5


_AXES = ("w", "x", "y", "z")


def load_params(filename: str) -> Params:
    p = Params()
    mins = [0.0, 0.0, 0.0, 0.0]
    maxs = [4.0, 4.0, 4.0, 4.0]
    center = [2.0, 2.0, 2.0, 2.0]
    with open(filename) as fin:
        tokens = fin.read().split()
    for key, value in zip(tokens[0::2], tokens[1::2]):
        p.raw[key] = value
        if key == "path":
            p.path = value
        elif key == "pathdata":
            p.pathdata = value
        elif key in ("T", "h", "deltaT", "deltaC", "radius"):
            setattr(p, key, _stof(value))
        elif key.endswith("_min") and key[0] in _AXES:
            mins[_AXES.index(key[0])] = _stof(value)
        elif key.endswith("_max") and key[0] in _AXES:
            maxs[_AXES.index(key[0])] = _stof(value)
        elif key in ("cw", "cx", "cy", "cz"):
            center[_AXES.index(key[1])] = _stof(value)
        elif key == "total":
            p.total = int(value) + 2  # same as the C++ Params
        elif key == "step":
            p.step = int(value)
    p.mins, p.maxs, p.center = tuple(mins), tuple(maxs), tuple(center)
    for name in ("T", "h", "radius"):
        if getattr(p, name) is None:
            raise ValueError(f"'{name}' missing in {filename} (uninitialised in the C++ too)")
    return p
