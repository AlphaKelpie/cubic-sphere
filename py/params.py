"""Parser for the cubic-sphere ``params*.dat`` files.

Mirrors ``params.hpp``: every numeric value goes through ``std::stof`` (i.e. it
is rounded to float32) and is then stored in a double. ``total`` gets +2.
"""

from dataclasses import dataclass, field

import numpy as np


def _stof(value: str) -> float:
    return float(np.float32(float(value)))


@dataclass
class Params:
    path: str = ""
    pathdata: str = ""
    T: float = 0.0
    h: float = 0.0
    deltaT: float = 0.0
    deltaC: float = 0.0
    w: tuple = (0.0, 4.0)
    x: tuple = (0.0, 4.0)
    y: tuple = (0.0, 4.0)
    z: tuple = (0.0, 4.0)
    radius: float = 0.0
    center: tuple = (2.0, 2.0, 2.0, 2.0)
    total: int = 2002
    step: int = 100
    source: str = field(default="", repr=False)

    @property
    def bounds(self):
        return (self.w, self.x, self.y, self.z)

    @property
    def crown(self) -> bool:
        """Spherical-crown model (deltaC != 0) vs whole hypercube."""
        return self.deltaC != 0.0

    @classmethod
    def load(cls, filename: str) -> "Params":
        raw = {}
        with open(filename) as fin:
            tokens = fin.read().split()
        for key, value in zip(tokens[0::2], tokens[1::2]):
            raw[key] = value

        p = cls(source=filename)
        p.path = raw.get("path", "")
        p.pathdata = raw.get("pathdata", "")
        for key in ("T", "h", "deltaT", "deltaC", "radius"):
            if key in raw:
                setattr(p, key, _stof(raw[key]))
        lim = {}
        for ax in "wxyz":
            lim[ax] = (_stof(raw.get(f"{ax}_min", "0.")), _stof(raw.get(f"{ax}_max", "4.")))
        p.w, p.x, p.y, p.z = lim["w"], lim["x"], lim["y"], lim["z"]
        p.center = tuple(_stof(raw.get(f"c{ax}", "2.")) for ax in "wxyz")
        if "total" in raw:
            p.total = int(raw["total"]) + 2
        if "step" in raw:
            p.step = int(raw["step"])
        return p
