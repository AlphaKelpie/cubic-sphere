"""Reading/writing rho in the C++ point order.

* ``read_r_dat``: parses a C++ ``*_r.dat`` text file (``N\\n\\n`` then one
  value per line), caching the parsed array as .npy.
* ``to_points`` / ``from_points``: convert between the brick field on the GPU
  and a float32[N] array in C++ point order, in chunks.
* ``AsyncWriter``: writes snapshots in a background thread so the GPU keeps
  computing while the file is written.
"""

import hashlib
import os
import threading
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd

_CHUNK = 50_000_000


def read_r_dat(path, cache_dir, log=print):
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"rho_{os.path.basename(path)}_{key}.npy")
    if os.path.exists(cache):
        log(f"Loading cached {cache}")
        return np.load(cache)
    log(f"Parsing {path} (one-time, cached afterwards)...")
    t0 = time.time()
    with open(path) as f:
        n = int(f.readline().split()[0])
    vals = pd.read_csv(path, skiprows=2, header=None, dtype=np.float32, engine="c",
                       na_values=["-nan", "nan"], keep_default_na=False).iloc[:, 0].to_numpy()
    if len(vals) != n:
        raise RuntimeError(f"{path}: header says {n} values, found {len(vals)}")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache, vals)
    log(f"  {n} values in {time.time() - t0:.0f}s")
    return vals


def read_s_dat_header(path):
    with open(path) as f:
        tok = f.readline().split()
    return int(tok[0]), float(tok[1])


def to_points(field, slot):
    """float32[N] in C++ point order from a brick field (device)."""
    flat = field.reshape(-1)
    out = np.empty(len(slot), dtype=np.dtype(field.dtype))
    for s in range(0, len(slot), _CHUNK):
        idx = jnp.asarray(slot[s:s + _CHUNK])
        out[s:s + _CHUNK] = np.asarray(flat[idx])
    return out


def from_points(values, slot, op):
    """Brick field (device) from float[N] values in C++ point order."""
    flat = jnp.zeros(op.NBt * op.mask[0].size, op.dtype)
    for s in range(0, len(slot), _CHUNK):
        flat = flat.at[jnp.asarray(slot[s:s + _CHUNK])].set(
            jnp.asarray(values[s:s + _CHUNK], dtype=op.dtype))
    return flat.reshape(op.field_shape)


def write_text(path, values):
    """C++ ``saveRho`` text format (``N\\n\\n`` + one value per line, %g)."""
    with open(path, "w") as f:
        f.write(f"{len(values)}\n\n")
        for s in range(0, len(values), 10_000_000):
            np.savetxt(f, values[s:s + 10_000_000], fmt="%g")


class AsyncWriter:
    """One background writer thread; ``submit`` waits for the previous write."""

    def __init__(self, text=False, log=print):
        self.text = text
        self.log = log
        self._t = None

    def submit_field(self, prefix, field, slot):
        """Copy the brick field to host now; reorder to point order + write in the thread."""
        self.wait()
        host = np.asarray(field).reshape(-1)  # device -> host copy (blocks ~1 s)
        self.submit(prefix, None, lambda: host[slot])

    def submit(self, prefix, values, make_values=None):
        self.wait()

        def job():
            t0 = time.time()
            vals = make_values() if make_values is not None else values
            if self.text:
                write_text(prefix + "_r.dat", vals)
            else:
                np.save(prefix + "_r.npy", vals)
            self.log(f"    saved {os.path.basename(prefix)} ({time.time() - t0:.1f}s)")

        self._t = threading.Thread(target=job, daemon=False)
        self._t.start()

    def wait(self):
        if self._t is not None:
            self._t.join()
            self._t = None
