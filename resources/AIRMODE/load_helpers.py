import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
from knn import estimate_p_theta_knn
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors





# --------------------------- helpers ---------------------------
def load_mat_auto(path: str):
    """ Load .mat (v7 via scipy; v7.3 via h5py) -> dict of numpy arrays.
        Transpose 2D arrays that look like (features, N) to (N, features). """
    path = str(path)
    try:
        m = loadmat(path)
        out = {k: v for k, v in m.items() if not k.startswith("__")}
        for k, v in list(out.items()):   # heuristic transpose: small #rows (≤20) and more columns than rows
            a = np.array(v)
            out[k] = a.T if a.ndim == 2 and a.shape[0] <= 20 and a.shape[1] > a.shape[0] else a
        return out
    except NotImplementedError:
        # v7.3 (HDF5)
        out = {}
        with h5py.File(path, "r") as f:
            def read_obj(obj):
                a = np.array(obj)
                if a.ndim == 2 and a.shape[0] <= 20 and a.shape[1] > a.shape[0]:
                    a = a.T
                return a
            for k in f.keys():
                out[k] = read_obj(f[k])
        return out


# -------------------- robust MAT loader (v7 / v7.3) --------------------
def load_mat_any(path: str):
    """Return a Python object tree from .mat (v7 via scipy, v7.3 via h5py)."""
    try:
        # v7 / v7.2
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # v7.3 (HDF5): expose raw h5py; we’ll traverse groups/datasets
        return h5py.File(path, "r")


def _is_ndarray_num(x):
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)

def _unwrap_scipy_struct(x):
    """Convert scipy loadmat structs/cells to pure Python containers."""
    # MATLAB struct becomes numpy.void or object with __dict__/dtype.names
    if hasattr(x, '__dict__') and len(getattr(x, '__dict__', {})) > 0:
        return {k: _unwrap_scipy_struct(v) for k, v in x.__dict__.items()}
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [_unwrap_scipy_struct(v) for v in x.ravel()]
    if isinstance(x, np.ndarray) and x.dtype.names:
        # structured array -> dict of fields
        d = {}
        item = x  # could be array of structs or single struct
        if item.ndim == 0:
            item = np.array([item])
        for field in x.dtype.names:
            d[field] = _unwrap_scipy_struct(x[field])
        return d
    if isinstance(x, np.ndarray):
        return x
    return x

def _iter_h5(obj, path=""):
    """Yield (path, array) for 2D numeric datasets in h5py trees."""
    if isinstance(obj, h5py.Dataset):
        arr = np.array(obj)
        yield path, arr
    elif isinstance(obj, h5py.Group):
        for k in obj.keys():
            yield from _iter_h5(obj[k], path + "/" + k)

def _iter_arrays_py(obj, path="root"):
    """Recursively yield (path, array) for 2D numeric arrays in Python containers."""
    if _is_ndarray_num(obj) and obj.ndim == 2:
        yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_arrays_py(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_arrays_py(v, f"{path}[{i}]")

def find_theta_matrix_from_mat(matobj):
    """ Search matobj for a 2D numeric array with 11 (or 12) cols/rows.
    returns (theta_(n,11), path_str). If 12 found, drop last as sigma.
    """
    cands = []
    if not isinstance(matobj, h5py.File): # SciPy tree
        tree = {k: _unwrap_scipy_struct(v) for k, v in matobj.items() if not k.startswith("__")}
        for p, arr in _iter_arrays_py(tree):
            r, c = arr.shape
            if 11 in (r, c) or 12 in (r, c):
                cands.append((p, arr))
    else:
        for p, arr in _iter_h5(matobj): # HDF5 tree
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                r, c = arr.shape
                if 11 in (r, c) or 12 in (r, c):
                    cands.append((p, arr))

    if not cands:
        raise KeyError("No 11/12-D arrays found anywhere in TMCMC file.")

    def norm(arr):  # pick the biggest candidate by sample count
        r, c = arr.shape
        if c in (11, 12):  # orient to (n, d)
            n = r
        elif r in (11, 12):
            n = c
        else:
            n = max(r, c)
        return n

    p_best, A = max(cands, key=lambda pc: norm(pc[1]))
    r, c = A.shape
    if c in (11, 12):  # orient to (n, d)
        Theta = np.array(A)  # (n, d)
    elif r in (11, 12):
        Theta = np.array(A).T
    else: # shouldn’t happen due to filter
        Theta = np.array(A)
    if Theta.shape[1] == 12: # drop sigma if present
        Theta = Theta[:, :11]
    return Theta.astype(np.float32), p_best


