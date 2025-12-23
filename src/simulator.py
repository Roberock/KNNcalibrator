# src/simulator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union
import numpy as np

Array = np.ndarray

def _as_2d(x: Array, d: Optional[int] = None) -> Array:
    x = np.asarray(x)
    if x.ndim == 1:
        if d is not None and x.shape[0] != d:
            raise ValueError(f"Expected shape ({d},), got {x.shape}")
        return x[None, :]
    if x.ndim == 2:
        if d is not None and x.shape[1] != d:
            raise ValueError(f"Expected shape (n,{d}), got {x.shape}")
        return x
    raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")


def _broadcast_rows(*arrs: Array) -> Tuple[Array, ...]:
    """
    Broadcast rows so each input has the same leading dimension n.

    Allowed:
      - (1,d) with (n,d) -> repeats the single row
      - (n,d) with (n,d) -> ok

    Not allowed:
      - (m,d) with (n,d) when m != n and m != 1
    """
    n = max(a.shape[0] for a in arrs)
    out = []
    for a in arrs:
        if a.shape[0] == n:
            out.append(a)
        elif a.shape[0] == 1:
            out.append(np.repeat(a, n, axis=0))
        else:
            raise ValueError(f"Cannot broadcast rows: got {a.shape[0]} vs required {n}")
    return tuple(out)


# =========================
# Core simulator interface
# =========================
class Simulator(ABC):
    """
    Black-box simulator interface: (xa, xe, xc) -> y.

    - xa: aleatoric inputs  (n, d_a)
    - xe: epistemic inputs  (n, d_e)
    - xc: design variables  (n, d_c)
    - y : outputs           (n, d_y) or (n, T, d_y) for trajectories

    This interface is intentionally numpy-first.
    A NumPyro backend can be attached via `as_numpyro_model(...)`.
    """

    dim_xa: int
    dim_xe: int
    dim_xc: int
    dim_y: Optional[int]  # can be None if trajectory or variable-size outputs

    @abstractmethod
    def evaluate(self, xa: Array, xe: Array, xc: Array, **kwargs) -> Array:
        """(xa, xe, xc) -> y"""
        raise NotImplementedError

    def __call__(self, xa: Array, xe: Array, xc: Array, **kwargs) -> Array:
        return self.evaluate(xa=xa, xe=xe, xc=xc, **kwargs)

    def eval_batch(self, xa: Array, xe: Array, xc: Array, **kwargs) -> Array:
        """
        Like evaluate(), but enforces shapes and broadcasts rows.
        Input -> Output:
          xa: (n or 1, d_a), xe: (n or 1, d_e), xc: (n or 1, d_c) -> y: (n, ...)
        """
        xa = _as_2d(xa, self.dim_xa)
        xe = _as_2d(xe, self.dim_xe)
        xc = _as_2d(xc, self.dim_xc)
        xa, xe, xc = _broadcast_rows(xa, xe, xc)
        return self.evaluate(xa=xa, xe=xe, xc=xc, **kwargs)

    # ----------------------------
    # Optional NumPyro integration
    # ----------------------------
    def as_numpyro_model(
        self,
        obs_name: str = "y_obs",
        return_latent: bool = False,
    ) -> Callable[..., Any]:
        """
        Return a *callable* compatible with NumPyro model style.

        Input -> Output (callable):
          (xa, xe, xc, y_obs=None, **kwargs) -> None or dict of latent sites

        Notes:
          - This function does NOT import numpyro directly.
          - It returns a callable that can be used inside a numpyro model
            by the user (who will import numpyro and numpyro.distributions).
          - The user decides how to define observation noise and likelihood.
        """
        def _model(xa: Array, xe: Array, xc: Array, y_obs: Optional[Array] = None, **kwargs):
            """
            A thin bridge: computes simulator output and exposes it as a deterministic node.

            The user would typically do, inside this function (or in a wrapper around it):
              numpyro.deterministic("y_mean", y_mean)
              numpyro.sample(obs_name, dist.Normal(y_mean, sigma), obs=y_obs)
            """
            y_mean = self.eval_batch(xa=xa, xe=xe, xc=xc, **kwargs)
            if return_latent:
                return {"y_mean": y_mean, "y_obs": y_obs}
            return None

        return _model


# =========================
# Simple wrapper for callables
# =========================
@dataclass
class CallableSimulator(Simulator):
    """
    Wrap a python callable f(xa, xe, xc, **kwargs) -> y.

    The callable must accept:
      xa: (n,d_a), xe: (n,d_e), xc: (n,d_c)
    and return y with leading dimension n.
    """

    f: Callable[..., Array]
    dim_xa: int
    dim_xe: int
    dim_xc: int
    dim_y: Optional[int] = None

    def evaluate(self, xa: Array, xe: Array, xc: Array, **kwargs) -> Array:
        y = self.f(xa=xa, xe=xe, xc=xc, **kwargs)
        y = np.asarray(y)
        if y.shape[0] != xa.shape[0]:
            raise ValueError(f"Simulator output leading dim mismatch: y {y.shape} vs n={xa.shape[0]}")
        return y
