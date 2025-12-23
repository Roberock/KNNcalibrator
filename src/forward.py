# src/forward.py
import numpy as np
from typing import Any, Callable, Dict, Tuple, Optional

Array = np.ndarray


"""def forward_propagation(pdf_theta, simulator, xi, n_samples):

    # sample parameters
    try:
        samples = pdf_theta.sample(n_samples)   # shape (n_samples, d_theta)
    except:
        try:
            samples = pdf_theta # passed as samples
        except:
            print('pass pdf object with .sample() method or pass samples to be propagated directly')
    # propagate through simulator
    y = np.array([simulator(theta, xi) for theta in samples])
    pdf_y = np.ravel(y)                                # returned as samples of p(y|xi)
    return pdf_y, {"theta": samples, "y":y}


"""



def ensure_2d_y(y: Array) -> Array:
    """Force y to (N, d_y) no matter what the simulator returns."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2:
        return y
    # common accidental shapes: (N,1,1) or (N,T,1) -> flatten feature dims
    return y.reshape(y.shape[0], -1)


def forward_propagation(
    pdf_theta: Any,                      # object with .sample(n) OR array of samples
    simulator: Callable[..., Array],         # should accept (theta, xi) in *batch*
    xi: Any,
    n_samples: int,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Array, Dict[str, Array]]:
    """
    Input -> Output:
      pdf_theta, simulator, xi, n_samples -> (y_samples, samples_dict)

    Returns:
      y_samples: (N, d_y)
      samples: {"theta": (N,d_theta), "y": (N,d_y)}
    """
    if rng is None:
        rng = np.random.default_rng()

    # get theta samples
    if hasattr(pdf_theta, "sample"):
        theta = np.asarray(pdf_theta.sample(n_samples), float)
    else:
        theta = np.asarray(pdf_theta, float)
        if theta.ndim == 1:
            theta = theta[None, :]

    # batched simulator call (preferred)
    y = simulator(theta, xi)
    y = ensure_2d_y(y)

    if y.shape[0] != theta.shape[0]:
        raise ValueError(f"Simulator returned {y.shape[0]} rows but theta has {theta.shape[0]}")

    return y, {"theta": theta, "y": y}
