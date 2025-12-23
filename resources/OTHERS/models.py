import numpy as np


def paraboloid_model(theta, xi=0.0):
    """
    Vectorized paraboloid model

    theta = (xa1, xa2, xe1, xe2)
    xa : aleatoric variables
    xe : epistemic parameters
    xi : design variable (scalar or array)
    """
    theta = np.atleast_2d(theta).astype(float)
    n = theta.shape[0]

    A = 1.0
    B = 0.5
    C = 1.5

    # Aleatoric and epistemic
    (xa1, xa2, xe1 , xe2) = (theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3])

    # xe1 ~ 2.0
    # xe2 ~ 0.0
    # xa1,xa2 ~ N([4.0,4.0],[std 0.5 and cov 0.9])

    # Design variable
    xi = np.asarray(xi, float)
    if xi.ndim == 0:
        xi = np.full(n, xi)
    else:
        xi = xi.ravel()
        if xi.size != n:
            raise ValueError("xi must be scalar or have same length as theta")

    # Model
    y = (
        A * xa1 ** 2 +
        B * xa1 * xa2 * (1.0 + xi) +
        C * (xa2 + xi) ** 2
    )

    # Small observation noise
    #y += xe1/10 * np.random.randn(n) + xe2

    return y.reshape(-1, 1)
