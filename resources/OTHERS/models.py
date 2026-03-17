import numpy as np

rng = np.random.default_rng(42)

# ------------------------ MODEL 1 --------------------------
# ----------------- Simple paraboloid mode ------------------
# -----------------------------------------------------------
def paraboloid_model(theta, xi=0.0, A=1.0, B=0.5, C=1.5):
    """Vectorized paraboloid, mild noise; supports scalar or vector xi."""
    theta = np.atleast_2d(theta).astype(float)
    x1, x2 = theta[:, 0], theta[:, 1]
    xi = np.asarray(xi, float)
    if xi.ndim == 0:
        xi = np.full(theta.shape[0], xi)
    elif xi.ndim == 2:
        xi = xi.ravel()
    y = A * x1**2 + B * x1 * x2 * (1.0 + xi) + C * (x2 + xi) ** 2
    return y.reshape(-1, 1) if theta.shape[0] > 1 else np.array([y.item()])

# Target uncertainty & sampler
def true_target_sampler_paraboloid(N=50):
    mu1 = np.array([4.0, 4.0])
    C1  = np.array([[0.20, 0.10],
                    [0.10, 0.25]])
    return rng.multivariate_normal(mu1, C1, size=N)

# to run the experiment (empirical data generator)
def paraboloid_DGM(N=100, xi=0.0, A=1.0, B=0.5, C=1.5):
    """Vectorized paraboloid, mild noise; supports scalar or vector xi."""

    # n number of samples from tue_target_sampler
    theta = true_target_sampler_paraboloid(N)

    theta = np.atleast_2d(theta).astype(float)
    x1, x2 = theta[:, 0], theta[:, 1]
    xi = np.asarray(xi, float)
    if xi.ndim == 0:
        xi = np.full(theta.shape[0], xi)
    elif xi.ndim == 2:
        xi = xi.ravel()
    y = A * x1**2 + B * x1 * x2 * (1.0 + xi) + C * (x2 + xi) ** 2
    return y.reshape(-1, 1) if theta.shape[0] > 1 else np.array([y.item()])





# ------------------------ MODEL 2 --------------------------
# -------- Interesting simulator with ridges & switches -----
# -----------------------------------------------------------
def ridge_switch_model(theta_cloud, xi=1.0, ripple=0.35):
    """  theta_cloud: (N, 2) with columns (theta1, theta2)
        xi: scalar design variable
        Returns y_emp: (N, 3) with features [y1, y2, y3] """
    th1, th2 = theta_cloud[:, 0],  theta_cloud[:, 1]
    xi = float(xi)
    alpha = 0.6 * np.tanh(0.2 * xi) # xi-dependent shift and rotation -> level-sets twist with xi
    beta  = 0.4 * np.sin(0.5 * xi)
    c1 = 0.8 * np.sin(0.3 * xi) # shift center in theta-space as xi changes
    c2 = 0.8 * np.cos(0.2 * xi)
    u =  (th1 - c1) * np.cos(beta) - (th2 - c2) * np.sin(beta) + alpha * xi # rotated/shifted coordinates (xi-dependent)
    v =  (th1 - c1) * np.sin(beta) + (th2 - c2) * np.cos(beta) - 0.3 * xi
    r = np.sqrt(u**2 + (1.6 * v)**2) # radius (drives many-to-one mapping)
    # ----- regime switch in mapping (piecewise physics) -----
    # For small |xi|, emphasize r^2; for large |xi|, emphasize |u| and v^2 differently
    if np.abs(xi) < 2.0:
        base = r**2
        twist = ripple * np.sin(3.0 * np.arctan2(1.6*v, u))   # ripple along angle
    else:
        base = 0.6 * r**2 + 0.9 * np.abs(u) + 0.3 * v**2
        twist = ripple * np.cos(2.0 * np.arctan2(1.6*v, u) + 0.4*xi)

    # Outputs:
    y1 = base + twist # y1: curved with angular ripple
    y2 = (th1 - 1.2*th2 + 0.3*np.sign(xi))**2 + 0.05 * xi**2 # y2: anisotropy-sensitive difference square + mild xi^2 trend
    z = 0.9*(0.8*th1 - 0.6*th2) + 0.25*np.tanh(0.3*xi) - 0.15*base**0.5
    y3 = 1.0 / (1.0 + np.exp(-z))  # # y3: bounded "rate" via sigmoid of a xi-dependent linear form in (0,1)
    Y = np.stack([y1, y2, y3], axis=1)

    # ----- heteroscedastic noise (grows with r and |xi|) -----
    sigma = 0.05 + 0.02*np.abs(xi) + 0.03*r # scale per-output differently
    eps = np.column_stack([ rng.normal(0, 1.0 * sigma),  rng.normal(0, 0.6 * sigma),  rng.normal(0, 0.15 * sigma)])

    return Y + eps



# Target uncertainty & sampler
def true_target_sampler_ridge_switch_model(N=100, w=0.2):
    n1 = int(np.round(w * N))
    n2 = N - n1

    # Mode A: tight, positively correlated
    mu1 = np.array([1.5, -0.5])
    C1  = np.array([[0.20, 0.85],
                    [0.85, 0.30]])
    th1 = rng.multivariate_normal(mu1, C1, size=n1)
    # Mode B: wider, negatively correlated
    mu2 = np.array([-1.0, 1.2])
    C2  = np.array([[0.60, -0.85],
                    [-0.85, 0.40]])
    th2 = rng.multivariate_normal(mu2, C2, size=n2)
    return np.vstack([th1, th2])


# to run the experiment (empirical data generator)
def ridge_switch_DGM(N=100, xi=0.0):
    # n number of samples from tue_target_sampler
    theta_cloud = true_target_sampler_ridge_switch_model(N)
    return  ridge_switch_model(theta_cloud, xi=xi, ripple=0.35)

