import numpy as np

def paraboloid_model(theta,   # parameters to be calibrated
                     xi=0.0,  # the design variable
                     ):

    """Vectorized paraboloid, mild noise; supports scalar or vector xi."""
    A, B, C = 1.0, 0.5, 1.5  # other fixed known parametes
    theta = np.atleast_2d(theta).astype(float)
    x1, x2 = theta[:, 0], theta[:, 1]
    xi = np.asarray(xi, float)
    if xi.ndim == 0:
        xi = np.full(theta.shape[0], xi)
    elif xi.ndim == 2:
        xi = xi.ravel()
    y = A * x1**2 + B * x1 * x2 * (1.0 + xi) + C * (x2 + xi) ** 2
    y = y + 0.2 * np.random.randn(theta.shape[0])  # small noise
    return y.reshape(-1, 1) if theta.shape[0] > 1 else np.array([y.item()])


def theta_sampler(n, lb=-15, ub=15):
    return np.random.uniform(lb, ub, size=(n, 2))


def data_generation_mechanism(case = 1,
                              N_emp = None,
                              simulator =None,
                              sample_prior = None):

    rng = np.random.default_rng(42)

    if simulator is None:
        # Example simple simulator (replace with your expensive model)
        print('using default paraboloid model')
        def simulator(X, xi=0.0):
            return paraboloid_model(theta=X, xi=xi)

    if sample_prior is None:
        # Prior sampler
        print('using default uniform sampler for prior')
        def sample_prior(n):
            return theta_sampler(n=n)

    empirical_observations = []
    if case == 1:
        # ---------------------------------------------------------------------
        # CASE 1: 1 empirical sample, 1 true theta, 1 experiment design
        # ---------------------------------------------------------------------
        theta_true= np.array([[2.0, 3.0]])  # shape (1,2)
        xi_list = np.array([0.0])
        for xi in xi_list:
            y_emp = simulator(theta_true, xi)  # shape (1,1)
            empirical_observations.append((y_emp, xi))  # list of (y_emp, xi)

        print("CASE 1:",
              f"number of designs={len(empirical_observations)}, samples_per_design={empirical_observations[0][0].shape[0]}")
    elif case == 2:
        # ---------------------------------------------------------------------
        # CASE 2: 100 samples of theta (distributed), 1 experiment design
        # ---------------------------------------------------------------------
        if N_emp is None:
            N_emp = 100
        theta_true = rng.normal(4.0, 0.5, size=(N_emp, 2))  # shape (100,2)
        xi_list = np.array([0.0])

        empirical_observations = []
        for xi in xi_list:
            y_emp = simulator(theta_true, xi)  # shape (100,1)
            empirical_observations.append((y_emp, xi))

        print("CASE 2:",
              f"number of designs={len(empirical_observations)}, samples_per_design={empirical_observations[0][0].shape[0]}")
    elif case == 3:
        # ---------------------------------------------------------------------
        # CASE 3: 100 samples of theta (distributed), 4–5 different experiments
        # ---------------------------------------------------------------------
        if N_emp is None:
            N_emp = 100
        theta_true = rng.normal(4.0, 0.5, size=(N_emp, 2))  # shape (100,2)
        xi_list = np.array([-2.0, -1.0, 0.0, 2.0, 4.0])  # 5 designs

        empirical_observations = []
        for xi in xi_list:
            y_emp = simulator(theta_true, xi)  # shape (100,1)
            empirical_observations.append((y_emp, xi))

        print("CASE 3:",
              f"number of designs={len(empirical_observations)}, samples_per_design={empirical_observations[0][0].shape[0]}")
    else:

        print('case must be = 1 2 or 3.....')

    output = (empirical_observations, # observation-design pairs
              theta_true)  # true target

    return output