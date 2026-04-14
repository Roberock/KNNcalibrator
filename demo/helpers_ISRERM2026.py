import numpy as np
import matplotlib.pyplot as plt
from resources.loader_usecases import prepare_case, prepare_case_2_data
from scipy.stats import multivariate_normal

def sample_from_posterior_mixture(solver, n_samples=2000, random_state=123):
    """ Draw approximate posterior samples from the new solver. """
    return solver.posterior_particles(n_samples=n_samples, replace=True)


def posterior_mode_from_db_multidesign(solver):
    """ Return the highest-weight archive particle under the full posterior. """
    return solver.posterior_mode_from_db()


def plot_posterior_vs_true_theta_case1(  solver,  Demp,   xlim=(-15, 15),  ylim=(-15, 15),  gridsize=120,  n_levels=12, ):
    xx = np.linspace(xlim[0], xlim[1], gridsize)
    yy = np.linspace(ylim[0], ylim[1], gridsize)
    X1, X2 = np.meshgrid(xx, yy)
    grid = np.column_stack([X1.ravel(), X2.ravel()])

    Z = solver.posterior_pdf(grid).reshape(gridsize, gridsize)
    theta_true = np.vstack([np.asarray(Demp[k]["theta"], dtype=float) for k in Demp])

    zmin = np.min(Z)
    zmax = np.max(Z)
    levels = np.linspace(zmin + 0.05 * (zmax - zmin), zmax, n_levels)

    plt.figure(figsize=(7, 6))
    plt.contourf(X1, X2, Z, levels=20, alpha=0.25)
    cs = plt.contour(X1, X2, Z, levels=levels, linewidths=1.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.2e")

    plt.scatter(
        theta_true[:, 0],
        theta_true[:, 1],
        s=25, c='r', marker='+',
        alpha=0.85,
        label="True empirical theta",
    )

    mode_x, _ = solver.posterior_mode_from_db()
    plt.scatter(mode_x[0], mode_x[1], marker='x', s=120, linewidths=2.5, c='k', label='Posterior mode')

    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title("Posterior iso-density contours vs true empirical parameter samples")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_posterior_predictive_vs_empirical(
    solver,
    n_post_samples=2000,
    max_cols=3,
    random_state=123,
):
    X_post = sample_from_posterior_mixture(solver, n_samples=n_post_samples, random_state=random_state)

    designs = list(solver.designs)
    n_designs = len(designs)
    ncols = min(max_cols, n_designs)
    nrows = int(np.ceil(n_designs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, design in zip(axes, designs):
        Y_pred = np.asarray(solver.model(X_post, design), dtype=float)
        if Y_pred.ndim == 2 and Y_pred.shape[1] == 1:
            Y_pred = Y_pred[:, 0]

        Y_emp = np.asarray(solver.Y_emp_by_design[design], dtype=float)
        if Y_emp.ndim == 2 and Y_emp.shape[1] == 1:
            Y_emp = Y_emp[:, 0]

        ax.hist(Y_pred, bins=40, density=True, alpha=0.6, label="Posterior predictive")
        ax.hist(Y_emp, bins=20, density=True, alpha=0.6, label="Empirical")
        ax.set_title(f"Design = {design}")
        ax.set_xlabel("Response")
        ax.set_ylabel("Density")
        ax.legend()

    for ax in axes[n_designs:]:
        ax.axis("off")

    fig.suptitle("Posterior predictive vs empirical data", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_empirical_vs_posterior_intervals(
    solver,
    n_post_samples=3000,
    random_state=123,
):
    X_post = sample_from_posterior_mixture(solver, n_samples=n_post_samples, random_state=random_state)

    designs = list(solver.designs)
    emp_means, pred_means, pred_lo, pred_hi = [], [], [], []

    for design in designs:
        Y_pred = np.asarray(solver.model(X_post, design), dtype=float)
        if Y_pred.ndim == 2 and Y_pred.shape[1] == 1:
            Y_pred = Y_pred[:, 0]

        Y_emp = np.asarray(solver.Y_emp_by_design[design], dtype=float)
        if Y_emp.ndim == 2 and Y_emp.shape[1] == 1:
            Y_emp = Y_emp[:, 0]

        emp_means.append(np.mean(Y_emp))
        pred_means.append(np.mean(Y_pred))
        pred_lo.append(np.quantile(Y_pred, 0.05))
        pred_hi.append(np.quantile(Y_pred, 0.95))

    x = np.arange(len(designs))

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        x,
        pred_means,
        yerr=[np.array(pred_means) - np.array(pred_lo), np.array(pred_hi) - np.array(pred_means)],
        fmt="o",
        capsize=4,
        label="Posterior predictive mean ± 90% interval",
    )
    plt.scatter(x, emp_means, marker="x", s=100, linewidths=2.5, label="Empirical mean")
    plt.xticks(x, [str(d) for d in designs])
    plt.xlabel("Design")
    plt.ylabel("Response")
    plt.title("Empirical means vs posterior predictive intervals")
    plt.legend()
    plt.tight_layout()
    plt.show()



def posterior_pdf_by_design(solver, X, design):
    """  Evaluate a design-specific posterior-style surface.
    We combine the archive importance correction with the single design factor:
        p_e(theta) \propto [pi(theta)/q(theta)] * L_e(theta)
    This is not the full multi-design posterior; it is a design-wise view that is
    useful for diagnosis and visualization. """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    logpe = np.asarray(solver.prior.logpdf(X), dtype=float)
    logLe = solver.evaluate_design_factor_logpdf(X, design)
    out = np.exp(logpe + logLe - np.max(logpe + logLe))
    return out


def posterior_mode_by_design_from_db(solver, design):
    Xd = solver.X_db
    pd = posterior_pdf_by_design(solver, Xd, design)
    j = int(np.argmax(pd))
    return Xd[j], float(pd[j])


def plot_posterior_x_by_design_case1(solver,  xlim=(-15, 15),  ylim=(-15, 15),
                                     gridsize=120,  max_cols=3,   filled=True, ):
    xx = np.linspace(xlim[0], xlim[1], gridsize)
    yy = np.linspace(ylim[0], ylim[1], gridsize)
    X1, X2 = np.meshgrid(xx, yy)
    grid = np.column_stack([X1.ravel(), X2.ravel()])

    designs = list(solver.designs)
    n_designs = len(designs)
    ncols = min(max_cols, n_designs)
    nrows = int(np.ceil(n_designs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, design in zip(axes, designs):
        Z = posterior_pdf_by_design(solver, grid, design).reshape(gridsize, gridsize)
        mode_x, _ = posterior_mode_by_design_from_db(solver, design)

        if filled:
            cf = ax.contourf(X1, X2, Z, levels=20)
            fig.colorbar(cf, ax=ax, shrink=0.85, label="Relative density")
        else:
            cs = ax.contour(X1, X2, Z, levels=8)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2e")

        ax.scatter(mode_x[0], mode_x[1], marker="x", s=100, linewidths=2.5, label="Mode")
        ax.set_title(f"Design = {design}")
        ax.set_xlabel(r"$	heta_1$")
        ax.set_ylabel(r"$	heta_2$")
        ax.legend()

    for ax in axes[n_designs:]:
        ax.axis("off")

    fig.suptitle("Design-wise posterior views", y=1.02)
    plt.tight_layout()
    plt.show()



def adapt_case1_for_multidesign(M, Demp, Dsim):
    """  Convert case 1 into the shared-archive API used by the new class.
        Returns   -------
        model_with_design : callable  Signature model_with_design(X, design) -> Y
        Y_emp_by_design : dict  {design: empirical Y array}
        sim_db : dict   Shared archive with one X database and one Y database per design.
    """
    Y_emp_by_design = {
                    Demp[k]["xi"]: np.asarray(Demp[k]["y_data"], dtype=float).reshape(-1, 1)
                    for k in Demp }

    designs = [Dsim[k]["xi"] for k in Dsim]
    first_key = next(iter(Dsim))
    X_shared = np.asarray(Dsim[first_key]["theta"], dtype=float)

    # Sanity check: all design-specific entries should correspond to the same theta archive
    for k in Dsim:
        Xk = np.asarray(Dsim[k]["theta"], dtype=float)
        if Xk.shape != X_shared.shape or not np.allclose(Xk, X_shared):
            raise ValueError(
                "Case 1 demonstrator now assumes a shared theta archive across designs. "
                "The provided Dsim entries do not match."
            )

    sim_db = {"X": X_shared,
              "Y_by_design":
                    {Dsim[k]["xi"]: np.asarray(Dsim[k]["y_data"], dtype=float).reshape(-1, 1)
                     for k in Dsim
                     }
              }

    def model_with_design(X, design):
        return np.asarray(M(X, xi=design), dtype=float).reshape(-1, 1)

    return model_with_design, Y_emp_by_design, sim_db

def plot_posterior_vs_true_theta_by_design_case1(
    solver,
    Demp,
    xlim=(-15, 15),
    ylim=(-15, 15),
    gridsize=120,
    max_cols=3,
):
    xx = np.linspace(xlim[0], xlim[1], gridsize)
    yy = np.linspace(ylim[0], ylim[1], gridsize)
    X1, X2 = np.meshgrid(xx, yy)
    grid = np.column_stack([X1.ravel(), X2.ravel()])

    designs = list(solver.designs)
    n_designs = len(designs)
    ncols = min(max_cols, n_designs)
    nrows = int(np.ceil(n_designs / ncols))

    theta_by_design = {
        Demp[k]["xi"]: np.asarray(Demp[k]["theta"], dtype=float)
        for k in Demp
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, design in zip(axes, designs):
        Z = posterior_pdf_by_design(solver, grid, design).reshape(gridsize, gridsize)
        mode_x, _ = posterior_mode_by_design_from_db(solver, design)
        theta_true = theta_by_design[design]

        cs = ax.contour(X1, X2, Z, levels=8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2e")

        ax.scatter(theta_true[:, 0], theta_true[:, 1], s=10, alpha=0.30, label="True empirical theta")
        ax.scatter(mode_x[0], mode_x[1], marker="x", s=100, linewidths=2.5, label="Mode")

        ax.set_title(f"Design = {design}")
        ax.set_xlabel(r"$	heta_1$")
        ax.set_ylabel(r"$	heta_2$")
        ax.legend()

    for ax in axes[n_designs:]:
        ax.axis("off")

    fig.suptitle("Design-wise posterior views vs true empirical theta", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_airmode_posterior_style(
        solver,
        theta_tmcmc,
        theta_latex_names,
        Y_latex_names,
        pairs_2_plt=((6, 5), (3, 2), (5, 3), (1, 9)),
        n2plt=10_000,
        n_post=10_000,
        design="AIRMODE",
        random_state=123,
):
    X_sim = np.asarray(solver.X_db, dtype=float)
    Y_sim = np.asarray(solver.Y_db_by_design[design], dtype=float)
    Y_emp = np.asarray(solver.Y_emp_by_design[design], dtype=float)

    X_post = sample_from_posterior_mixture(
        solver,
        n_samples=min(n_post, max(1000, n2plt)),
        random_state=random_state,
    )
    Y_post = np.asarray(solver.model(X_post, design), dtype=float)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    axs = ax.flatten()

    for pairs, axi in zip(pairs_2_plt, axs):
        i, j = pairs
        axi.scatter(X_sim[:n2plt, i], X_sim[:n2plt, j], 10, c="b", alpha=0.25, label="sim")
        axi.scatter(X_post[:n2plt, i], X_post[:n2plt, j], 4, c="k", alpha=0.20, label="posterior")
        axi.scatter(theta_tmcmc[:n2plt, i], theta_tmcmc[:n2plt, j], 6, c="r", marker="+", label="TMCMC ref")
        axi.set_xlabel(theta_latex_names[i])
        axi.set_ylabel(theta_latex_names[j])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    axs = ax.flatten()

    for pairs, axi in zip(pairs_2_plt, axs):
        i, j = pairs
        axi.scatter(Y_sim[:n2plt, i], Y_sim[:n2plt, j], 10, c="b", alpha=0.25, label="sim")
        axi.scatter(Y_post[:n2plt, i], Y_post[:n2plt, j], 4, c="k", alpha=0.20, label="posterior predictive")
        axi.scatter(Y_emp[:, i], Y_emp[:, j], 10, c="r", marker="+", label="emp")
        axi.set_xlabel(Y_latex_names[i])
        axi.set_ylabel(Y_latex_names[j])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return X_post, Y_post


def sliced_wasserstein_2(X, Y, n_proj=200, random_state=123):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.shape[1] != Y.shape[1]:
        if X.shape[0] == Y.shape[1]:
            X = X.T
        elif Y.shape[0] == X.shape[1]:
            Y = Y.T
        else:
            raise ValueError(f"Incompatible shapes: X={X.shape}, Y={Y.shape}")

    n = min(len(X), len(Y))
    if len(X) > n:
        X = X[rng.choice(len(X), size=n, replace=False)]
    if len(Y) > n:
        Y = Y[rng.choice(len(Y), size=n, replace=False)]

    d = X.shape[1]
    vals = []
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v) + 1e-12
        xp = np.sort(X @ v)
        yp = np.sort(Y @ v)
        vals.append(np.mean((xp - yp) ** 2))
    return float(np.mean(vals))

def prepare_airmode_for_calibration(Nemp=200, Nsim=2000):
    """  Prepare AIRMODE case study for the shared-archive API.  """
    M, Demp, Dsim = prepare_case(2, Nemp=Nemp, Nsim=Nsim)
    def model_design(X, design):
        return np.asarray(M(X), dtype=float)
    Y_emp_by_design = {   "AIRMODE": np.asarray(Demp["y_data"], dtype=float) }
    X_sim = np.asarray(Dsim["theta"], dtype=float)
    Y_sim = np.asarray(Dsim["y_data"], dtype=float)
    sim_db = {  "X": X_sim,    "Y_by_design": {"AIRMODE": Y_sim},  }
    d_x = X_sim.shape[1]
    prior = multivariate_normal(mean=np.zeros(d_x), cov=np.eye(d_x))
    return model_design, Y_emp_by_design, sim_db, prior


def prepare_airmode_for_calibration(Nemp=20, Nsim=5000):
    """ Prepare AIRMODE case study for the shared-archive API. """
    M, Demp, Dsim = prepare_case_2_data(
        Nemp=Nemp,  Nsim=Nsim,  sim_data_path="../resources/AIRMODE/airmod_io_repo_style_500k.csv",
        emp_data_path="../resources/AIRMODE/data/DLRAirmodData.mat",  model_path="../resources/AIRMODE/airmode_surrogate.pt",
        scaler_path="../resources/AIRMODE/airmode_y_scaler.pkl", )

    def model_design(X, design):
        return np.asarray(M(X), dtype=float)

    if isinstance(Demp, dict) and "y_data" in Demp:
        Y_emp = np.asarray(Demp["y_data"], dtype=float)
    elif isinstance(Demp, dict) and 0 in Demp and "y_data" in Demp[0]:
        Y_emp = np.asarray(Demp[0]["y_data"], dtype=float)
    else:
        raise ValueError(f"Unsupported Demp format. Keys: {list(Demp.keys())}")

    if isinstance(Dsim, dict) and "theta" in Dsim and "y_data" in Dsim:
        X_sim = np.asarray(Dsim["theta"], dtype=float)
        Y_sim = np.asarray(Dsim["y_data"], dtype=float)
    elif isinstance(Dsim, dict) and 0 in Dsim and "theta" in Dsim[0] and "y_data" in Dsim[0]:
        X_sim = np.asarray(Dsim[0]["theta"], dtype=float)
        Y_sim = np.asarray(Dsim[0]["y_data"], dtype=float)
    else:
        raise ValueError(f"Unsupported Dsim format. Keys: {list(Dsim.keys())}")
    Y_emp_by_design = {"AIRMODE": Y_emp}
    sim_db = {  "X": X_sim,  "Y_by_design": {"AIRMODE": Y_sim},  }
    d_x = X_sim.shape[1]
    prior = multivariate_normal(mean=np.zeros(d_x), cov=np.eye(d_x))
    return model_design, Y_emp_by_design, sim_db, prior