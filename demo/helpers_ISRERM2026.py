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


def plot_posterior_vs_true_theta_case1(solver,  Demp,   xlim=(-15, 15),  ylim=(-15, 15),  gridsize=120,  n_levels=12, ):
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
        plot_sim=True,
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

        if plot_sim:
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
        if plot_sim:
            axi.scatter(Y_sim[:n2plt, i], Y_sim[:n2plt, j], 10, c="b", alpha=0.25, label="sim")
        axi.scatter(Y_post[:n2plt, i], Y_post[:n2plt, j], 4, c="k", alpha=0.20, label="posterior predictive")
        axi.scatter(Y_emp[:, i], Y_emp[:, j], 20, c="r", marker="+", label="emp")
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

    #todo: temporary patch weird data
    X0 = [list(0.3 * np.ones((1, 11))[0]) + [-0.6] for k in range(Nemp)]
    Demp_sim = M(X0 * (1 + np.random.normal(0, 0.1, size=np.shape(X0))))
    Demp[0]['y_data'] = Demp_sim

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


def _add_density_contours(ax, X, color, levels=4, linewidths=1.4, alpha=0.9, zorder=3):
    """
    Add 2D KDE contours to an axis.
    X must be (n, 2).
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 20:
        return

    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)

    # small padding
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= 0.08 * max(dx, 1e-8)
    xmax += 0.08 * max(dx, 1e-8)
    ymin -= 0.08 * max(dy, 1e-8)
    ymax += 0.08 * max(dy, 1e-8)

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 120),
        np.linspace(ymin, ymax, 120)
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kde = gaussian_kde(X.T)
        zz = kde(grid).reshape(xx.shape)

        # use quantile-like contour levels
        zmax = zz.max()
        if zmax <= 0:
            return
        levs = np.linspace(0.05, 0.95, levels) * zmax

        ax.contour(
            xx, yy, zz,
            levels=levs,
            colors=color,
            linewidths=linewidths,
            alpha=alpha,
            zorder=zorder,
        )
    except Exception:
        pass


def _style_axis(ax):
    ax.set_facecolor("#fafafa")
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#bfbfbf")
        spine.set_linewidth(0.8)


def _panel_label(ax, txt):
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9)
    )


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def add_cov_ellipse(ax, X, n_std=2.0, edgecolor="#1d4ed8", label=None):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    cov = np.cov(X[:, 0], X[:, 1])

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)

    ell = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        lw=2.0,
        ls="-",
        edgecolor=edgecolor,
        alpha=0.9,
        label=label,
        zorder=5,
    )
    ax.add_patch(ell)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


def _add_density_contours(
    ax,
    samples_2d,
    color="k",
    levels=4,
    zorder=3,
    linewidths=1.2,
    grid_size=250,
    pad_frac=0.20,
    enclose_all=False,
    outer_level_factor=0.98,
):
    """
    Add KDE contours to a 2D scatter plot.

    If enclose_all=True, the outermost KDE contour is chosen so that it
    encloses all samples in samples_2d, up to grid resolution.
    """

    samples_2d = np.asarray(samples_2d, dtype=float)
    samples_2d = samples_2d[np.isfinite(samples_2d).all(axis=1)]

    if samples_2d.shape[0] < 5:
        return None

    x = samples_2d[:, 0]
    y = samples_2d[:, 1]

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None

    try:
        kde = gaussian_kde(samples_2d.T)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(123)
        scale = np.maximum(np.std(samples_2d, axis=0), 1.0)
        jittered = samples_2d + rng.normal(
            0, 1e-9, size=samples_2d.shape
        ) * scale
        kde = gaussian_kde(jittered.T)

    sample_density = kde(samples_2d.T)

    if enclose_all:
        outer_level = np.min(sample_density) * outer_level_factor

    dx = np.ptp(x)
    dy = np.ptp(y)

    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    pad = pad_frac

    for _ in range(8):
        xmin = x.min() - pad * dx
        xmax = x.max() + pad * dx
        ymin = y.min() - pad * dy
        ymax = y.max() + pad * dy

        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, grid_size),
            np.linspace(ymin, ymax, grid_size),
        )

        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        if not enclose_all:
            break

        edge_values = np.concatenate([
            zz[0, :],
            zz[-1, :],
            zz[:, 0],
            zz[:, -1],
        ])

        # Ensures the outer contour is not clipped by the plot boundary.
        if np.max(edge_values) < outer_level:
            break

        pad *= 1.6

    if enclose_all:
        if levels <= 1:
            contour_levels = np.array([outer_level])
        else:
            inner_levels = np.quantile(
                sample_density,
                np.linspace(0.25, 0.90, levels - 1)
            )

            contour_levels = np.concatenate([
                [outer_level],
                inner_levels,
            ])
    else:
        contour_levels = np.quantile(
            sample_density,
            np.linspace(0.20, 0.90, levels)
        )

    zmin = np.nanmin(zz)
    zmax = np.nanmax(zz)

    contour_levels = np.asarray(contour_levels)
    contour_levels = contour_levels[
        (contour_levels > zmin) & (contour_levels < zmax)
    ]

    contour_levels = np.unique(contour_levels)

    if len(contour_levels) == 0:
        return None

    return ax.contour(
        xx,
        yy,
        zz,
        levels=contour_levels,
        colors=color,
        linewidths=linewidths,
        zorder=zorder,
    )


def plot_airmode_comparison_fancy(
        solver,
        theta_tmcmc,
        design="AIRMODE",
        plot_sim=False,
        n2plt=1000,
        pairs_theta=((1, 5), (3, 2), (5, 3), (1, 9)),
        pairs_y=((1, 5), (3, 2), (5, 3), (1, 9)),
        theta_latex_names=None,
        Y_latex_names=None,
        posterior_label="Posterior",
        ref_label="TMCMC reference",
        predictive_label="Posterior predictive",
        empirical_label="Empirical",
        sim_label="Archive",
):
    # ------------------------------------------------------------
    # labels
    if theta_latex_names is None:
        theta_latex_names = [
            r"$\theta_1$", r"$\theta_2$", r"$\theta_3$",
            r"$\theta_4$", r"$\theta_5$", r"$\theta_6$",
            r"$\theta_7$", r"$\theta_8$", r"$\theta_9$",
            r"$\theta_{10}$", r"$\theta_{11}$"
        ]

    if Y_latex_names is None:
        Y_latex_names = [
            r"$D_1$", r"$D_2$", r"$D_3$", r"$D_4$", r"$D_5$",
            r"$D_6$", r"$D_7$", r"$D_8$", r"$D_9$", r"$D_{10}$"
        ]

    # ------------------------------------------------------------
    # data
    X_sim = np.asarray(solver.X_db, dtype=float)
    Y_sim = np.asarray(solver.Y_db_by_design[design], dtype=float)
    Y_emp = np.asarray(solver.Y_emp_by_design[design], dtype=float)

    X_post = solver.sample_posterior_particles_smooth(n_samples=n2plt)
    X_post = np.asarray(X_post, dtype=float)

    Y_post = np.asarray(solver.model(X_post, design), dtype=float)

    theta_tmcmc = np.asarray(theta_tmcmc, dtype=float)

    d_cmp = min(X_post.shape[1], theta_tmcmc.shape[1])

    # ------------------------------------------------------------
    # colors
    c_post = "#1f1f1f"  # black-ish
    c_ref = "#d62728"   # red
    c_emp = "#d62728"
    c_sim = "#1f77b4"   # blue

    # ------------------------------------------------------------
    # FIGURE 1: parameter space
    ncols = 2
    nrows = int(np.ceil(len(pairs_theta) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows))
    axs = np.asarray(axs).flatten()

    for p, (i, j) in enumerate(pairs_theta):
        ax = axs[p]
        _style_axis(ax)

        if i >= d_cmp or j >= d_cmp:
            ax.axis("off")
            continue

        # Optional archive cloud
        if plot_sim:
            ax.scatter(
                X_sim[:n2plt, i],
                X_sim[:n2plt, j],
                s=10,
                c=c_sim,
                alpha=0.12,
                edgecolors="none",
                zorder=1,
            )

        # Posterior samples
        ax.scatter(
            X_post[:n2plt, i],
            X_post[:n2plt, j],
            s=14,
            c=c_post,
            alpha=0.18,
            edgecolors="none",
            zorder=2,
        )

        # TMCMC reference
        ax.scatter(
            theta_tmcmc[:n2plt, i],
            theta_tmcmc[:n2plt, j],
            s=28,
            c=c_ref,
            alpha=0.75,
            marker="+",
            linewidths=1.0,
            zorder=4,
        )

        # Posterior KDE contours.
        # The outermost contour encloses all posterior samples.
        _add_density_contours(
            ax,
            X_post[:n2plt][:, [i, j]],
            color=c_post,
            levels=4,
            zorder=3,
            enclose_all=True,
        )

        # TMCMC KDE contours.
        _add_density_contours(
            ax,
            theta_tmcmc[:n2plt][:, [i, j]],
            color=c_ref,
            levels=4,
            zorder=5,
            enclose_all=False,
        )

        ax.set_xlabel(theta_latex_names[i], fontsize=11)
        ax.set_ylabel(theta_latex_names[j], fontsize=11)
        ax.set_title(
            f"{theta_latex_names[i]} vs {theta_latex_names[j]}",
            fontsize=12,
            pad=8,
        )

        _panel_label(ax, f"({chr(97 + p)})")

    # Hide unused axes
    for ax in axs[len(pairs_theta):]:
        ax.axis("off")

    legend_handles = []

    if plot_sim:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=c_sim,
                markeredgecolor="none",
                alpha=0.35,
                label=sim_label,
            )
        )

    legend_handles.extend([
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=c_post,
            markeredgecolor="none",
            alpha=0.55,
            label=posterior_label,
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            linestyle="None",
            markersize=10,
            color=c_ref,
            label=ref_label,
        ),
    ])

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=11,
    )

    fig.suptitle(
        "AIRMODE: Posterior parameter samples vs TMCMC reference",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ------------------------------------------------------------
    # FIGURE 2: response / predictive space
    ncols = 2
    nrows = int(np.ceil(len(pairs_y) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows))
    axs = np.asarray(axs).flatten()

    for p, (i, j) in enumerate(pairs_y):
        ax = axs[p]
        _style_axis(ax)

        if i >= Y_post.shape[1] or j >= Y_post.shape[1]:
            ax.axis("off")
            continue

        # Optional archive simulated responses
        if plot_sim:
            ax.scatter(
                Y_sim[:n2plt, i],
                Y_sim[:n2plt, j],
                s=10,
                c=c_sim,
                alpha=0.12,
                edgecolors="none",
                zorder=1,
            )

        # Posterior predictive samples
        ax.scatter(
            Y_post[:n2plt, i],
            Y_post[:n2plt, j],
            s=14,
            c=c_post,
            alpha=0.18,
            edgecolors="none",
            zorder=2,
        )

        # Empirical data
        ax.scatter(
            Y_emp[:, i],
            Y_emp[:, j],
            s=34,
            c=c_emp,
            alpha=0.85,
            marker="+",
            linewidths=1.2,
            zorder=4,
        )

        # Posterior predictive KDE contours.
        # The outermost contour encloses all posterior predictive samples.
        _add_density_contours(
            ax,
            Y_post[:n2plt][:, [i, j]],
            color=c_post,
            levels=4,
            zorder=3,
            enclose_all=True,
        )

        ax.set_xlabel(Y_latex_names[i], fontsize=11)
        ax.set_ylabel(Y_latex_names[j], fontsize=11)
        ax.set_title(
            f"{Y_latex_names[i]} vs {Y_latex_names[j]}",
            fontsize=12,
            pad=8,
        )

        _panel_label(ax, f"({chr(97 + p)})")

    # Hide unused axes
    for ax in axs[len(pairs_y):]:
        ax.axis("off")

    legend_handles = []

    if plot_sim:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=c_sim,
                markeredgecolor="none",
                alpha=0.35,
                label=sim_label,
            )
        )

    legend_handles.extend([
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=c_post,
            markeredgecolor="none",
            alpha=0.55,
            label=predictive_label,
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            linestyle="None",
            markersize=10,
            color=c_emp,
            label=empirical_label,
        ),
    ])

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=11,
    )

    fig.suptitle(
        "AIRMODE: Posterior predictive responses vs empirical observations",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_posterior_with_marginals(
    X_post,
    theta_true=None,
    xlim=None,
    ylim=None,
    figsize=(7, 7),
    n_grid_1d=300,
    n_grid_2d=100,
    scatter_alpha=0.20,
    scatter_size=10,
    contour_levels=8,
    filled_levels=15,
    show_filled=True,
    show_contours=True,
):
    """
    Plot 2D posterior samples with:
      - central scatter plot
      - 2D KDE contours on the scatter
      - 1D KDE marginals on top and right

    Parameters
    ----------
    X_post : array-like of shape (n_samples, 2)
        Posterior samples.
    theta_true : array-like of shape (2,) or (n_points, 2), optional
        True/target parameter value(s) to overlay.
    xlim : tuple, optional
        Limits for theta_1 axis, e.g. (-2, 2).
    ylim : tuple, optional
        Limits for theta_2 axis, e.g. (-2, 2).
    figsize : tuple
        Figure size.
    n_grid_1d : int
        Number of points for 1D KDE grids.
    n_grid_2d : int
        Number of points per axis for 2D KDE grid.
    scatter_alpha : float
        Alpha for posterior scatter points.
    scatter_size : float
        Marker size for posterior scatter points.
    contour_levels : int
        Number of 2D contour levels.
    filled_levels : int
        Number of filled contour levels.
    show_filled : bool
        Whether to show filled 2D KDE contours.
    show_contours : bool
        Whether to show 2D KDE contour lines.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict
        Dictionary containing axes:
        {"kde_x": ax_kde_x, "scatter": ax_scatter, "kde_y": ax_kde_y}
    """
    X_post = np.asarray(X_post, dtype=float)
    if X_post.ndim != 2 or X_post.shape[1] != 2:
        raise ValueError("X_post must have shape (n_samples, 2).")

    if theta_true is not None:
        theta_true = np.asarray(theta_true, dtype=float)
        if theta_true.ndim == 1:
            theta_true = theta_true.reshape(1, -1)
        if theta_true.shape[1] != 2:
            raise ValueError("theta_true must have shape (2,) or (n_points, 2).")

    if xlim is None:
        pad_x = 0.05 * (X_post[:, 0].max() - X_post[:, 0].min() + 1e-12)
        xlim = (X_post[:, 0].min() - pad_x, X_post[:, 0].max() + pad_x)

    if ylim is None:
        pad_y = 0.05 * (X_post[:, 1].max() - X_post[:, 1].min() + 1e-12)
        ylim = (X_post[:, 1].min() - pad_y, X_post[:, 1].max() + pad_y)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        hspace=0.05,
        wspace=0.05,
    )

    ax_kde_x = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_kde_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # =========================================================
    # 2D KDE
    # =========================================================
    x_grid_2d = np.linspace(xlim[0], xlim[1], n_grid_2d)
    y_grid_2d = np.linspace(ylim[0], ylim[1], n_grid_2d)
    xx, yy = np.meshgrid(x_grid_2d, y_grid_2d)

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([X_post[:, 0], X_post[:, 1]])

    kde_2d = gaussian_kde(values)
    zz = np.reshape(kde_2d(positions), xx.shape)

    if show_filled:
        ax_scatter.contourf(xx, yy, zz, levels=filled_levels, alpha=0.3)

    if show_contours:
        ax_scatter.contour(xx, yy, zz, levels=contour_levels, linewidths=1)

    # =========================================================
    # scatter
    # =========================================================
    ax_scatter.scatter(
        X_post[:, 0],
        X_post[:, 1],
        s=scatter_size,
        alpha=scatter_alpha,
        label="posterior",
    )

    if theta_true is not None:
        ax_scatter.scatter(
            theta_true[:, 0],
            theta_true[:, 1],
            color="r",
            marker="+",
            s=20,
            alpha=0.3,
            label="target",
        )

    ax_scatter.grid(True)
    ax_scatter.set_xlim(xlim)
    ax_scatter.set_ylim(ylim)
    ax_scatter.set_xlabel(r"$\theta_1$")
    ax_scatter.set_ylabel(r"$\theta_2$")
    ax_scatter.legend()

    # =========================================================
    # 1D KDE for theta_1
    # =========================================================
    x_grid = np.linspace(xlim[0], xlim[1], n_grid_1d)
    kde_x = gaussian_kde(X_post[:, 0])
    dens_x = kde_x(x_grid)

    ax_kde_x.plot(x_grid, dens_x)
    ax_kde_x.fill_between(x_grid, dens_x, alpha=0.2)
    ax_kde_x.set_xlim(xlim)
    ax_kde_x.grid(True)
    ax_kde_x.tick_params(labelbottom=False)
    ax_kde_x.set_ylabel("density")

    if theta_true is not None:

        if len(theta_true)>10:
            kde_theta = gaussian_kde(theta_true[:, 0])
            dens_theta = kde_theta(x_grid)

            ax_kde_x.plot(x_grid, dens_theta, color="r")
            ax_kde_x.fill_between(x_grid, dens_theta, alpha=0.2)

        else:
            for val in theta_true[:, 0]:
                ax_kde_x.axvline(val, color="r", linestyle="--", alpha=0.2)

    # =========================================================
    # 1D KDE for theta_2
    # =========================================================
    y_grid = np.linspace(ylim[0], ylim[1], n_grid_1d)
    kde_y = gaussian_kde(X_post[:, 1])
    dens_y = kde_y(y_grid)

    ax_kde_y.plot(dens_y, y_grid)
    ax_kde_y.fill_betweenx(y_grid, 0, dens_y, alpha=0.2)
    ax_kde_y.set_ylim(ylim)
    ax_kde_y.grid(True)
    ax_kde_y.tick_params(labelleft=False)
    ax_kde_y.set_xlabel("density")

    if theta_true is not None:

        if len(theta_true)>10:
            kde_theta = gaussian_kde(theta_true[:, 1])
            dens_theta = kde_theta(y_grid)
            ax_kde_y.plot(dens_theta, y_grid, color="r")
            ax_kde_y.fill_betweenx(y_grid, dens_theta, alpha=0.2)

        else:
            for val in theta_true[:, 1]:
                ax_kde_y.axhline(val, color="r", linestyle="--", alpha=0.2)

    plt.tight_layout()

    return fig, {"kde_x": ax_kde_x, "scatter": ax_scatter, "kde_y": ax_kde_y}




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8, labelsize=11)
    ax.grid(False)


def _panel_label(ax, txt):
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12
    )


def _add_kde_contours(
    ax,
    samples_2d,
    color="k",
    linestyle="-",
    linewidth=1.5,
    levels=(0.50, 0.75, 0.90),
    zorder=3,
    alpha=1.0,
):
    """
    Add KDE contours from 2D samples.
    levels are quantiles of KDE values at the sample locations.
    """

    samples_2d = np.asarray(samples_2d, dtype=float)
    samples_2d = samples_2d[np.isfinite(samples_2d).all(axis=1)]

    if samples_2d.shape[0] < 10:
        return

    x = samples_2d[:, 0]
    y = samples_2d[:, 1]

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return

    try:
        kde = gaussian_kde(samples_2d.T)
    except np.linalg.LinAlgError:
        return

    # plotting grid
    padx = 0.20 * max(np.ptp(x), 1e-6)
    pady = 0.20 * max(np.ptp(y), 1e-6)

    xx, yy = np.meshgrid(
        np.linspace(x.min() - padx, x.max() + padx, 200),
        np.linspace(y.min() - pady, y.max() + pady, 200)
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # contour levels based on quantiles of density at sample locations
    dens_at_samples = kde(samples_2d.T)
    contour_levels = np.quantile(dens_at_samples, levels)
    contour_levels = np.unique(contour_levels)

    if len(contour_levels) == 0:
        return

    cs = ax.contour(
        xx, yy, zz,
        levels=contour_levels,
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    return cs


def plot_airmode_reference_overlay(
    my_samples=None,
    my_label="My results",
    my_color="royalblue",
    plot_my_scatter=True,
    plot_my_kde=True,
    scatter_alpha=0.10,
    scatter_size=5,
):
    """
    Plot a paper-style AIRMODE figure and optionally superimpose user's samples.

    Parameters
    ----------
    my_samples : array-like of shape (n_samples, 12), optional
        Your posterior samples ordered as:
        [theta1, theta2, ..., theta11, theta12]
        where theta12 is assumed to correspond to sigma.
    my_label : str
        Label for your results in legend.
    my_color : str
        Color for your results.
    """

    # ------------------------------------------------------------------
    # Reference means from your Table 4 (10% variance)
    # NOTE: theta12 is assumed to correspond to sigma in the paper figure.
    tm_means = {
        "theta1": 0.297,
        "theta2": 0.308,
        "theta3": 0.288,
        "theta4": 0.300,
        "theta5": 0.233,
        "theta6": 0.371,
        "theta7": 0.301,
        "theta8": 0.306,
        "theta9": 0.310,
        "theta10": 0.305,
        "theta11": 0.315,
        "theta12": -0.545,
    }

    tmcmc_means = {
        "theta1": 0.283,
        "theta2": 0.294,
        "theta3": 0.305,
        "theta4": 0.284,
        "theta5": 0.379,
        "theta6": 0.513,
        "theta7": -0.042,
        "theta8": 0.510,
        "theta9": 0.684,
        "theta10": -0.035,
        "theta11": 0.290,
        "theta12": 0.015,
    }

    # ------------------------------------------------------------------
    # Mapping from parameter name to column index in samples
    param_to_idx = {f"theta{i}": i - 1 for i in range(1, 13)}

    # ------------------------------------------------------------------
    # Panel definitions
    # These are chosen to match the panels visible in the figure.

    # Fallback bounds if my_samples is None
    default_bounds = {
        ("theta5", "theta6"): ((-1.0, 1.0), (-1.0, 1.0)),
        ("theta10", "theta11"): ((-1.0, 1.0), (-1.0, 1.0)),
        ("theta7", "theta8"): ((-1.0, 1.0), (-1.0, 1.0)),
        ("theta7", "theta9"): ((-1.0, 1.0), (-1.0, 1.0)),
        ("theta2", "theta12"): ((-1.0, 1.0), (-1.0, 1.0)),
    }

    panel_defs = [
        ("theta5", "theta6", r"$\theta_5$", r"$\theta_6$"),
        ("theta10", "theta11", r"$\theta_{10}$", r"$\theta_{11}$"),
        ("theta7", "theta8", r"$\theta_7$", r"$\theta_8$"),
        ("theta7", "theta9", r"$\theta_7$", r"$\theta_9$"),
        ("theta2", "theta11", r"$\theta_2$", r"$\theta_{11}$"),
    ]


    panels = [
        ("theta5", "theta6", r"$\theta_5$", r"$\theta_6$", (-1.0, 1.0), (-1.0, 1.0)),
        ("theta10", "theta11", r"$\theta_{10}$", r"$\theta_{11}$", (-1.0, 1.0), (-1.0, 1.0)),
        ("theta7", "theta8", r"$\theta_7$", r"$\theta_8$", (-1.0, 1.0), (-1.0, 1.0)),
        ("theta7", "theta9", r"$\theta_7$", r"$\theta_9$", (-1.0, 1.0), (-1.0, 1.0)),
        ("theta2", "theta11", r"$\theta_2$", r"$\theta_{11}$", (-1.0, 1.0), (-1.0, 1.0)),
    ]

    panels = []

    for px, py, lx, ly in panel_defs:
        if my_samples is None:
            xlim, ylim = default_bounds[(px, py)]
        else:
            my_samples_arr = np.asarray(my_samples, dtype=float)

            ix = param_to_idx[px]
            iy = param_to_idx[py]

            x_vals = np.concatenate([
                my_samples_arr[:, ix],
                [tm_means[px], tmcmc_means[px]]
            ])
            y_vals = np.concatenate([
                my_samples_arr[:, iy],
                [tm_means[py], tmcmc_means[py]]
            ])

            x_vals = x_vals[np.isfinite(x_vals)]
            y_vals = y_vals[np.isfinite(y_vals)]

            x_min, x_max = np.min(x_vals), np.max(x_vals)
            y_min, y_max = np.min(y_vals), np.max(y_vals)

            # Add a little padding
            x_pad = 0.08 * max(x_max - x_min, 1e-6)
            y_pad = 0.08 * max(y_max - y_min, 1e-6)

            # If all values are identical, make a small symmetric interval
            if x_max == x_min:
                x_pad = 0.05 * max(abs(x_min), 1.0)
            if y_max == y_min:
                y_pad = 0.05 * max(abs(y_min), 1.0)

            xlim = (x_min - x_pad, x_max + x_pad)
            ylim = (y_min - y_pad, y_max + y_pad)

        panels.append((px, py, lx, ly, xlim, ylim))
    # ------------------------------------------------------------------
    # Prepare figure: 2 rows x 3 columns, last panel used for legend
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    axs = axs.flatten()

    letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    # ------------------------------------------------------------------
    # Plot each panel
    for k, (px, py, lx, ly, xlim, ylim) in enumerate(panels):
        ax = axs[k]
        _style_axis(ax)

        # reference means
        x_tm = tm_means[px]
        y_tm = tm_means[py]
        x_tmcmc = tmcmc_means[px]
        y_tmcmc = tmcmc_means[py]

        # TM mean
        ax.scatter(
            x_tm, y_tm,
            color="red",
            s=90,
            marker="o",
            facecolors="none",
            linewidths=1.8,
            zorder=4
        )

        # TMCMC mean
        ax.scatter(
            x_tmcmc, y_tmcmc,
            color="black",
            s=85,
            marker="x",
            linewidths=2.0,
            zorder=4
        )

        # user's results
        if my_samples is not None:
            my_samples = np.asarray(my_samples, dtype=float)
            ix = param_to_idx[px]
            iy = param_to_idx[py]
            pts = my_samples[:, [ix, iy]]

            if plot_my_scatter:
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=scatter_size,
                    color=my_color,
                    alpha=scatter_alpha,
                    edgecolors="none",
                    zorder=1
                )

            if plot_my_kde:
                _add_kde_contours(
                    ax,
                    pts,
                    color=my_color,
                    linestyle="-",
                    linewidth=1.6,
                    levels=(0.01, 0.20, 0.8, 0.99),
                    zorder=3,
                    alpha=0.95,
                )

        ax.set_xlabel(lx, fontsize=14)
        ax.set_ylabel(ly, fontsize=14)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True)
        _panel_label(ax, letters[k])

    # ------------------------------------------------------------------
    # Legend in the 6th subplot
    ax_leg = axs[5]
    ax_leg.axis("off")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.8,
            label="TM mean (Table 4)"
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            markersize=10,
            color="black",
            markeredgewidth=2.0,
            label="TMCMC mean (Table 4)"
        )
    ]

    if my_samples is not None:
        handles.append(
            Line2D(
                [0], [0],
                color=my_color,
                linewidth=1.8,
                label=my_label
            )
        )

    ax_leg.legend(
        handles=handles,
        loc="center",
        frameon=True,
        fontsize=12
    )

    fig.suptitle(
        "AIRMODE-style reference plot for superimposing custom results",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()