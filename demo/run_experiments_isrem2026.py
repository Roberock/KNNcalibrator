import matplotlib.pyplot as plt
import torch

# from adaptive_ABC_KNN_KDE import AdaptiveInverseKNNKDE as SOLVER
from adaptive_ABC_KNN_KDE import SimpleAdaptiveKNNABC as SOLVER

from adaptive_ABC_KNN_KDE import UniformBoxPrior
from resources.loader_usecases import prepare_case, prepare_case_2_data
from helpers_ISRERM2026 import *
from resources.AIRMODE.load_helpers import *


KNN = 40
def run_case_1_paraboloid(DGM=3):
    # problem 1 - paraboloid inverse problem
    _, Demp, _ = prepare_case(1, Nemp=10_000, Nsim=10, DGM=DGM)
    Theta_target, Y_target  = [D['theta'] for _, D in Demp.items()], [D['y_data'] for _, D in Demp.items()]

    M, Demp, Dsim = prepare_case(1, Nemp=200, Nsim=100, DGM=DGM)
    M_design, Y_emp_by_design, sim_db = adapt_case1_for_multidesign(M, Demp, Dsim)

    ## RUN - the adaptive refine inversion with the solver
    prior = UniformBoxPrior(low=[-15, -15],
                            high=[15, 15])

    solver = SOLVER(model=M_design,  Y_emp_by_design=Y_emp_by_design,
                    prior=prior,  sim_db=sim_db,  K=KNN, ridge=1e-3, )

    hist, diag = solver.adaptive_refine(
        max_iter=20,  n_new_per_iter=5,
        top_frac=0.1,   inflate=0.3,
        improve_tol=0.001,   patience=4,  min_iter=2,
        n_post_pred=1000,
        sampler_mcmcm=False,  # 'mcmc' or 'ball'
        keep_best_state=True,
        true_target=Theta_target[0],  # unavailable in practice
        verbose=True,
    )

    X_post = solver.sample_posterior_particles_smooth(n_samples=2000)

    plt.scatter(X_post[:, 0], X_post[:, 1], color='b', marker='o', label='posterior')
    plt.scatter(Theta_target[0][:, 0], Theta_target[0][:, 1], color='r', marker='+', label='target')
    plt.grid(True)
    plt.ylabel(r'$\theta_2$')
    plt.xlabel(r'$\theta_1$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #theta_mode = solver.posterior_mode()

    print(" 🎬 Generating posterior progression plot...")
    print("\n🎯 SUMMARY:")
    print(f"• Initial mean radius: {hist['mean_radius'].iloc[0]:.3f} → Final: {hist['mean_radius'].iloc[-1]:.3f}")
    if "max_radius" in hist.columns:
        print(f"• Initial max radius: {hist['max_radius'].iloc[0]:.3f} → Final: {hist['max_radius'].iloc[-1]:.3f}")
    if "db_size" in hist.columns:
        print(f"• Database: {hist['db_size'].iloc[0]} → {hist['db_size'].iloc[-1]} samples")
    print(f"• Iterations: {len(hist)} (stopped: {hist['stop_reason'].iloc[-1]})")
    #mode_x, mode_p = solver.posterior_mode_from_db()
    #print(f"• Final posterior mode: {mode_x}")


def run_case_2_airmod():
    # ============================================================
    # AIRMODE calibration workflow
    model_airmode, Y_emp_by_design, sim_db, prior = prepare_airmode_for_calibration(Nemp=30, Nsim=1000)

    # ---------- Paths, Load, Extract TMCMC ----------
    ref_tmcmc_data_path = "../resources/AIRMODE/data/reference_TMCMC_DLRAirmod.mat"
    tmobj = load_mat_any(str(ref_tmcmc_data_path))
    theta_tmcmc, src_path = find_theta_matrix_from_mat(tmobj)
    print(f"[TMCMC] extracted from '{src_path}' -> {theta_tmcmc.shape}")



    solver = SOLVER(
        model=model_airmode,  Y_emp_by_design=Y_emp_by_design,
        prior=prior,   sim_db=sim_db,   K=KNN,     ridge=1e-2,  seed=123,     )

    """hist, diag = solver.adaptive_refine_v0( max_iter=50,  top_frac=0.1,
                                         n_new_per_iter=20,  inflate=1.0, min_iter=20,
                                         target_shrink=0.00, improve_tol=0.001,    patience=3,  )
    hist, diag = solver.adaptive_refine(
        max_iter=20,
        top_frac=0.2,
        n_new_per_iter=50,
        inflate=1.5,
        prior_weight=0.20,
        posterior_weight=0.50,
        local_weight=0.30,
        posterior_temp=0.70,
        improve_tol=0.05,
        patience=4,
        min_iter=3,
        n_post_pred=200,
        keep_best_state=True,
        true_target=theta_tmcmc,  # unavailable in practice
    )"""
    hist, diag = solver.adaptive_refine(
        max_iter=8,
        n_new_per_iter=50,
        inflate=0.8,
        true_target=theta_tmcmc,  # unavailable in practice
    )
    print("AIRMODE SUMMARY")
    print(f"Initial mean radius: {hist['mean_radius'].iloc[0]:.4f}")
    print(f"Final mean radius:   {hist['mean_radius'].iloc[-1]:.4f}")
    print(f"Iterations:          {len(hist)}")
    print(f"Stop reason:         {hist['stop_reason'].iloc[-1]}")

    db_size_initial = sim_db["X"].shape[0]
    db_size_final = solver.X_db.shape[0]
    print(f"Database size:       {db_size_initial} -> {db_size_final}")


    ### PLOTING NOW
    theta_latex_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$",
                         r"$\theta_4$", r"$\theta_5$", r"$\theta_6$",
                         r"$\theta_7$", r"$\theta_8$", r"$\theta_9$",
                         r"$\theta_{10}$", r"$\theta_{11}$"]
    Y_latex_names = [r"$D_1$", r"$D_2$", r"$D_3$", r"$D_4$", r"$D_5$",
                     r"$D_6$", r"$D_7$", r"$D_8$", r"$D_9$", r"$D_{10}$"]
    design = "AIRMODE"
    plot_sim=False
    n2plt = 1_000
    X_sim = np.asarray(solver.X_db, dtype=float)
    Y_sim = np.asarray(solver.Y_db_by_design[design], dtype=float)
    Y_emp = np.asarray(solver.Y_emp_by_design[design], dtype=float)
    X_post = solver.sample_posterior_particles_smooth(1000)
    Y_post = np.asarray(solver.model(X_post, design), dtype=float)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    axs = ax.flatten()
    pairs_2_plt = [[6, 5], [3, 2], [5, 3], [1, 9]]

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
        axi.scatter(Y_emp[:, i], Y_emp[:, j], 10, c="r", marker="+", label="emp")
        axi.set_xlabel(Y_latex_names[i])
        axi.set_ylabel(Y_latex_names[j])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




def run_EXP_ISRERM2026():
    run_case_1_paraboloid()
    run_case_2_airmod()



if __name__ == '__main__':

    run_EXP_ISRERM2026()