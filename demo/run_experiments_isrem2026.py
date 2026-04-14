import torch
from adaptive_ABC_KNN_KDE import AdaptiveInverseKNNKDE as SOLVER
from adaptive_ABC_KNN_KDE import UniformBoxPrior
from resources.loader_usecases import prepare_case, prepare_case_2_data
from helpers_ISRERM2026 import *
from resources.AIRMODE.load_helpers import *

def run_case_1_paraboloid():
    # problem 1 - paraboloid inverse problem
    _, Demp, _ = prepare_case(1, Nemp=10_000, Nsim=10)
    Theta_target, Y_target  = [D['theta'] for _, D in Demp.items()], [D['y_data'] for _, D in Demp.items()]

    M, Demp, Dsim = prepare_case(1, Nemp=100, Nsim=50)
    M_design, Y_emp_by_design, sim_db = adapt_case1_for_multidesign(M, Demp, Dsim)

    ## RUN - the adaptive refine inversion with the solver
    solver = SOLVER(model=M_design,
                    Y_emp_by_design=Y_emp_by_design,
                    prior=UniformBoxPrior(low=[-15, -15], high=[15, 15]),
                    sim_db=sim_db,  K=5, ridge=1e-3, )

    """hist, diag = solver.adaptive_refine_v0( max_iter=50, top_frac=0.5,
                                         n_new_per_iter=100, inflate=1.0,
                                         min_iter=10, target_shrink=0.001,
                                         improve_tol=0.001, patience=5,
                                         true_target = Theta_target[0])
    """
    hist, diag = solver.adaptive_refine(
        max_iter=20,
        top_frac=0.15,
        n_new_per_iter=250,
        inflate=0.5,
        prior_weight=0.20,
        posterior_weight=0.50,
        local_weight=0.30,
        posterior_temp=0.70,
        improve_tol=0.005,
        patience=4,
        min_iter=3,
        n_post_pred=1000,
        keep_best_state=True,
        true_target=Theta_target[0],  # unavailable in practice
    )
    print(" 🎬 Generating posterior progression plot...")


    print("\n🎯 SUMMARY:")
    print(f"• Initial mean radius: {hist['mean_radius'].iloc[0]:.3f} → Final: {hist['mean_radius'].iloc[-1]:.3f}")
    if "max_radius" in hist.columns:
        print(f"• Initial max radius: {hist['max_radius'].iloc[0]:.3f} → Final: {hist['max_radius'].iloc[-1]:.3f}")
    if "db_size" in hist.columns:
        print(f"• Database: {hist['db_size'].iloc[0]} → {hist['db_size'].iloc[-1]} samples")
    print(f"• Iterations: {len(hist)} (stopped: {hist['stop_reason'].iloc[-1]})")
    mode_x, mode_p = solver.posterior_mode_from_db()
    print(f"• Final posterior mode: {mode_x}")

    plot_posterior_vs_true_theta_case1(solver, Demp)
    plot_posterior_predictive_vs_empirical(solver, n_post_samples=3000)
    plot_empirical_vs_posterior_intervals(solver, n_post_samples=3000)
    solver.fit_local_models()
    plot_posterior_x_by_design_case1(solver)
    plot_posterior_vs_true_theta_by_design_case1(solver, Demp)

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
        prior=prior,   sim_db=sim_db,   K=20,     ridge=1e-2,  seed=123,     )

    """hist, diag = solver.adaptive_refine_v0( max_iter=50,  top_frac=0.1,
                                         n_new_per_iter=20,  inflate=1.0, min_iter=20,
                                         target_shrink=0.00, improve_tol=0.001,    patience=3,  )"""
    hist, diag = solver.adaptive_refine(
        max_iter=20,
        top_frac=0.15,
        n_new_per_iter=250,
        inflate=0.5,
        prior_weight=0.20,
        posterior_weight=0.50,
        local_weight=0.30,
        posterior_temp=0.70,
        improve_tol=0.005,
        patience=4,
        min_iter=3,
        n_post_pred=1000,
        keep_best_state=True,
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


    X_post, Y_post = plot_airmode_posterior_style(
        solver=solver,
        theta_tmcmc=theta_tmcmc,
        theta_latex_names=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$", r"$\theta_6$",
                           r"$\theta_7$", r"$\theta_8$", r"$\theta_9$", r"$\theta_{10}$", r"$\theta_{11}$"],
        Y_latex_names=[r"$D_1$", r"$D_2$", r"$D_3$", r"$D_4$", r"$D_5$", r"$D_6$", r"$D_7$", r"$D_8$", r"$D_9$",
                       r"$D_{10}$"],
        pairs_2_plt=[[6, 5], [3, 2], [5, 3], [1, 9]],
        n2plt=10_000,
        n_post=10_000,
        design="AIRMODE",
    )


def run_EXP_ISRERM2026():
    run_case_1_paraboloid()
    run_case_2_airmod()



if __name__ == '__main__':

    run_EXP_ISRERM2026()