import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any, Optional

import time
from scipy.spatial import cKDTree
from scipy.special import digamma
from src.backward import KNNCalibrator
from sklearn.neighbors import NearestNeighbors


def simulate_y_from_xa(simulator, xa_samples, xi, xe_fixed=(2.0, 0.0)):
    """
    xa_samples: (N,2)
    returns y: (N,1)  (squeezed to (N,1) even if simulator returns (N,1,1))
    """
    xa = np.asarray(xa_samples, float)
    N = xa.shape[0]
    xe = np.tile(np.asarray(xe_fixed, float), (N, 1))
    theta_full = np.hstack([xa, xe])  # (N,4)
    xi_arr = np.full((N, 1), float(xi))
    y = np.asarray(simulator(theta_full, xi=xi_arr))
    y = np.squeeze(y)
    return y.reshape(N, 1)

def mi_kraskov(X, Y, k=20, seed=0, jitter=1e-10):
    """
    Kraskov kNN mutual information estimator for continuous variables.
    X: (N,dx), Y: (N,dy)
    Uses max-norm (L_infty) balls.
    """
    # todo: verify this is a correct way to estimate the Expected Information Gain ---> is it the same as Mutual Info??

    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    N = X.shape[0]
    assert Y.shape[0] == N

    # break ties / duplicates
    X = X + jitter * rng.standard_normal(X.shape)
    Y = Y + jitter * rng.standard_normal(Y.shape)

    Z = np.hstack([X, Y])
    treeZ = cKDTree(Z)
    treeX = cKDTree(X)
    treeY = cKDTree(Y)

    # distance to k-th neighbor in joint space (exclude self => query k+1)
    dists, _ = treeZ.query(Z, k=k+1, p=np.inf)
    eps = dists[:, k] - 1e-15  # strict radius

    nx = np.empty(N, dtype=int)
    ny = np.empty(N, dtype=int)
    for i in range(N):
        nx[i] = len(treeX.query_ball_point(X[i], eps[i], p=np.inf)) - 1
        ny[i] = len(treeY.query_ball_point(Y[i], eps[i], p=np.inf)) - 1

    return float(digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1)))




def select_next_xi_by_eig(
    simulator,
    xa_posterior,          # (M,2) samples from current belief p(xa | data)
    xi_candidates,         # list/array of candidate xi
    xe_fixed=(2.0, 0.0),
    n_eval=20000,          # subsample size used to score MI
    k_mi=20,
    seed=0
):
    rng = np.random.default_rng(seed)
    xa_posterior = np.asarray(xa_posterior, float)
    M = xa_posterior.shape[0]
    n_use = min(n_eval, M)
    idx = rng.choice(M, size=n_use, replace=False)
    Xa = xa_posterior[idx]

    scores = []
    times = []
    # todo: here we are esimtating the Mutual Info for each candidate in a grid....
    #  simulate_y_from_xa should be probably replaced by some function from the forward.py module
    for xi in xi_candidates:
        t0 = time.perf_counter()
        y = simulate_y_from_xa(simulator, Xa, xi, xe_fixed=xe_fixed)  # (n_use,1)
        s = mi_kraskov(Xa, y, k=k_mi, seed=seed)
        times.append(time.perf_counter() - t0)
        scores.append(s)

    scores = np.asarray(scores, float)
    best_i = int(np.argmax(scores))
    diag = {
        "n_eval": n_use,
        "k_mi": k_mi,
        "time_per_xi_s_mean": float(np.mean(times)),
        "time_total_s": float(np.sum(times)),
    }
    return float(xi_candidates[best_i]), scores, diag


def observe_empirical(pdf_theta_true, simulator, xi, n_emp, rng=None):
    """
    Query the real system (here: your UM_theta) at design xi.
    Returns y_emp with shape (n_emp, 1).
    """
    # todo: better if this goes in a dgm.py data generating mechanism module?
    #  not appropriate here in the design of experiment doy.py module
    if rng is None:
        rng = np.random.default_rng()
    theta = pdf_theta_true.sample(n_emp)  # (n_emp, d_theta_full) e.g. (n,4)
    y = simulator(theta, xi)  # allow scalar xi
    y = np.asarray(y)
    y = np.squeeze(y)
    return y.reshape(-1, 1)


def build_sim_db(pdf_theta_sim, simulator, xi_candidates, n_sim, rng=None):
    """
    Build simulated archive with xi varying across candidates.
    Returns simulated_data dict for KNNCalibrator.setup()
    """
    # todo: better if this goes in  the simulator.py module?
    if rng is None:
        rng = np.random.default_rng()

    theta_sim = pdf_theta_sim.sample(n_sim)  # (n_sim, 4) here
    xi_vals = rng.choice(np.asarray(xi_candidates, float), size=n_sim, replace=True)
    xi_sim = xi_vals.reshape(-1, 1)  # (n_sim,1)
    y_sim = simulator(theta_sim, xi=xi_sim)  # many codes use kw xi
    y_sim = np.asarray(y_sim)
    y_sim = np.squeeze(y_sim).reshape(-1, 1)

    return {"y": y_sim, "theta": theta_sim[:, :2], "xi": xi_sim}  # theta is xa only


def select_next_xi_by_eig_masked(
        simulator, xa_posterior, xi_candidates, used_xi,
        xe_fixed=(2.0, 0.0), n_eval=20000, k_mi=30, seed=0):

    # todo: why do we need a masked selection???
    #  I'd have expected the EIG to realize twice the same experiment is maybe not informative....or maybe wrong intuition here

    xi_next, scores, diag = select_next_xi_by_eig(
        simulator=simulator,
        xa_posterior=xa_posterior,
        xi_candidates=xi_candidates,
        xe_fixed=xe_fixed,
        n_eval=n_eval,
        k_mi=k_mi,
        seed=seed
    )

    # if selector returned a used xi (can happen on ties), pick best unused
    used = set(float(x) for x in used_xi)
    xi_candidates = np.asarray(xi_candidates, float)
    order = np.argsort(scores)[::-1]
    for j in order:
        cand = float(xi_candidates[j])
        if cand not in used:
            return cand, scores, diag
    return float(xi_next), scores, diag  # fallback


# todo: better if this one takes a 'calibrator' model for a more general posterior estimation procedure ?
def run_sequential_doe_knn(
        simulator,
        pdf_theta_true,  # UM_theta(...)  -> used to generate empirical data
        pdf_theta_prior,  # e.g. a UniformPrior(...) -> used to build simulation archive
        xi0,  # initial design (given)
        xi_candidates,  # grid/list of candidate designs for selection
        nq=5,  # number of NEW experiments to do after xi0
        n_emp=200,  # replicates per empirical query
        n_sim=200_000,  # size of simulated archive
        knn=50,
        a_tol=0.15,
        combine="stack",  # recommend "stack" for stability
        resample_n=10_000,
        seed=0,
        n_eval_eig=20_000,
        k_mi=30,
):
    rng = np.random.default_rng(seed)
    xi_candidates = np.asarray(xi_candidates, float)
    xi_candidates = np.round(xi_candidates, 10)  # avoid float-key headaches

    xi0 = float(np.round(xi0, 10))

    # ---- BUILD SIMULATED ARCHIVE over ALL candidate designs ----
    t0 = time.perf_counter()
    simulated_data = build_sim_db(pdf_theta_prior, simulator, xi_candidates, n_sim=n_sim, rng=rng)
    t_simdb = time.perf_counter() - t0

    # ---- SETUP THE CALIBRATOR USING ALL THE SIMULATED DATA ----
    calib = KNNCalibrator(knn=knn, evaluate_model=False, a_tol=a_tol)
    calib.setup(simulated_data=simulated_data, xi_list=list(xi_candidates))

    # ---- COLLECT INITIAL EMPIRICAL OBSERVATIONS FOR THE FIRST DESIGN at xi0 ----

    t0 = time.perf_counter()
    y0 = observe_empirical(pdf_theta_true, simulator, xi0, n_emp=n_emp, rng=rng)
    t_emp0 = time.perf_counter() - t0

    observations = [(y0, xi0)]

    # ---- initial posterior ----
    t0 = time.perf_counter()
    post = calib.calibrate(observations, combine=combine, resample_n=resample_n)
    xa_post = post["theta"]
    t_cal0 = time.perf_counter() - t0

    history = {
        "xi": [xi0],
        "scores": [],          # EIG proxy curve per step
        "post": [xa_post],     # posterior samples per step
        "timing": {
            "simdb_s": t_simdb,
            "emp_s": [t_emp0],
            "cal_s": [t_cal0],
            "eig_s": [],
        },
        "meta": {
            "n_sim": n_sim, "n_emp": n_emp, "knn": knn, "a_tol": a_tol,
            "combine": combine, "resample_n": resample_n,
            "n_eval_eig": n_eval_eig, "k_mi": k_mi,
        }
    }

    used_xi = {xi0}

    # ---- sequential loop ----
    for q in range(nq):
        # 1) pick next xi by EIG/MI under current posterior
        t0 = time.perf_counter()
        xi_next, score_curve, diag_eig = select_next_xi_by_eig_masked(
            simulator=simulator,
            xa_posterior=xa_post,
            xi_candidates=xi_candidates,
            used_xi=used_xi,
            n_eval=n_eval_eig,
            k_mi=k_mi,
            seed=seed + 1000 + q,
        )
        t_eig = time.perf_counter() - t0

        # 2) collect new empirical data at xi_next
        t0 = time.perf_counter()
        y_new = observe_empirical(pdf_theta_true, simulator, xi_next, n_emp=n_emp, rng=rng)
        t_emp = time.perf_counter() - t0

        observations.append((y_new, xi_next))

        # 3) update posterior given ALL collected data so far
        t0 = time.perf_counter()
        post = calib.calibrate(observations, combine=combine, resample_n=resample_n)
        xa_post = post["theta"]
        t_cal = time.perf_counter() - t0

        # book-keeping experimental hisory report
        used_xi.add(xi_next)
        history["xi"].append(xi_next)
        history["scores"].append(score_curve)
        history["post"].append(xa_post)
        history["timing"]["eig_s"].append(t_eig)
        history["timing"]["emp_s"].append(t_emp)
        history["timing"]["cal_s"].append(t_cal)

        print(f"[q={q + 1}/{nq}] xi_next={xi_next:.4f} | "
              f"EIG_time={t_eig:.2f}s emp_time={t_emp:.2f}s cal_time={t_cal:.2f}s "
              f"| post_n={xa_post.shape[0]}")

    return history




def _ensure_2d(a):
    a = np.asarray(a, float)
    if a.ndim == 1:
        return a[:, None]
    return a

def _filter_by_xi(y, theta, xi, xi_star, a_tol):
    xi = _ensure_2d(xi)
    xi_star = np.asarray(xi_star, float).reshape(1, -1)
    # absolute infinity-norm tolerance (usually safer than relative when xi_star ~ 0)
    keep = np.max(np.abs(xi - xi_star), axis=1) <= a_tol
    return y[keep], theta[keep], xi[keep]

def _cov_logdet(theta_block, eps=1e-9):
    # theta_block: (k, d_theta)
    S = np.cov(theta_block.T, bias=False)
    S = np.atleast_2d(S)
    S = S + eps * np.eye(S.shape[0])
    sign, ld = np.linalg.slogdet(S)
    # if numerical issues: penalize
    return float(ld) if sign > 0 else float("inf")

def estimate_eig_score_knn(
    simulated_data,          # the archive, a list as follows [y_sim, theta_sim, xi_sim]
    xi_candidate,            # the design candidate
    knn=50,                  # the number of k-nearest neighbours
    a_tol=0.05,              # bin half-width in the xi filtration process
    M_y=200,                 # predictive samples y to average over
    predictive="archive",    # "archive" or "posterior"
    y_predictive=None,       # optional callable: y_predictive(xi)->(M_y,d_y)
    rng=None,
):
    """
    Returns:
      score: float  (lower is better; corresponds to lower E[H(theta|y,xi)])
    """
    rng = np.random.default_rng() if rng is None else rng
    y_sim, theta_sim, xi_sim = simulated_data
    y_sim = _ensure_2d(y_sim)
    theta_sim = _ensure_2d(theta_sim)

    # 1) condition archive on xi
    y_xi, th_xi, _ = _filter_by_xi(y_sim, theta_sim, xi_sim, xi_candidate, a_tol)
    if y_xi.shape[0] < knn:
        return float("inf")  # not enough support at this xi

    # 2) fit kNN in y-space
    scaler = StandardScaler()
    scaler.fit(y_xi)
    Z = scaler.transform(y_xi)
    neigh = NearestNeighbors(n_neighbors=knn)
    neigh.fit(Z)

    # 3) sample predictive y
    if predictive == "archive":
        take = rng.choice(y_xi.shape[0], size=min(M_y, y_xi.shape[0]), replace=False)
        y_draws = y_xi[take]
    elif predictive == "posterior":
        if y_predictive is None:
            raise ValueError("predictive='posterior' requires y_predictive(xi)->y_draws.")
        y_draws = _ensure_2d(y_predictive(xi_candidate))
    else:
        raise ValueError("predictive must be 'archive' or 'posterior'.")

    # 4) expected posterior dispersion via kNN inversion
    y_drawsZ = scaler.transform(y_draws)
    _, idx = neigh.kneighbors(y_drawsZ)

    Hs = []
    for ids in idx:
        theta_block = th_xi[ids]          # (knn, d_theta)
        Hs.append(_cov_logdet(theta_block))

    return float(np.mean(Hs))
