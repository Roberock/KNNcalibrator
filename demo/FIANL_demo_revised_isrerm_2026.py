import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


import torch

from adaptive_ABC_KNN_KDE import UniformBoxPrior

from resources.loader_usecases import prepare_case, prepare_case_2_data
from helpers_ISRERM2026 import *
from resources.AIRMODE.load_helpers import *

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SAMPLE_WORST_RADIUS = True

class SimpleAdaptiveKNNABC:
    """
    Adaptive archive-based KNN-KDE inverse solver.

    Interpretation
    --------------
    The method reconstructs an input density consistent with observed outputs
    by:
      1) keeping a shared archive of simulated inputs and outputs,
      2) identifying local neighborhoods in response space via KNN,
      3) mapping them back to theta-space,
      4) combining the resulting local kernel models across designs,
      5) adaptively enriching the archive where response-space support is weak.

    Assumptions
    -----------
    - model(X, design) -> simulated outputs
    - prior.rvs(size=...) and prior.logpdf(X) exist
    """

    def __init__(
        self,
        model,
        Y_emp_by_design,
        prior,
        sim_db=None,
        N0=500,
        K=10,
        ridge=1e-3,
        seed=123,
        knn_metric="euclidean",
    ):
        self.model = model
        self.prior = prior
        self.K = int(K)
        self.ridge = float(ridge)
        self.knn_metric = knn_metric
        self.rng = np.random.default_rng(seed)

        self.designs = list(Y_emp_by_design.keys())
        if len(self.designs) == 0:
            raise ValueError("Y_emp_by_design is empty.")

        self.Y_emp_by_design = {
            d: self._ensure_2d(Y_emp_by_design[d]) for d in self.designs
        }

        self.d_x = self._infer_dx()

        # shared archive
        self.X_db = None
        self.Y_db_by_design = {}
        self.log_prior_db = None
        self.log_prop_db = None

        # fitted local models and posterior weights
        self.local_models = []
        self.log_post_weights_db = None
        self.post_weights_db = None

        # history
        self.history = []

        if sim_db is None:
            self._initialize_prior_db(N0)
        else:
            self._load_db(sim_db)

    # ------------------------------------------------------------------
    # basic utilities
    def _infer_dx(self):
        if hasattr(self.prior, "d_x"):
            return int(self.prior.d_x)
        if hasattr(self.prior, "low") and hasattr(self.prior, "high"):
            return len(np.asarray(self.prior.low))
        if hasattr(self.prior, "mean"):
            return len(np.asarray(self.prior.mean))
        raise ValueError("Cannot infer d_x from prior.")

    @staticmethod
    def _ensure_2d(Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        return Y

    def _prior_rvs(self, size):
        X = self.prior.rvs(size=size)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, self.d_x)
        return X

    def _simulate_all_designs(self, X):
        X = np.asarray(X, dtype=float)
        out = {}
        for d in self.designs:
            out[d] = self._ensure_2d(self.model(X, d))
        return out

    def _initialize_prior_db(self, N0):
        X = self._prior_rvs(N0)
        Y_by_design = self._simulate_all_designs(X)
        logp = np.asarray(self.prior.logpdf(X), dtype=float)

        self.X_db = X
        self.Y_db_by_design = Y_by_design
        self.log_prior_db = logp
        self.log_prop_db = logp.copy()

    def _load_db(self, sim_db):
        self.X_db = np.asarray(sim_db["X"], dtype=float)
        self.Y_db_by_design = {
            d: self._ensure_2d(sim_db["Y_by_design"][d]) for d in self.designs
        }
        self.log_prior_db = np.asarray(
            sim_db.get("log_prior", self.prior.logpdf(self.X_db)),
            dtype=float,
        )
        self.log_prop_db = np.asarray(
            sim_db.get("log_prop", self.log_prior_db.copy()),
            dtype=float,
        )

    @staticmethod
    def _safe_mvn_logpdf(X, mean, cov):
        return multivariate_normal(mean=mean, cov=cov, allow_singular=True).logpdf(X)

    def _weighted_cov(self, X, w):
        w = np.asarray(w, dtype=float)
        w = w / max(w.sum(), 1e-300)
        mu = np.sum(X * w[:, None], axis=0)
        Xc = X - mu
        denom = max(1e-12, 1.0 - np.sum(w ** 2))
        cov = (Xc.T * w) @ Xc / denom
        return cov + self.ridge * np.eye(X.shape[1])

    # ------------------------------------------------------------------
    # step 1: build local models
    def fit_local_models(self):
        self.local_models = []
        n_db = self.X_db.shape[0]
        K_eff = min(self.K, n_db)

        for design in self.designs:
            Y_emp = self.Y_emp_by_design[design]
            Y_sim = self.Y_db_by_design[design]

            scaler = StandardScaler()
            Y_sim_s = scaler.fit_transform(Y_sim)
            Y_emp_s = scaler.transform(Y_emp)

            knn = NearestNeighbors(n_neighbors=K_eff, metric=self.knn_metric)
            knn.fit(Y_sim_s)
            distances, indices = knn.kneighbors(Y_emp_s)

            for i in range(Y_emp.shape[0]):
                idx = indices[i]
                Xi = self.X_db[idx]
                di = distances[i]

                eps_i = float(np.max(di) + 1e-12)
                log_import = self.log_prior_db[idx] - self.log_prop_db[idx]

                logw = -0.5 * (di / eps_i) ** 2 + log_import
                logw -= logsumexp(logw)
                w = np.exp(logw)

                cov_i = self._weighted_cov(Xi, w)

                lm = {
                    "design": design,
                    "empirical_id": int(i),
                    "indices": idx,
                    "Xi": Xi,
                    "weights": w,
                    "cov": cov_i,
                    "radius_y": float(np.max(di)),
                    "mean_knn_dist": float(np.mean(di)),
                    "eps": eps_i,
                }
                lm["component_covs"] = self._component_covs_from_lm(
                    lm,
                    k_cov=5,
                    inflate=1.0,
                    ridge_rel=1e-2,
                )

                self.local_models.append(lm)

        return self.local_models

    # ------------------------------------------------------------------
    # step 2: local kernels and posterior on archive
    def _local_kernel_logpdf(self, X, lm, inflate=1.0, k_cov=5, ridge_rel=1e-2):
        """
        Local Gaussian mixture with component-specific covariances.
        Each center mu_k in lm["Xi"] gets not isotropic covariances estimated from k_cov
         nearest neighbours inside Xi, so curved / nonlinear structures
        are preserved much better than with one shared global covariance.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xi = np.asarray(lm["Xi"], dtype=float)
        w = np.asarray(lm["weights"], dtype=float)
        w = w / w.sum()

        n = Xi.shape[0]
        k_eff = min(max(2, k_cov), n)

        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(Xi)
        _, idx = nn.kneighbors(Xi)

        parts = []
        global_scale = np.trace(lm["cov"]) / self.d_x

        for j, (wk, mu) in enumerate(zip(w, Xi)):
            Xloc = Xi[idx[j]]

            # local empirical covariance around this center
            Xc = Xloc - Xloc.mean(axis=0, keepdims=True)
            cov_j = (Xc.T @ Xc) / max(len(Xloc) - 1, 1)

            # small ridge for stability
            cov_j = inflate * cov_j + ridge_rel * global_scale * np.eye(self.d_x)

            parts.append(
                np.log(wk + 1e-300) + self._safe_mvn_logpdf(X, mu, cov_j)
            )

        return logsumexp(np.column_stack(parts), axis=1)

    def evaluate_design_factor_logpdf(self, X, design, inflate=1.0):
        """
        Evaluate the design-specific surrogate factor at arbitrary X:
            L_e(theta) = average of local theta-space kernel mixtures for design e
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        lms = [lm for lm in self.local_models if lm["design"] == design]
        if len(lms) == 0:
            raise ValueError(f"No local models found for design {design}. Run fit_local_models().")

        local_logs = [self._local_kernel_logpdf(X, lm, inflate=inflate) for lm in lms]
        return logsumexp(np.column_stack(local_logs), axis=1) - np.log(len(lms))

    def compute_posterior_weights(self, inflate=1.0):
        if len(self.local_models) == 0:
            self.fit_local_models()

        logw = self.log_prior_db - self.log_prop_db

        for design in self.designs:
            lms = [lm for lm in self.local_models if lm["design"] == design]
            local_logs = [self._local_kernel_logpdf(self.X_db, lm, inflate=inflate) for lm in lms]
            logL = logsumexp(np.column_stack(local_logs), axis=1) - np.log(len(lms))
            logw = logw + logL

        logw -= logsumexp(logw)
        self.log_post_weights_db = logw
        self.post_weights_db = np.exp(logw)
        return self.post_weights_db

    # ------------------------------------------------------------------
    # posterior support, mode, density, smooth pseudo-samples
    def posterior_support(self):
        if self.post_weights_db is None:
            self.compute_posterior_weights()
        return self.X_db.copy(), self.post_weights_db.copy()

    def posterior_mode(self):
        if self.post_weights_db is None:
            self.compute_posterior_weights()
        j = int(np.argmax(self.post_weights_db))
        return self.X_db[j]

    def posterior_mode_from_db(self):
        if self.post_weights_db is None:
            self.compute_posterior_weights()
        j = int(np.argmax(self.post_weights_db))
        return self.X_db[j], float(self.post_weights_db[j])

    def posterior_pdf(self, X, cov_scale=1.0, return_log=False):
        """
        Continuous density obtained by KDE over the weighted archive support.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))

        if self.post_weights_db is None:
            self.compute_posterior_weights()

        w = np.asarray(self.post_weights_db, dtype=float)
        w = w / w.sum()

        cov = self._weighted_cov(self.X_db, w)
        cov = float(cov_scale) * cov

        log_parts = []
        for wk, mu in zip(w, self.X_db):
            log_parts.append(
                np.log(wk + 1e-300) + self._safe_mvn_logpdf(X, mu, cov)
            )

        log_pdf = logsumexp(np.column_stack(log_parts), axis=1)
        return log_pdf if return_log else np.exp(log_pdf)

    def sample_posterior_particles_smooth(  self,
                                            n_samples=1000,
                                            fallback_cov_scale=0.05,
                                            ):
        """
        Continuous pseudo-samples from the posterior using particle-specific,
        locally adaptive anisotropic covariances.

        Strategy
        --------
        1) sample archive indices with posterior weights
        2) assign each archive point its own covariance by aggregating the local
           component covariances from the local models in which it appears
        3) draw Gaussian perturbations around the selected archive centers

        Parameters
        ----------
        n_samples : int
            Number of pseudo-samples to generate.
        k_cov : int
            Number of neighbours inside each Xi cloud used to estimate each
            component covariance.
        inflate : float
            Multiplier for local covariances.
        ridge_rel : float
            Relative ridge added to each covariance.
        fallback_cov_scale : float
            Small global fallback covariance for archive points that never appear
            in any local model.

        Returns
        -------
        X_new : ndarray, shape (n_samples, d_x)
            Smoothed posterior pseudo-samples.
        """
        if self.post_weights_db is None:
            self.compute_posterior_weights()
        if len(self.local_models) == 0:
            self.fit_local_models()

        w_post = np.asarray(self.post_weights_db, dtype=float)
        w_post = w_post / w_post.sum()

        n_db = self.X_db.shape[0]

        # ------------------------------------------------------------
        # build one covariance per archive particle
        cov_num = np.zeros((n_db, self.d_x, self.d_x), dtype=float)
        cov_den = np.zeros(n_db, dtype=float)

        for lm in self.local_models:
            covs_lm = lm["component_covs"]
            for j_local, idx_db in enumerate(lm["indices"]):
                idx_db = int(idx_db)

                # weight local covariance contribution by the component weight
                alpha = float(lm["weights"][j_local])

                cov_num[idx_db] += alpha * covs_lm[j_local]
                cov_den[idx_db] += alpha

        # small global fallback
        fallback_cov = fallback_cov_scale * self._weighted_cov(self.X_db, w_post)

        covs_db = np.empty((n_db, self.d_x, self.d_x), dtype=float)
        for j in range(n_db):
            if cov_den[j] > 0:
                covs_db[j] = cov_num[j] / cov_den[j]
            else:
                covs_db[j] = fallback_cov

        # ------------------------------------------------------------
        # sample archive centers according to posterior weights
        idx = self.rng.choice(
            n_db,
            size=n_samples,
            replace=True,
            p=w_post,
        )

        # perturb each selected center with its own covariance
        X_new = np.empty((n_samples, self.d_x), dtype=float)
        for i, j in enumerate(idx):
            X_new[i] = self.rng.multivariate_normal(
                mean=self.X_db[j],
                cov=covs_db[j],
            )

        return X_new
    # ------------------------------------------------------------------
    # step 3: diagnostics
    def diagnostics(self):
        if len(self.local_models) == 0:
            self.fit_local_models()

        radii = np.array([lm["radius_y"] for lm in self.local_models], dtype=float)
        spreads = np.array([np.trace(lm["cov"]) for lm in self.local_models], dtype=float)

        return {
            "mean_radius": float(np.mean(radii)),
            "max_radius": float(np.max(radii)),
            "worst_id": int(np.argmax(radii)),
            "radii": radii,
            "spreads": spreads,
        }

    def _component_covs_from_lm(self, lm, k_cov=10, inflate=1.0, ridge_rel=1e-2):
        """
        Compute component-specific local covariances for the centers in lm["Xi"].

        Each center gets a covariance estimated from its k_cov nearest neighbours
        inside Xi, plus a small ridge for numerical stability.
        """
        Xi = np.asarray(lm["Xi"], dtype=float)
        n = Xi.shape[0]

        if n == 1:
            # degenerate case: fallback to lm covariance
            return [inflate * lm["cov"]]

        k_eff = min(max(2, k_cov), n)

        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(Xi)
        _, idx = nn.kneighbors(Xi)

        # use lm["cov"] only as a scale reference for the ridge
        global_scale = max(np.trace(lm["cov"]) / self.d_x, 1e-12)

        covs = []
        for neigh_idx in idx:
            Xloc = Xi[neigh_idx]
            Xc = Xloc - Xloc.mean(axis=0, keepdims=True)
            cov_j = (Xc.T @ Xc) / max(len(Xloc) - 1, 1)

            cov_j = inflate * cov_j + ridge_rel * global_scale * np.eye(self.d_x)
            covs.append(cov_j)

        return covs


    # ------------------------------------------------------------------
    # step 4: refinement
    def sample_from_worst_ball(self, n_new=100, inflate=1.0):
        """
        Targeted enrichment using the worst local KNN ball.

        Draw exactly n_new samples from a Gaussian mixture centered at the
        mapped-back neighbors of the worst local model.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()

        diag = self.diagnostics()
        lm = self.local_models[diag["worst_id"]]

        centers = np.asarray(lm["Xi"], dtype=float)
        mix_w = np.asarray(lm["weights"], dtype=float)
        mix_w = mix_w / mix_w.sum()
        cov_list = lm["component_covs"]

        comp_idx = self.rng.choice(len(centers), size=n_new, replace=True, p=mix_w)

        X_new = np.empty((n_new, self.d_x), dtype=float)
        for i, j in enumerate(comp_idx):
            X_new[i] = self.rng.multivariate_normal(
                mean=centers[j],
                cov=cov_list[j]*inflate,
            )

        log_parts = []
        for wk, mu, cov_j in zip(mix_w, centers, cov_list):
            log_parts.append(
                np.log(wk + 1e-300) + self._safe_mvn_logpdf(X_new, mu, cov_j*inflate)
            )
        logq_new = logsumexp(np.column_stack(log_parts), axis=1)
        info = {}
        return X_new, logq_new, info

    def sample_from_flagged_balls_fast(
            self,
            n_new=100,
            top_frac=0.20,
            inflate=1.0,
            k_cov=5,
            ridge_rel=1e-2,
            priority="radius_x_spread",
            prior_weight=0.05,
            min_balls=1,
            max_balls=None,
            exact_logq=True,
    ):
        """
        Faster sampler from flagged balls using:
        - cached component covariances
        - grouped Gaussian draws per component
        - optional exact log proposal density
        """
        if len(self.local_models) == 0:
            self.fit_local_models()

        n_loc = len(self.local_models)
        if n_loc == 0:
            raise ValueError("No local models available.")

        # ------------------------------------------------------------
        # 1) score and choose flagged balls
        radii = np.array([lm["radius_y"] for lm in self.local_models], dtype=float)
        spreads = np.array([np.trace(lm["cov"]) for lm in self.local_models], dtype=float)

        if priority == "radius":
            scores = radii
        elif priority == "spread":
            scores = spreads
        elif priority == "radius_x_spread":
            scores = radii * np.sqrt(np.maximum(spreads, 1e-12))
        else:
            raise ValueError("priority must be 'radius', 'spread', or 'radius_x_spread'.")

        scores = np.maximum(scores, 1e-12)

        n_flagged = max(min_balls, int(np.ceil(top_frac * n_loc)))
        if max_balls is not None:
            n_flagged = min(n_flagged, int(max_balls))
        n_flagged = min(n_flagged, n_loc)

        flagged_ids = np.argsort(scores)[-n_flagged:]
        flagged_ids = flagged_ids[np.argsort(scores[flagged_ids])[::-1]]
        flagged = [self.local_models[int(i)] for i in flagged_ids]

        ball_scores = scores[flagged_ids]
        ball_mix_w = ball_scores / ball_scores.sum()

        # ------------------------------------------------------------
        # 2) prepare flagged ball structures and cache covariances
        flagged_data = []
        for lm, bid in zip(flagged, flagged_ids):
            if "component_covs" not in lm:
                lm["component_covs"] = self._component_covs_from_lm(
                    lm,
                    k_cov=k_cov,
                    inflate=inflate,
                    ridge_rel=ridge_rel,
                )

            centers = np.asarray(lm["Xi"], dtype=float)
            center_w = np.asarray(lm["weights"], dtype=float)
            center_w = center_w / center_w.sum()
            cov_list = lm["component_covs"]

            flagged_data.append({
                "ball_id": int(bid),
                "centers": centers,
                "center_w": center_w,
                "cov_list": cov_list,
            })

        # ------------------------------------------------------------
        # 3) split prior / local allocations
        prior_weight = float(np.clip(prior_weight, 0.0, 1.0))
        n_prior = int(self.rng.binomial(n_new, prior_weight)) if prior_weight > 0 else 0
        n_local = n_new - n_prior

        local_counts = (
            self.rng.multinomial(n_local, ball_mix_w)
            if n_local > 0 else np.zeros(len(flagged_data), dtype=int)
        )

        X_parts = []
        source_labels = []

        # prior samples
        if n_prior > 0:
            X_prior = self._prior_rvs(n_prior)
            X_parts.append(X_prior)
            source_labels.extend(["prior"] * n_prior)

        # ------------------------------------------------------------
        # 4) local samples, grouped by selected component
        for cnt, fd in zip(local_counts, flagged_data):
            if cnt == 0:
                continue

            centers = fd["centers"]
            center_w = fd["center_w"]
            cov_list = fd["cov_list"]

            comp_idx = self.rng.choice(len(centers), size=cnt, replace=True, p=center_w)
            unique_comp, comp_counts = np.unique(comp_idx, return_counts=True)

            X_ball_parts = []
            for j, nj in zip(unique_comp, comp_counts):
                Xj = self.rng.multivariate_normal(
                    mean=centers[j],
                    cov=cov_list[j],
                    size=int(nj),
                )
                X_ball_parts.append(Xj)

            X_ball = np.vstack(X_ball_parts)
            X_parts.append(X_ball)
            source_labels.extend([f"ball_{fd['ball_id']}"] * X_ball.shape[0])

        if len(X_parts) == 0:
            X_new = np.empty((0, self.d_x), dtype=float)
            logq_new = np.zeros(0, dtype=float)
            info = {
                "flagged_ids": [],
                "n_flagged": 0,
                "ball_mix_w": np.array([], dtype=float),
                "local_counts": np.array([], dtype=int),
                "prior_weight": prior_weight,
                "n_prior": 0,
            }
            return X_new, logq_new, info

        X_new = np.vstack(X_parts)

        # shuffle
        perm = self.rng.permutation(X_new.shape[0])
        X_new = X_new[perm]
        source_labels = [source_labels[i] for i in perm]

        # ------------------------------------------------------------
        # 5) exact or approximate log proposal density
        if exact_logq:
            log_terms = []

            if prior_weight > 0:
                log_terms.append(
                    np.log(prior_weight + 1e-300) +
                    np.asarray(self.prior.logpdf(X_new), dtype=float)
                )

            local_weight_total = 1.0 - prior_weight
            if local_weight_total > 0:
                for w_ball, fd in zip(ball_mix_w, flagged_data):
                    centers = fd["centers"]
                    center_w = fd["center_w"]
                    cov_list = fd["cov_list"]

                    center_terms = []
                    for wk, mu, cov_j in zip(center_w, centers, cov_list):
                        center_terms.append(
                            np.log(wk + 1e-300) +
                            self._safe_mvn_logpdf(X_new, mu, cov_j)
                        )

                    log_q_ball = logsumexp(np.column_stack(center_terms), axis=1)
                    log_terms.append(
                        np.log(local_weight_total + 1e-300) +
                        np.log(w_ball + 1e-300) +
                        log_q_ball
                    )

            logq_new = logsumexp(np.column_stack(log_terms), axis=1)
        else:
            # cheap approximation
            if prior_weight > 0:
                logq_new = np.log(prior_weight + 1e-300) + np.asarray(self.prior.logpdf(X_new), dtype=float)
            else:
                logq_new = np.zeros(X_new.shape[0], dtype=float)

        info = {
            "flagged_ids": [fd["ball_id"] for fd in flagged_data],
            "n_flagged": int(n_flagged),
            "ball_mix_w": ball_mix_w,
            "local_counts": local_counts,
            "prior_weight": float(prior_weight),
            "n_prior": int(n_prior),
            "source_labels": source_labels,
        }

        return X_new, logq_new, info


    def append_to_archive(self, X_new, logq_new):
        X_new = np.asarray(X_new, dtype=float)
        logq_new = np.asarray(logq_new, dtype=float)

        if X_new.ndim != 2 or X_new.shape[1] != self.d_x:
            raise ValueError("X_new has wrong shape.")
        if logq_new.shape[0] != X_new.shape[0]:
            raise ValueError("logq_new length does not match X_new.")

        Y_new_by_design = self._simulate_all_designs(X_new)
        logp_new = np.asarray(self.prior.logpdf(X_new), dtype=float)

        self.X_db = np.vstack([self.X_db, X_new])
        self.log_prior_db = np.concatenate([self.log_prior_db, logp_new])
        self.log_prop_db = np.concatenate([self.log_prop_db, logq_new])

        for d in self.designs:
            self.Y_db_by_design[d] = np.vstack([self.Y_db_by_design[d], Y_new_by_design[d]])

    def _posterior_predictive_score(self, sliced_wasserstein_2, n_post=1000, cov_scale=0.5):
        """
        Observable predictive discrepancy:
            compare posterior predictive responses to empirical responses in Y-space.

        Returns
        -------
        score : float
            Average sqrt(sliced_wasserstein_2) across designs.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()
        if self.post_weights_db is None:
            self.compute_posterior_weights()


        X_post = self.sample_posterior_particles_smooth( n_samples=n_post, fallback_cov_scale=cov_scale )
        vals = []
        for design in self.designs:
            Y_pred = self._ensure_2d(self.model(X_post, design))
            Y_emp = self.Y_emp_by_design[design]
            vals.append(np.sqrt(sliced_wasserstein_2(
                np.asarray(Y_emp, dtype=float),
                np.asarray(Y_pred, dtype=float),
                n_proj=100
            )))
        return float(np.mean(vals))

    def adaptive_refine(
        self,
        sliced_wasserstein_2,
        max_iter=20,
        top_frac=0.20,
        n_new_per_iter=100,
        inflate=1.0,
        improve_tol=0.01,
        patience=4,
        min_iter=3,
        n_post_pred=1000,
        posterior_cov_scale=0.5,
        keep_best_state=True,
        true_target=None,
        verbose=True,
        visual_diagnostic=True,
    ):
        """
        Adaptive archive refinement with history keys compatible with the
        previous richer implementation.

        Parameters
        ----------
        sliced_wasserstein_2 : callable
            Function computing squared sliced Wasserstein distance between two clouds.
        """
        self.history = []
        best_pred_score = np.inf
        best_iter = None
        stagnant_rounds = 0
        prev_mean_radius = None
        prev_mode = None
        prev_Xi = None
        best_state = None

        for it in range(max_iter):
            # refresh local models and archive posterior
            self.fit_local_models()
            self.compute_posterior_weights(inflate=1.0)

            diag = self.diagnostics()
            radii = np.asarray(diag["radii"], dtype=float)

            mean_radius = float(np.mean(radii))
            max_radius = float(np.max(radii))
            mode_x, mode_w = self.posterior_mode_from_db()

            # predictive score
            pred_score = self._posterior_predictive_score(
                sliced_wasserstein_2=sliced_wasserstein_2,
                n_post=n_post_pred,
                cov_scale=posterior_cov_scale,
            )

            # posterior-to-posterior change
            Xi = self.sample_posterior_particles_smooth( n_samples=1000)

            theta_change = (
                np.nan if prev_Xi is None
                else float(np.sqrt(sliced_wasserstein_2(
                    np.asarray(prev_Xi, dtype=float),
                    np.asarray(Xi, dtype=float),
                    n_proj=100
                )))
            )

            # optional truth-based diagnostic
            if true_target is not None:
                TT = np.asarray(true_target, dtype=float)
                if TT.ndim == 1:
                    TT = TT.reshape(1, -1)

                if TT.shape[1] == Xi.shape[1]:
                    Wasser_Distance = float(np.sqrt(
                        sliced_wasserstein_2(TT, Xi, n_proj=200)
                    ))
                else:
                    Wasser_Distance = None
                    if verbose:
                        print(
                            f"[adaptive_refine] skipping truth-based Wasserstein: "
                            f"true_target dim={TT.shape[1]}, posterior dim={Xi.shape[1]}"
                        )
            else:
                Wasser_Distance = None

            rel_impr_radius = (
                np.nan if prev_mean_radius is None
                else (prev_mean_radius - mean_radius) / max(prev_mean_radius, 1e-12)
            )
            mode_shift = (
                np.nan if prev_mode is None
                else float(np.linalg.norm(mode_x - prev_mode))
            )

            n_flagged = max(1, int(np.ceil(top_frac * len(radii))))

            improved = pred_score < (best_pred_score - improve_tol)
            if improved:
                best_pred_score = pred_score
                best_iter = int(it)

                if keep_best_state:
                    best_state = {
                        "X_db": self.X_db.copy(),
                        "Y_db_by_design": {d: self.Y_db_by_design[d].copy() for d in self.designs},
                        "log_prior_db": self.log_prior_db.copy(),
                        "log_prop_db": self.log_prop_db.copy(),
                    }
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1

            stop = False
            reason = "continue"

            if it + 1 >= min_iter and stagnant_rounds >= patience:
                stop = True
                reason = "predictive stagnation"
            elif it == max_iter - 1:
                stop = True
                reason = "max_iter"

            self.history.append({
                "iteration": int(it),
                "db_size": int(self.X_db.shape[0]),
                "mean_radius": mean_radius,
                "max_radius": max_radius,
                "posterior_mode_weight": float(mode_w),
                "n_flagged": int(n_flagged),
                "rel_improvement_radius": float(rel_impr_radius) if not np.isnan(rel_impr_radius) else np.nan,
                "mode_shift": float(mode_shift) if not np.isnan(mode_shift) else np.nan,
                "theta_change": float(theta_change) if not np.isnan(theta_change) else np.nan,
                "predictive_score": float(pred_score),
                "best_predictive_score_so_far": float(best_pred_score),
                "best_iter_so_far": int(best_iter) if best_iter is not None else -1,
                "truth_wasserstein": Wasser_Distance,
                "stop": bool(stop),
                "stop_reason": reason,
            })

            if verbose:
                msg = (
                    f"[iter {it + 1}/{max_iter}] "
                    f"db={self.X_db.shape[0]} "
                    f"pred={pred_score:.4g} "
                    f"mean_rad={mean_radius:.4g} "
                    f"max_rad={max_radius:.4g} "
                    f"flags={n_flagged}"
                )
                if Wasser_Distance is not None:
                    msg += f" truth_W={Wasser_Distance:.4g}"
                print(msg)

            if stop:
                break


            if SAMPLE_WORST_RADIUS:

                X_new, logq_new, _  = self.sample_from_worst_ball(
                    n_new=n_new_per_iter,
                    inflate=inflate,
                )
            else:
                X_new, logq_new, _  = self.sample_from_flagged_balls_fast(n_new=n_new_per_iter,
                                                                          inflate=inflate, )


            if visual_diagnostic:
                plt.scatter(self.X_db[:, 0], self.X_db[:, 1], label='archive', alpha=0.1)
                plt.scatter(X_new[:, 0], X_new[:, 1], c='k', label='sample from flagged', marker='d', alpha=0.3)
                for idx_lm, lm in enumerate(self.local_models):
                    if idx_lm==0:
                        plt.scatter(lm['Xi'][:, 0],lm['Xi'][:, 1],label='knn ', c='b', alpha=0.6)
                    else:
                        plt.scatter(lm['Xi'][:, 0],lm['Xi'][:, 1], c='b', alpha=0.6)
                plt.scatter(true_target[:, 0], true_target[:, 1], s=20, c='r', label='target', marker='+', alpha=0.9)
                plt.grid()
                plt.legend()
                plt.show()

            self.append_to_archive(X_new, logq_new)

            prev_mean_radius = mean_radius
            prev_mode = mode_x.copy()
            prev_Xi = Xi.copy()

        if keep_best_state and best_state is not None:
            self.X_db = best_state["X_db"]
            self.Y_db_by_design = best_state["Y_db_by_design"]
            self.log_prior_db = best_state["log_prior_db"]
            self.log_prop_db = best_state["log_prop_db"]

        self.fit_local_models()
        self.compute_posterior_weights(inflate=1.0)

        return pd.DataFrame(self.history), self.diagnostics()



###--------------------
# -------------------- START -----------------------
###--------------------


# LOAD THE NEW CLASS AS SOLVER
SOLVER = SimpleAdaptiveKNNABC

# SOLVER CONFIG
top_frac = 0.5
n_new_per_iter= 500
posterior_cov_scale=0.01
ridge, seed = 1e-2, 123
min_iter, max_iter = 5, 100
inflate = 0.95
improve_tol = 0.001
patience = 4
n_post_pred = 100
Number_of_KNN = 100
Nemp= 200
Nemp_airmode = 500
Nsim = 500
Nsim_airmode = 5_000
xlim = [-10,  10]
ylim = [-10,  10]
visual_diagnostic  = True

def run_case_1_paraboloid(DGM=3):
    # -------------------- Problem 1 - paraboloid inverse problem
    _, Demp, _ = prepare_case(1, Nemp=10_000, Nsim=2, DGM=DGM)
    Theta_target = [D["theta"] for _, D in Demp.items()]

    M, Demp, Dsim = prepare_case(1, Nemp=Nemp, Nsim=Nsim, DGM=DGM)
    M_design, Y_emp_by_design, sim_db = adapt_case1_for_multidesign(M, Demp, Dsim)

    prior = UniformBoxPrior(low=[xlim[0], ylim[0]],
                            high=[xlim[1], ylim[1]])

    solver = SOLVER(
        model=M_design,
        Y_emp_by_design=Y_emp_by_design,
        prior=prior,
        sim_db=sim_db,
        K=Number_of_KNN,
        ridge=ridge,
        seed=seed,
    )

    hist, diag = solver.adaptive_refine(
        sliced_wasserstein_2=sliced_wasserstein_2,
        max_iter=max_iter,
        n_new_per_iter=n_new_per_iter,
        top_frac=top_frac,
        inflate=inflate,
        improve_tol=improve_tol,
        patience=patience,
        min_iter=min_iter,
        n_post_pred=n_post_pred,
        posterior_cov_scale=posterior_cov_scale,
        keep_best_state=True,
        true_target=Theta_target[0],   # works for case 1
        verbose=True,
        visual_diagnostic=visual_diagnostic,
    )

    # ------------------------------------------------------------
    # posterior samples
    X_post = solver.sample_posterior_particles_smooth( n_samples=2000)
    theta_true = np.asarray(Theta_target[0], dtype=float)
    if theta_true.ndim == 1:
        theta_true = theta_true.reshape(1, -1)

    plot_posterior_with_marginals(
        X_post,
        theta_true=theta_true,
        xlim=xlim,
        ylim=ylim,
        figsize=(7, 7),
        n_grid_1d=300,
        n_grid_2d=100,
        scatter_alpha=0.70,
        scatter_size=20,
        contour_levels=4,
        filled_levels=15,
        show_filled=True,
        show_contours=True,
        )
    plt.show()

    print("🎬 Generating posterior progression plot...")
    print("\n🎯 SUMMARY:")
    print(f"• Initial mean radius: {hist['mean_radius'].iloc[0]:.3f} → Final: {hist['mean_radius'].iloc[-1]:.3f}")
    if "max_radius" in hist.columns:
        print(f"• Initial max radius: {hist['max_radius'].iloc[0]:.3f} → Final: {hist['max_radius'].iloc[-1]:.3f}")
    if "db_size" in hist.columns:
        print(f"• Database: {hist['db_size'].iloc[0]} → {hist['db_size'].iloc[-1]} samples")
    print(f"• Iterations: {len(hist)} (stopped: {hist['stop_reason'].iloc[-1]})")

    mode_x, mode_w = solver.posterior_mode_from_db()
    print(f"• Final posterior mode: {mode_x}")
    print(f"• Final posterior mode weight: {mode_w:.4g}")

def run_case_2_airmod():
    # ------------------------------------------------------------
    # AIRMODE calibration workflow
    model_airmode, Y_emp_by_design, sim_db, prior = prepare_airmode_for_calibration(
                                                    Nemp=Nemp_airmode,
                                                    Nsim=Nsim_airmode,
                                                )

    # ---------- Paths, Load, Extract TMCMC ----------
    ref_tmcmc_data_path = "../resources/AIRMODE/data/reference_TMCMC_DLRAirmod.mat"
    tmobj = load_mat_any(str(ref_tmcmc_data_path))
    theta_tmcmc, src_path = find_theta_matrix_from_mat(tmobj)
    print(f"[TMCMC] extracted from '{src_path}' -> {theta_tmcmc.shape}")

    solver = SOLVER(
        model=model_airmode,
        Y_emp_by_design=Y_emp_by_design,
        prior=prior,
        sim_db=sim_db,
        K=Number_of_KNN,
        ridge=ridge,
        seed=seed,
    )

    # NOTE:
    # theta_tmcmc has 11 dims while your AIRMODE solver state may have 12 dims.
    # So keep true_target=None unless you explicitly align the coordinates.
    hist, diag = solver.adaptive_refine(
        sliced_wasserstein_2=sliced_wasserstein_2,
        max_iter=max_iter,
        n_new_per_iter=n_new_per_iter,
        top_frac=top_frac,
        inflate=inflate,
        improve_tol=improve_tol,
        patience=patience,
        min_iter=min_iter,
        n_post_pred=n_post_pred,
        posterior_cov_scale=posterior_cov_scale,
        keep_best_state=True,
        true_target=theta_tmcmc[:,:10],
        verbose=True,
        visual_diagnostic=visual_diagnostic
    )

    X_post =solver.sample_posterior_particles_smooth(n_samples=5_000)

    df_my_samples = pd.DataFrame(X_post, columns=[
        "theta1", "theta2", "theta3", "theta4", "theta5", "theta6",
        "theta7", "theta8", "theta9", "theta10", "theta11", "theta12" ])

    plot_airmode_reference_overlay(
                        my_samples=df_my_samples,
                        my_label="ABC-KDE-KNN posterior",
                        my_color="royalblue"  )

    print("\nAIRMODE SUMMARY")
    print(f"Initial mean radius: {hist['mean_radius'].iloc[0]:.4f}")
    print(f"Final mean radius:   {hist['mean_radius'].iloc[-1]:.4f}")
    print(f"Iterations:          {len(hist)}")
    print(f"Stop reason:         {hist['stop_reason'].iloc[-1]}")

    db_size_initial = sim_db["X"].shape[0]
    db_size_final = solver.X_db.shape[0]
    print(f"Database size:       {db_size_initial} -> {db_size_final}")
    print(f"X_post median:         {np.quantile(X_post,0.5, axis=0)}")
    print(f"X_post std:         {np.std(X_post, axis=0)}")
    # plotting
    pairs_2_plt = ((4, 5), (9, 10), (6, 7), (6, 8), (1, 11))
    plot_airmode_comparison_fancy(
        solver=solver,
        theta_tmcmc=theta_tmcmc,
        design="AIRMODE",
        plot_sim=False,
        n2plt=2000,
        pairs_theta=pairs_2_plt,
        pairs_y=((1, 2), (1, 3), (2, 3), (3, 4), (2, 5), (3, 9)),
    )


def run_EXP_ISRERM2026():
    run_case_2_airmod()

    run_case_1_paraboloid(DGM=1)
    run_case_1_paraboloid(DGM=2)
    run_case_1_paraboloid(DGM=3)



if __name__ == '__main__':

    run_EXP_ISRERM2026()