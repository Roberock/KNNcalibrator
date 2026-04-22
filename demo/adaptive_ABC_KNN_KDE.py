# Adaptive KNN-KDE Inverse Problem Solver (Modular Version)
# ========================================================
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from demo.helpers_ISRERM2026 import sliced_wasserstein_2

import matplotlib.pyplot as plt

class AdaptiveInverseKNNKDE:
    """
    Adaptive inverse solver for models M(theta, design) -> y.

    Methodological structure
    ------------------------
    1) Shared archive in parameter space:
       We maintain one global archive {theta_k}_{k=1}^N.
       Each theta_k is propagated through *all* designs.

    2) KNN in response space, design by design:
       For each empirical response y_i^e at design e, we find the K nearest
       simulated responses among {y_k^e}. Their indices map back to the same
       shared parameter particles theta_k.

    3) Local kernel reconstruction in parameter space:
       The mapped-back neighbors define a local Gaussian-kernel mixture in theta-space.
       This is a local conditional surrogate, not yet the final posterior.

    4) Design-wise kernel-ABC factors:
       For each design e, we average the local kernels across empirical replicates.
       This gives a design-specific pseudo-likelihood factor L_e(theta).

    5) Global posterior surrogate:
           p_hat(theta | D_emp) ∝ [pi(theta) / q(theta)] * Π_e L_e(theta)
       where q is the actual proposal density used to generate the archive particle.
       If the archive starts from the prior only, then q = pi and the ratio is 1.

    6) Adaptive enrichment:
       Large radii in response space, diffuse local covariance in parameter space,
       or low local ESS signal poor local support. We then sample new theta values
       from a mixture of flagged local kernels (plus a small prior component),
       simulate them under all designs, append them to the archive, and iterate.
    """

    def __init__(self,
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
        """  Parameters
            ----------
            model : callable
                Forward model with signature:
                    model(X, design) -> Y
                where:
                    X has shape (n, d_x)
                    Y has shape (n, d_y) or (n,)
            Y_emp_by_design : dict
                {design: empirical responses array of shape (m_e, d_y)}
            prior : object
                Prior distribution on theta with:
                    - logpdf(X)
                    - preferably rvs(size, random_state)
                and ideally attributes d_x or mean/cov for dimension inference.
            sim_db : dict or None
                Optional shared archive:
                {
                    "X": X_sim,                                # shape (N, d_x)
                    "Y_by_design": {design: Y_sim_design},     # each shape (N, d_y)
                    "log_prior": logpi,                        # optional, shape (N,)
                    "log_prop": logq                           # optional, shape (N,)
                }
            N0 : int
                Initial archive size if sim_db is None.
            K : int
                Number of nearest neighbors in response space.
            ridge : float
                Ridge added to local covariance matrices.
            seed : int
                Random seed.
            knn_metric : str
                Metric used in response space for KNN.
        """
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

        first_design = self.designs[0]
        self.d_y = self.Y_emp_by_design[first_design].shape[1]
        self.d_x = self._infer_dx()

        # Shared archive in parameter space
        self.X_db = None                  # shape (N, d_x)
        self.log_prior_db = None          # shape (N,)
        self.log_prop_db = None           # shape (N,)

        # Responses by design, same row indexing as X_db
        self.Y_db_by_design = {}          # design -> array shape (N, d_y)

        # Fitted local models from the latest pass
        self.local_models = []

        # Posterior weights on the archive from the latest pass
        self.log_design_factor_by_design = {}
        self.log_post_weights_db = None
        self.post_weights_db = None

        # Diagnostics / adaptive history
        self.history = []

        if sim_db is None:
            self._initialize_prior_db(N0=N0)
        else:
            self._load_shared_db(sim_db)

    # ------------------------------------------------------------------
    # Basic helpers
    def _infer_dx(self):
        if hasattr(self.prior, "d_x"):
            return int(self.prior.d_x)
        if hasattr(self.prior, "mean"):
            return int(np.asarray(self.prior.mean).shape[0])
        if hasattr(self.prior, "low") and hasattr(self.prior, "high"):
            return int(np.asarray(self.prior.low).shape[0])
        raise ValueError("Cannot infer d_x from prior.")

    @staticmethod
    def _ensure_2d(Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return Y

    def _prior_rvs(self, size):
        if hasattr(self.prior, "rvs"):
            try:
                X = self.prior.rvs(size=size, random_state=self.rng)
            except TypeError:
                X = self.prior.rvs(size=size)
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, self.d_x)
            return X

        if hasattr(self.prior, "mean") and hasattr(self.prior, "cov"):
            return np.asarray(
                self.rng.multivariate_normal(self.prior.mean, self.prior.cov, size=size),
                dtype=float,
            )

        raise ValueError("Prior must provide rvs() or mean/cov.")

    def _simulate_all_designs(self, X):
        """ Simulate all designs for the same shared particles X. """
        X = np.asarray(X, dtype=float)
        Y_by_design = {}
        for design in self.designs:
            Y_by_design[design] = self._ensure_2d(self.model(X, design))
        return Y_by_design

    def _initialize_prior_db(self, N0=1000):
        """ Initial archive: draw theta from the prior, then simulate all designs.
        Methodological note:  This is the correct archive structure for later multiplicative
        aggregation across designs, because every row corresponds to the same  theta_k evaluated under all designs."""
        X = self._prior_rvs(N0)
        Y_by_design = self._simulate_all_designs(X)
        logp = np.asarray(self.prior.logpdf(X), dtype=float)

        self.X_db = X
        self.Y_db_by_design = Y_by_design
        self.log_prior_db = logp
        self.log_prop_db = logp.copy()

    def _load_shared_db(self, sim_db):
        X = np.asarray(sim_db["X"], dtype=float)
        Y_by_design = {
            d: self._ensure_2d(sim_db["Y_by_design"][d]) for d in self.designs
        }

        n = X.shape[0]
        for d in self.designs:
            if Y_by_design[d].shape[0] != n:
                raise ValueError(
                    f"Design {d} has {Y_by_design[d].shape[0]} rows, expected {n}."
                )

        log_prior = np.asarray(
            sim_db.get("log_prior", self.prior.logpdf(X)), dtype=float
        )
        log_prop = np.asarray(
            sim_db.get("log_prop", log_prior.copy()), dtype=float
        )

        if log_prior.shape[0] != n or log_prop.shape[0] != n:
            raise ValueError("log_prior/log_prop lengths do not match X.")

        self.X_db = X
        self.Y_db_by_design = Y_by_design
        self.log_prior_db = log_prior
        self.log_prop_db = log_prop

    @staticmethod
    def _weighted_cov(X, w, ridge):
        """
        Weighted covariance with a small ridge for numerical stability.
        """
        w = np.asarray(w, dtype=float)
        w = w / max(np.sum(w), 1e-300)
        mu = np.sum(X * w[:, None], axis=0)
        Xc = X - mu
        denom = max(1e-12, 1.0 - np.sum(w ** 2))
        cov = (Xc.T * w) @ Xc / denom
        return cov + ridge * np.eye(X.shape[1])

    @staticmethod
    def _safe_mvn_logpdf(X, mean, cov):
        """ Small wrapper to tolerate near-singular local covariances. """
        return multivariate_normal(mean=mean, cov=cov, allow_singular=True).logpdf(X)

    # ------------------------------------------------------------------
    # Local model fitting
    def fit_local_models(self):
        """
        Fit one local kernel model for each empirical response and design.

        For each (design e, empirical sample i):
        - find K nearest simulated responses in Y-space,
        - map indices back to shared theta-particles,
        - build local weights using a Gaussian kernel in response space and
          archive importance correction,
        - estimate a local covariance in parameter space.

        Methodological note:
         Efficient version:
        - batched KNN per design
        - batched local weights
        - batched weighted covariances via einsum

        Methodological note
        -------------------
        KNN is performed in scaled response space, design by design.
        The resulting distances, eps, and radii are therefore measured
        in that scaled space.

        eps_i is chosen adaptively as the K-th nearest-neighbor distance.
        Dense regions imply small eps_i; sparse regions imply large eps_i.
        Large eps_i is therefore a diagnostic of poor local archive support.
        """
        if self.X_db is None or self.X_db.shape[0] == 0:
            raise ValueError("Archive is empty.")

        n_db = self.X_db.shape[0]
        K_eff = min(self.K, n_db)
        self.local_models = []

        for design in self.designs:
            Y_emp = self.Y_emp_by_design[design]   # (n_emp, d_y)
            Y_sim = self.Y_db_by_design[design]    # (n_db, d_y)
            # ----------------------------------------------------------
            # 1) Scale response space per design
            scaler = StandardScaler()
            Y_sim_scaled = scaler.fit_transform(Y_sim)
            Y_emp_scaled = scaler.transform(Y_emp)
            # ----------------------------------------------------------
            # 2) KNN for all empirical responses at once
            knn = NearestNeighbors(n_neighbors=K_eff, metric=self.knn_metric)
            knn.fit(Y_sim_scaled)
            distances, indices = knn.kneighbors(Y_emp_scaled)
            # distances: (n_emp, K_eff)
            # indices:   (n_emp, K_eff)
            # ----------------------------------------------------------
            # 3) Gather corresponding parameter neighbors
            X_neighbors = self.X_db[indices]   # (n_emp, K_eff, d_x)
            # ----------------------------------------------------------
            # 4) Local bandwidth / radius in scaled response space
            eps = np.max(distances, axis=1) + 1e-12                 # (n_emp,)
            mean_knn_dist = np.mean(distances, axis=1)              # (n_emp,)
            radius_y = np.max(distances, axis=1)                    # (n_emp,)
            # ----------------------------------------------------------
            # 5) Importance correction for archive particles
            log_import = self.log_prior_db[indices] - self.log_prop_db[indices]  # (n_emp, K_eff)
            # ----------------------------------------------------------
            # 6) Batched local response-kernel weights
            logw = -0.5 * (distances / eps[:, None]) ** 2 + log_import
            logw -= logsumexp(logw, axis=1, keepdims=True)
            W = np.exp(logw)                                        # (n_emp, K_eff)
            # ----------------------------------------------------------
            # 7) Batched ESS
            ess = 1.0 / np.maximum(np.sum(W ** 2, axis=1), 1e-300)  # (n_emp,)
            # ----------------------------------------------------------
            # 8) Batched weighted covariance in parameter space
            mu = np.sum(X_neighbors * W[:, :, None], axis=1)  # Weighted mean: (n_emp, d_x)
            Xc = X_neighbors - mu[:, None, :]  # Centered neighbors: (n_emp, K_eff, d_x)

            # Unbiased-ish denominator matching the single-model helper
            denom = np.maximum(1e-12, 1.0 - np.sum(W ** 2, axis=1))  # (n_emp,)

            # Batched covariance:
            # covs[n] = sum_k W[n,k] * Xc[n,k]^T Xc[n,k] / denom[n]
            covs = np.einsum("nk,nki,nkj->nij", W, Xc, Xc) / denom[:, None, None]

            # Ridge stabilization
            covs += self.ridge * np.eye(self.d_x)[None, :, :]

            # ----------------------------------------------------------
            # 9) Lightweight bookkeeping loop
            n_emp = Y_emp.shape[0]
            for i in range(n_emp):
                self.local_models.append({
                    "local_id": len(self.local_models),
                    "design": design,
                    "empirical_id": int(i),
                    "indices": indices[i],          # global archive indices
                    "Xi": X_neighbors[i],           # (K_eff, d_x)
                    "weights": W[i],                # (K_eff,)
                    "cov": covs[i],                 # (d_x, d_x)
                    "radius_y": float(radius_y[i]),
                    "mean_knn_dist": float(mean_knn_dist[i]),
                    "eps": float(eps[i]),
                    "ess": float(ess[i]),
                })

        """ # Example - visualize 
        import matplotlib.pyplot as plt

        plt.scatter(self.X_db[:, 0], self.X_db[:, 1], 2, color='k', alpha=0.1)

        for c, XI in enumerate(self.local_models):
            if c < n_emp:
                col = 'red'
            elif c >= n_emp and c < 2 * n_emp:
                col = 'blue'
            elif c < 3 * n_emp and c >= 2 * n_emp:
                col = 'green'
            else:
                col = 'purple'
            plt.scatter(XI['Xi'][:, 0], XI['Xi'][:, 1], 15, color=col)
        plt.show()
        """

        return self.local_models

    def fit_local_models_inefficiently(self):

        if self.X_db is None or len(self.X_db) == 0:
            raise ValueError("Archive is empty.")

        n_db = self.X_db.shape[0]
        K_eff = min(self.K, n_db)
        self.local_models = []

        scaler = StandardScaler()

        for design in self.designs:
            Y_emp = self.Y_emp_by_design[design]
            Y_sim = self.Y_db_by_design[design]
            Y_sim_scaled = scaler.fit_transform(Y_sim)
            Y_emp_scaled = scaler.transform(Y_emp)
            knn = NearestNeighbors(n_neighbors=K_eff, metric=self.knn_metric)
            knn.fit(Y_sim_scaled)
            distances, indices = knn.kneighbors(Y_emp_scaled)

            for i in range(Y_emp.shape[0]):
                idx = indices[i]
                Xi = self.X_db[idx]
                di = np.asarray(distances[i], dtype=float)

                # Response-space tolerance = K-th neighbor distance
                eps_i = float(np.max(di) + 1e-12)

                # Importance correction for source archive particles:
                # if some archive samples came from adaptive proposals rather than
                # the prior, they should not dominate the local kernel construction.
                log_import_source = self.log_prior_db[idx] - self.log_prop_db[idx]

                # Local response kernel weights
                logw = -0.5 * (di / eps_i) ** 2 + log_import_source
                logw -= logsumexp(logw)
                w = np.exp(logw)

                cov_i = self._weighted_cov(Xi, w, self.ridge)
                ess_i = 1.0 / max(np.sum(w ** 2), 1e-300)

                self.local_models.append({
                    "local_id": len(self.local_models),
                    "design": design,
                    "empirical_id": int(i),
                    "indices": idx,            # indices in the shared archive
                    "Xi": Xi,                  # mapped-back parameter neighbors
                    "weights": w,              # local mixture weights
                    "cov": cov_i,              # local theta-space covariance
                    "radius_y": float(np.max(di)),
                    "mean_knn_dist": float(np.mean(di)),
                    "eps": float(eps_i),
                    "ess": float(ess_i),
                })

        return self.local_models

    # ------------------------------------------------------------------
    # Local kernels and design factors
    def _local_kernel_logpdf(self, X, lm, inflate=1.0):
        """
        Log-density of a local kernel mixture in theta-space:
            p_hat_i^e(theta) = sum_k w_k N(theta ; theta_k, inflate * Sigma_i^e)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        cov = inflate * lm["cov"]

        parts = []
        for w, mu in zip(lm["weights"], lm["Xi"]):
            parts.append(np.log(w + 1e-300) + self._safe_mvn_logpdf(X, mu, cov))
        return logsumexp(np.column_stack(parts), axis=1)

    def _local_models_for_design(self, design):
        """ Returns the local KDE -KNN models for the selected design """
        return [lm for lm in self.local_models if lm["design"] == design]

    def evaluate_design_factor_logpdf(self, X, design):
        """ Evaluate the design-specific pseudo-likelihood factor:
            L_e(theta) = (1 / N_e) sum_i p_hat_i^e(theta)
        Methodological note:
            This is the design-wise kernel-ABC factor. The final posterior is  obtained only after multiplying these factors
            across designs and combining with pi(theta)/q(theta).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        lms = self._local_models_for_design(design)
        if len(lms) == 0:
            raise ValueError(f"No local models found for design {design}. Did you run fit_local_models()?")

        local_logs = [self._local_kernel_logpdf(X, lm) for lm in lms]
        return logsumexp(np.column_stack(local_logs), axis=1) - np.log(len(lms))

    def compute_log_posterior_weights_on_archive(self):
        """ Compute normalized posterior weights on the current shared archive.
            Formula:
                log w_k ∝ [log pi(theta_k) - log q(theta_k)] + sum_e log L_e(theta_k)
            Methodological note:
            - L_e is the design-wise pseudo-likelihood factor.
            - pi/q is the archive importance correction.
            - The nearest neighbors themselves are not posterior draws.
              They only define local kernel support.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()

        n = self.X_db.shape[0]
        log_design_factor_by_design = {}
        total = self.log_prior_db - self.log_prop_db

        for design in self.designs:
            logL_e = self.evaluate_design_factor_logpdf(self.X_db, design)
            log_design_factor_by_design[design] = logL_e
            total = total + logL_e

        total = total - logsumexp(total)

        self.log_design_factor_by_design = log_design_factor_by_design
        self.log_post_weights_db = total
        self.post_weights_db = np.exp(total)
        return self.log_post_weights_db

    def posterior_weights_on_archive(self):
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()
        return self.post_weights_db.copy()

    def posterior_particles(self, n_samples=None, replace=True):
        """ Resample an unweighted particle cloud from the weighted archive posterior """
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        if n_samples is None:
            n_samples = self.X_db.shape[0]

        idx = self.rng.choice(
            self.X_db.shape[0],
            size=int(n_samples),
            replace=replace,
            p=self.post_weights_db,
        )
        return self.X_db[idx]

    def posterior_mean(self):
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()
        return np.average(self.X_db, axis=0, weights=self.post_weights_db)

    def posterior_mode_from_db(self):
        """ Return the highest-weight archive particle."""
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()
        j = int(np.argmax(self.post_weights_db))
        return self.X_db[j], float(self.post_weights_db[j])

    def posterior_pdf(self, X, smooth=True):
        """ Optional smooth posterior approximation for plotting.
            If smooth=True, we build a weighted mixture over archive particles using
            design-local covariances attached to the local kernels that contain them.
            This is mainly for visualization on low-dimensional synthetic examples.

            If smooth=False, this returns a simple weighted archive KDE using a single
            global covariance estimated from the posterior-weighted archive.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        # Simpler and stabler default: weighted global KDE from posterior archive
        if not smooth or len(self.local_models) == 0:
            cov = self._weighted_cov(self.X_db, self.post_weights_db, self.ridge)
            parts = [
                np.log(w + 1e-300) + self._safe_mvn_logpdf(X, mu, cov)
                for w, mu in zip(self.post_weights_db, self.X_db)
            ]
            return np.exp(logsumexp(np.column_stack(parts), axis=1))

        # Slightly richer visualization: attach to each archive point the average
        # covariance of local kernels in which it appears.
        covs_per_idx = [[] for _ in range(self.X_db.shape[0])]
        for lm in self.local_models:
            for idx in lm["indices"]:
                covs_per_idx[int(idx)].append(lm["cov"])

        cov_list = []
        for j in range(self.X_db.shape[0]):
            if len(covs_per_idx[j]) == 0:
                cov_list.append(self._weighted_cov(self.X_db, self.post_weights_db, self.ridge))
            else:
                cov_list.append(np.mean(covs_per_idx[j], axis=0) + self.ridge * np.eye(self.d_x))

        parts = [
            np.log(w + 1e-300) + self._safe_mvn_logpdf(X, self.X_db[j], cov_list[j])
            for j, w in enumerate(self.post_weights_db)
        ]
        return np.exp(logsumexp(np.column_stack(parts), axis=1))

    # ------------------------------------------------------------------
    # Diagnostics
    def diagnostics0(self):
        """  Local diagnostics used to decide where enrichment is needed.
            Priority combines:
            - large response-space radius
            - large local spread in parameter space
            - low local ESS
            These are heuristics, but they align well with the intended adaptive logic:
            large radii and diffuse local kernels mean the local inverse map is poorly resolved.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()

        radii = np.array([lm["radius_y"] for lm in self.local_models], dtype=float)
        spread_x = np.array([np.trace(lm["cov"]) for lm in self.local_models], dtype=float)
        ess = np.array([lm["ess"] for lm in self.local_models], dtype=float)

        r_med = max(np.median(radii), 1e-12)
        s_med = max(np.median(spread_x), 1e-12)
        u = self.K / np.maximum(ess, 1.0)
        u_med = max(np.median(u), 1e-12)

        qrad = max(np.quantile(radii, 0.5), 1e-12)
        # qspr = max(np.quantile(spread_x, 0.75), 1e-12)

        # priority = ( radii / qrad   + spread_x / qspr   + (0.4 * self.K) / np.maximum(ess, 1e-12)  )
        priority = (    radii / qrad   )


        # local posterior mass carried by neighbors
        """mass = np.array([
            np.sum(self.post_weights_db[lm["indices"]]) if self.post_weights_db is not None else 1.0
            for lm in self.local_models
        ], dtype=float)

        priority = (mass + 1e-12) ** 0.5 * (
                0.45 * (radii / r_med) +
                0.30 * (spread_x / s_med) +
                0.25 * (u / u_med)
        )"""

        return pd.DataFrame({
            "local_id": [lm["local_id"] for lm in self.local_models],
            "design": [lm["design"] for lm in self.local_models],
            "empirical_id": [lm["empirical_id"] for lm in self.local_models],
            "radius_y": radii,
            "mean_knn_dist": [lm["mean_knn_dist"] for lm in self.local_models],
            "eps": [lm["eps"] for lm in self.local_models],
            "spread_x_trace": spread_x,
            "ess": ess,
            "priority": priority,
        })

    def diagnostics(self):
        """
        Local diagnostics used for adaptive refinement.

        Revised priority:
          - response-space radius
          - theta-space spread
          - low local ESS
          - local posterior mass carried by the corresponding archive neighbors

        The mass term prevents spending simulations on diffuse but irrelevant regions.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        radii = np.array([lm["radius_y"] for lm in self.local_models], dtype=float)
        spread_x = np.array([np.trace(lm["cov"]) for lm in self.local_models], dtype=float)
        ess = np.array([lm["ess"] for lm in self.local_models], dtype=float)

        # local posterior mass carried by the archive particles in each local kernel
        mass = np.array([
            np.sum(self.post_weights_db[lm["indices"]]) for lm in self.local_models
        ], dtype=float)

        # robust normalizers
        r_med = max(np.median(radii), 1e-12)
        s_med = max(np.median(spread_x), 1e-12)
        u = self.K / np.maximum(ess, 1.0)
        u_med = max(np.median(u), 1e-12)

        # revised composite priority
        priority = (mass + 1e-12) ** 0.5 * (
                0.45 * (radii / r_med) +
                0.30 * (spread_x / s_med) +
                0.25 * (u / u_med)
        )

        return pd.DataFrame({
            "local_id": [lm["local_id"] for lm in self.local_models],
            "design": [lm["design"] for lm in self.local_models],
            "empirical_id": [lm["empirical_id"] for lm in self.local_models],
            "radius_y": radii,
            "mean_knn_dist": [lm["mean_knn_dist"] for lm in self.local_models],
            "eps": [lm["eps"] for lm in self.local_models],
            "spread_x_trace": spread_x,
            "ess": ess,
            "mass": mass,
            "priority": priority,
        })
    # ------------------------------------------------------------------
    # Adaptive enrichment
    def _build_flagged_mixture(self, flagged_local_ids):
        flagged = [self.local_models[int(fid)] for fid in flagged_local_ids]
        if len(flagged) == 0:
            raise ValueError("No flagged local models provided.")

        # Allocate more probability to poorly supported regions
        raw = np.array(
            [lm["radius_y"] * np.trace(lm["cov"]) / max(lm["ess"], 1e-12) for lm in flagged],
            dtype=float,
        )
        raw = np.maximum(raw, 1e-12)
        mix_w = raw / raw.sum()
        return flagged, mix_w

    def _proposal_logpdf(self, X, flagged, mix_w, prior_weight=0.10, inflate=1.0):
        """ Log-density of the targeted proposal mixture:
                q_new(theta) = alpha0 * pi(theta) + sum_j alpha_j * q_j(theta)

            where q_j are flagged local kernel mixtures.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        parts = [np.log(prior_weight + 1e-300) + np.asarray(self.prior.logpdf(X), dtype=float)]

        local_weight_scale = max(1.0 - prior_weight, 1e-12)
        for a, lm in zip(mix_w, flagged):
            parts.append(np.log(local_weight_scale * a + 1e-300) + self._local_kernel_logpdf(X, lm, inflate=inflate))

        return logsumexp(np.column_stack(parts), axis=1)

    def sample_from_posterior_kde(self, n_samples=1000, smooth=True):
        """ Sample from a continuous posterior KDE approximation.
        If smooth=False:  Uses a global Gaussian KDE:
                p_hat(theta|D) ≈ sum_k w_k N(theta ; X_db[k], Sigma_global)
        If smooth=True:  Uses a locally smoothed Gaussian mixture:
                p_hat(theta|D) ≈ sum_k w_k N(theta ; X_db[k], Sigma_k)
        Returns -->   X_new : ndarray, shape (n_samples, d_x)  """
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        n = self.X_db.shape[0]
        idx = self.rng.choice(n, size=int(n_samples), replace=True, p=self.post_weights_db)

        # ----------------------------------------------------------
        # Global KDE: same covariance for all particles
        if not smooth or len(self.local_models) == 0:
            cov = self._weighted_cov(self.X_db, self.post_weights_db, self.ridge)
            X_new = np.zeros((n_samples, self.d_x), dtype=float)
            for i, j in enumerate(idx):
                X_new[i] = self.rng.multivariate_normal(self.X_db[j], cov)
            return X_new

        # ----------------------------------------------------------
        # Local KDE: covariance attached to each archive particle
        covs_per_idx = [[] for _ in range(n)]
        for lm in self.local_models:
            for j in lm["indices"]:
                covs_per_idx[int(j)].append(lm["cov"])

        global_cov = self._weighted_cov(self.X_db, self.post_weights_db, self.ridge)

        cov_list = []
        for j in range(n):
            if len(covs_per_idx[j]) == 0:
                cov_list.append(global_cov)
            else:
                cov_list.append(np.mean(covs_per_idx[j], axis=0) + self.ridge * np.eye(self.d_x))

        X_new = np.zeros((n_samples, self.d_x), dtype=float)
        for i, j in enumerate(idx):
            X_new[i] = self.rng.multivariate_normal(self.X_db[j], cov_list[j])

        return X_new

    def _tempered_post_weights(self, temp=0.7):
        """
        Tempered posterior weights on the archive:
            w_tilde_k ∝ (w_k)^temp
        with temp in (0,1] for flattening / more exploration.
        """
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        w = np.asarray(self.post_weights_db, dtype=float)
        w = np.maximum(w, 1e-300)
        w = w ** float(temp)
        w /= np.sum(w)
        return w

    def _posterior_kde_cov(self, cov_scale=1.0):
        """
        Global KDE covariance built from the weighted archive posterior.
        """
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        cov = self._weighted_cov(self.X_db, self.post_weights_db, self.ridge)
        return float(cov_scale) * cov

    def _posterior_kde_logpdf(self, X, temp=0.7, cov_scale=1.0):
        """
        Continuous KDE approximation of the archive posterior:
            q_post(theta) = sum_k w_tilde_k N(theta ; X_db[k], Sigma_global)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        w = self._tempered_post_weights(temp=temp)
        cov = self._posterior_kde_cov(cov_scale=cov_scale)

        parts = []
        for wk, mu in zip(w, self.X_db):
            parts.append(np.log(wk + 1e-300) + self._safe_mvn_logpdf(X, mu, cov))
        return logsumexp(np.column_stack(parts), axis=1)

    def _sample_from_tempered_posterior_kde(self, n_new=100, temp=0.7, cov_scale=1.0):
        """
        Draw from the continuous KDE approximation of the current posterior.
        """
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        w = self._tempered_post_weights(temp=temp)
        cov = self._posterior_kde_cov(cov_scale=cov_scale)

        idx = self.rng.choice(self.X_db.shape[0], size=int(n_new), replace=True, p=w)
        X_new = np.zeros((int(n_new), self.d_x), dtype=float)
        for i, j in enumerate(idx):
            X_new[i] = self.rng.multivariate_normal(mean=self.X_db[j], cov=cov)

        logq = self._posterior_kde_logpdf(X_new, temp=temp, cov_scale=cov_scale)
        return X_new, logq

    def _local_centers(self):
        """
        Weighted local means in theta-space for each local model.
        """
        centers = []
        for lm in self.local_models:
            mu = np.sum(lm["Xi"] * lm["weights"][:, None], axis=0)
            centers.append(mu)
        return np.asarray(centers, dtype=float)

    def _select_diverse_flagged(self, diag, n_select=3, sep_scale=0.5):
        """
        Greedy diversity filter on flagged local models in theta-space.
        Prevents refining repeatedly on the same ridge / branch.

        Parameters
        ----------
        diag : DataFrame
            Must already be sorted descending by priority.
        n_select : int
            Number of flagged locals to retain.
        sep_scale : float
            Separation threshold as a fraction of the median local scale.
        """
        if len(diag) == 0:
            return np.array([], dtype=int)

        centers = self._local_centers()
        local_ids = diag["local_id"].to_numpy(dtype=int)

        # local scale based on average posterior spread
        local_scales = np.array([
            np.sqrt(max(np.trace(lm["cov"]), 1e-12)) for lm in self.local_models
        ], dtype=float)
        sep = float(sep_scale) * np.median(local_scales)

        chosen = []
        chosen_centers = []

        for lid in local_ids:
            c = centers[lid]
            if len(chosen_centers) == 0:
                chosen.append(lid)
                chosen_centers.append(c)
            else:
                dmin = np.min([np.linalg.norm(c - cc) for cc in chosen_centers])
                if dmin >= sep:
                    chosen.append(lid)
                    chosen_centers.append(c)

            if len(chosen) >= n_select:
                break

        # if diversity filter was too strict, fill from top priority
        if len(chosen) < n_select:
            for lid in local_ids:
                if lid not in chosen:
                    chosen.append(int(lid))
                if len(chosen) >= n_select:
                    break

        return np.asarray(chosen, dtype=int)

    def _posterior_predictive_score(self, n_post=1000, use_kde=False, temp=0.7):
        """
        Observable predictive discrepancy:
            compare posterior predictive responses to empirical responses in Y-space.

        Returns
        -------
        score : float
            Average sqrt(sliced_wasserstein_2) across designs.
            Smaller is better.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        if use_kde:
            X_post, _ = self._sample_from_tempered_posterior_kde(
                n_new=n_post,
                temp=temp,
                cov_scale=1.0
            )
        else:
            X_post = self.posterior_particles(n_samples=n_post, replace=True)

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


    def append_to_archive(self, X_new, logq_new):
        """
        Append new shared particles and simulate them under all designs.
        """
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

    def adaptive_refine_v0(self,
                        max_iter=10,
                        top_frac=0.30,
                        n_new_per_iter=100,
                        inflate=1.0,
                        prior_weight=0.10,
                        improve_tol=0.05,
                        patience=3,
                        target_shrink=0.60,
                        min_iter=2,
                        true_target=None):

        """  Run the adaptive loop.
            Stopping logic is based on:
            - reduction in mean and max response-space radii
            - stagnation in these diagnostics
        """
        self.history = []
        stagnant_rounds = 0

        base_mean_radius = None
        base_max_radius = None
        prev_mean_radius = None
        prev_mode = None

        for it in range(max_iter):
            # Refit local models and posterior on current archive
            self.fit_local_models()
            self.compute_log_posterior_weights_on_archive()
            diag = self.diagnostics().sort_values("priority", ascending=False).reset_index(drop=True)
            mean_radius = float(diag["radius_y"].mean())
            max_radius = float(diag["radius_y"].max())
            mode_x, mode_w = self.posterior_mode_from_db()

            if it == 0:
                base_mean_radius = mean_radius
                base_max_radius = max_radius

            rel_impr = (  np.nan  if prev_mean_radius is None
                          else (prev_mean_radius - mean_radius) / max(prev_mean_radius, 1e-12)  )
            mode_shift = (  np.nan if prev_mode is None
                            else float(np.linalg.norm(mode_x - prev_mode))             )

            n_flagged = max(1, int(np.ceil(top_frac * len(diag))))
            flagged_local_ids = diag["local_id"].iloc[:n_flagged].to_numpy()

            reached_target = ( (mean_radius <= target_shrink * base_mean_radius) and
                               (max_radius <= target_shrink * base_max_radius) )
            stalled = ( prev_mean_radius is not None and
                        rel_impr < improve_tol and
                        (np.isnan(mode_shift) or mode_shift < 1e-6) )

            stagnant_rounds = stagnant_rounds + 1 if stalled else 0
            stop = False
            reason = "continue"
            if it + 1 >= min_iter and reached_target:
                stop, reason = True, "radius target reached"
            elif it + 1 >= min_iter and stagnant_rounds >= patience:
                stop, reason = True, "stagnation"
            elif it == max_iter - 1:
                stop, reason = True, "max_iter"

            if true_target is not None:
                Xi_posterior = self.sample_from_posterior_kde(n_samples=10_000, smooth=True)
                Wasser_Distance = np.sqrt(sliced_wasserstein_2(np.asarray(true_target, dtype=float), Xi_posterior, n_proj=200))
                plt.scatter(true_target[:, 0], true_target[:, 1], 1, alpha=0.1)
                plt.scatter(Xi_posterior[:, 0], Xi_posterior[:, 1], alpha=0.1)
                plt.show()
            else:
                Wasser_Distance = None
                """
                plt.scatter(true_target[:,0],true_target[:,1], 1,  alpha =0.1)
                plt.scatter(self.X_db[self.post_weights_db>1e-4,0],self.X_db[self.post_weights_db>1e-4,1]) 
                plt.scatter(self.X_db[self.post_weights_db>0.001,0],self.X_db[self.post_weights_db>0.001,1]) 
                plt.scatter(self.X_db[self.post_weights_db>0.02,0],self.X_db[self.post_weights_db>0.02,1]) 
                plt.show()
                
                Xi_posterior = self.sample_from_posterior_kde(n_samples=10_000, smooth=False)
                plt.scatter(true_target[:, 0], true_target[:, 1], 1, alpha=0.1)
                plt.scatter(Xi_posterior[:, 0], Xi_posterior[:, 1], alpha=0.1)
                plt.show()
                """



            self.history.append({
                "iteration": int(it),
                "db_size": int(self.X_db.shape[0]),
                "mean_radius": mean_radius,
                "max_radius": max_radius,
                "posterior_mode_weight": float(mode_w),
                "n_flagged": int(n_flagged),
                "rel_improvement": float(rel_impr) if not np.isnan(rel_impr) else np.nan,
                "mode_shift": float(mode_shift) if not np.isnan(mode_shift) else np.nan,
                "stop": bool(stop),
                "stop_reason": reason,
                "Xi_best95": self.X_db[self.post_weights_db > np.quantile(self.post_weights_db,0.95), :],
            })

            print(
                f"[iter {it + 1}/{max_iter}] db={self.X_db.shape[0]} "
                f"mean_radius={mean_radius:.4g} "
                f"max_radius={max_radius:.4g}"
                f" flagged={n_flagged}"
                f"Wasserstein -> (Post||Target): {Wasser_Distance:.4f}"
            )

            if stop:
                break

            # Sample globally in theta-space, then simulate all designs
            X_new, logq_new = self.sample_targeted_batch(
                flagged_local_ids=flagged_local_ids,
                n_new=n_new_per_iter,
                inflate=inflate,
                prior_weight=prior_weight,
            )

            # try sample_from_worst_ball

            self.append_to_archive(X_new, logq_new)

            prev_mean_radius = mean_radius
            prev_mode = mode_x.copy()

        # Final refresh after the last append
        self.fit_local_models()
        self.compute_log_posterior_weights_on_archive()

        return pd.DataFrame(self.history), self.diagnostics()

    def adaptive_refine(
            self,
            max_iter=20,
            top_frac=0.20,
            n_new_per_iter=100,
            inflate=1.5,
            prior_weight=0.20,
            posterior_weight=0.50,
            local_weight=0.30,
            posterior_temp=0.70,
            improve_tol=0.01,
            patience=4,
            min_iter=3,
            n_post_pred=1000,
            keep_best_state=True,
            true_target=None,
    ):
        """
        Adaptive refinement driven by an observable predictive criterion.

        Main score:
            predictive_score_t
            = average response-space discrepancy between empirical data and
              posterior predictive simulations across designs.

        This is stable in practice because it does not require the true target
        in theta-space.

        Additional monitors:
            - mean / max response-space radius
            - posterior change between iterations in theta-space
            - optional truth-based diagnostic if true_target is available

        Returns
        -------
        hist : DataFrame
        diag : DataFrame
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
            # ----------------------------------------------
            # refresh local models and archive posterior
            self.fit_local_models()
            self.compute_log_posterior_weights_on_archive()
            diag = self.diagnostics().sort_values("priority", ascending=False).reset_index(drop=True)

            mean_radius = float(diag["radius_y"].mean())
            max_radius = float(diag["radius_y"].max())
            mode_x, mode_w = self.posterior_mode_from_db()

            # posterior predictive score (observable)
            pred_score = self._posterior_predictive_score(
                n_post=n_post_pred,
                use_kde=False,
                temp=posterior_temp,
            )

            # posterior-to-posterior change (proxy stability)
            Xi = self.posterior_particles(n_samples=2000, replace=True)
            theta_change = (
                np.nan if prev_Xi is None
                else float(np.sqrt(sliced_wasserstein_2(
                    np.asarray(prev_Xi, dtype=float),
                    np.asarray(Xi, dtype=float),
                    n_proj=100
                )))
            )

            # optional truth-based diagnostic for synthetic examples
            if true_target is not None:
                Wasser_Distance = float(np.sqrt(
                    sliced_wasserstein_2(np.asarray(true_target, dtype=float), Xi, n_proj=200)
                ))
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

            # ----------------------------------------------
            # best predictive checkpoint
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

            # ----------------------------------------------
            # diverse flagged locals
            n_flagged = max(1, int(np.ceil(top_frac * len(diag))))
            flagged_local_ids = self._select_diverse_flagged(diag, n_select=n_flagged, sep_scale=0.5)

            # ----------------------------------------------
            # stopping
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
                "n_flagged": int(len(flagged_local_ids)),
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

            print(
                f"[iter {it + 1}/{max_iter}] "
                f"db={self.X_db.shape[0]} "
                f"pred={pred_score:.4g} "
                f"mean_radius={mean_radius:.4g} "
                f"max_radius={max_radius:.4g} "
                f"flagged={len(flagged_local_ids)} "
                f"truth_W={Wasser_Distance if Wasser_Distance is not None else 'NA'}"
            )

            if stop:
                break

            # ----------------------------------------------
            # stable 3-way enrichment
            """
            X_new, logq_new = self.sample_targeted_batch(
                flagged_local_ids=flagged_local_ids,
                n_new=n_new_per_iter,
                inflate=inflate,
                prior_weight=prior_weight,
                posterior_weight=posterior_weight,
                local_weight=local_weight,
                posterior_temp=posterior_temp,
                posterior_cov_scale=1.0,
            )
            """

            X_new, logq_new, _ = self.sample_from_worst_ball(n_per_center=n_new_per_iter,
                                                      inflate=inflate, kernel="gaussian",
                                                      use_radius_only=True)

            self.append_to_archive(X_new, logq_new)

            prev_mean_radius = mean_radius
            prev_mode = mode_x.copy()
            prev_Xi = Xi.copy()
            if true_target is not None:
                plt.scatter(true_target[:, 0], true_target[:, 1], 1, alpha=0.1, c='b')
                plt.scatter(Xi[:, 0], Xi[:, 1], 10, alpha=0.1, c='r')
                plt.scatter(X_new[:, 0], X_new[:, 1], 100, alpha=0.1, c='k')
                plt.show()

        # ----------------------------------------------
        # restore best predictive state if requested
        if keep_best_state and best_state is not None:
            self.X_db = best_state["X_db"]
            self.Y_db_by_design = best_state["Y_db_by_design"]
            self.log_prior_db = best_state["log_prior_db"]
            self.log_prop_db = best_state["log_prop_db"]

        # final refresh
        self.fit_local_models()
        self.compute_log_posterior_weights_on_archive()

        return pd.DataFrame(self.history), self.diagnostics()


    # ------------------------------------------------------------------
    # Convenience summaries
    def archive_summary(self):
        return {
            "n_particles": int(self.X_db.shape[0]),
            "d_x": int(self.d_x),
            "d_y": int(self.d_y),
            "n_designs": int(len(self.designs)),
            "designs": list(self.designs),
        }

    def design_factor_on_archive(self):
        """
        Return a DataFrame with one column per design factor evaluated on the archive.
        """
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        out = pd.DataFrame({"idx": np.arange(self.X_db.shape[0])})
        for d in self.designs:
            out[f"logL_{d}"] = self.log_design_factor_by_design[d]
        out["log_prior_minus_log_prop"] = self.log_prior_db - self.log_prop_db
        out["log_post_weight"] = self.log_post_weights_db
        out["post_weight"] = self.post_weights_db
        return out

    def sample_from_worst_ball(
            self,
            n_per_center=20,
            inflate=1.0,
            kernel="gaussian",
            use_radius_only=True,
    ):
        """
        Simple refinement:
        1) pick the empirical response with the largest KNN ball
        2) take its K reconstructed input neighbours
        3) sample around each of those K input points
        4) return the proposal log-density as well
        """
        self.fit_local_models()

        diag = self.diagnostics().copy()
        sort_col = "radius_y" if use_radius_only else "priority"
        diag = diag.sort_values(sort_col, ascending=False).reset_index(drop=True)

        worst_local_id = int(diag.iloc[0]["local_id"])
        lm = self.local_models[worst_local_id]

        centers = lm["Xi"]  # shape (K, d_x)
        cov = inflate * lm["cov"]

        X_new_parts = []

        if kernel == "gaussian":
            for mu in centers:
                X_new_parts.append(
                    self.rng.multivariate_normal(mu, cov, size=n_per_center)
                )
            X_new = np.vstack(X_new_parts)

            # equal-weight Gaussian mixture log-density
            log_parts = []
            mix_w = 1.0 / len(centers)
            for mu in centers:
                log_parts.append(
                    np.log(mix_w) + self._safe_mvn_logpdf(X_new, mu, cov)
                )
            logq_new = logsumexp(np.column_stack(log_parts), axis=1)

        elif kernel == "uniform_box":
            std = np.sqrt(np.diag(cov))
            for mu in centers:
                U = self.rng.uniform(
                    low=mu - std,
                    high=mu + std,
                    size=(n_per_center, self.d_x),
                )
                X_new_parts.append(U)

            X_new = np.vstack(X_new_parts)

            # simple fallback: ignore proposal correction
            logq_new = np.asarray(self.prior.logpdf(X_new), dtype=float)

        else:
            raise ValueError("kernel must be 'gaussian' or 'uniform_box'")

        return X_new, logq_new, worst_local_id

    def continuous_log_posterior(self, X, logq_fn=None):
        """
        Continuous posterior surrogate evaluated at arbitrary theta.

        Target:
            log p_hat(theta | D) =
                log prior(theta)
                + sum_e log L_e(theta)
                - log q(theta)    [optional correction]

        Parameters
        ----------
        X : array, shape (n, d_x) or (d_x,)
            Evaluation points in parameter space.
        logq_fn : callable or None
            Optional callable:
                logq_fn(X) -> array shape (n,)
            representing the continuous proposal density correction.
            If None, no proposal correction is applied.

        Returns
        -------
        logp : ndarray, shape (n,)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))

        if len(self.local_models) == 0:
            self.fit_local_models()

        logp = np.asarray(self.prior.logpdf(X), dtype=float).reshape(-1)

        for design in self.designs:
            logp = logp + self.evaluate_design_factor_logpdf(X, design)

        if logq_fn is not None:
            logp = logp - np.asarray(logq_fn(X), dtype=float).reshape(-1)

        return logp


    def sample_targeted_batch_v0( self,  flagged_local_ids,
                               n_new=100,  inflate=1.0, prior_weight=0.10):
        """
        Draw new theta particles from a mixture of flagged local kernels plus a
        small prior component.

        Methodological note:
        This proposal keeps a nonzero prior component for global coverage, while
        concentrating most samples in under-supported local neighborhoods.
        """
        flagged, mix_w = self._build_flagged_mixture(flagged_local_ids)

        # Sample source components
        source_u = self.rng.uniform(size=n_new)
        X_new = np.zeros((n_new, self.d_x), dtype=float)

        prior_mask = source_u < prior_weight
        n_prior = int(prior_mask.sum())
        if n_prior > 0:
            X_new[prior_mask] = self._prior_rvs(n_prior)

        n_local = n_new - n_prior
        if n_local > 0:
            chosen_flag = self.rng.choice(len(flagged), size=n_local, p=mix_w)
            rows_local = np.where(~prior_mask)[0]

            for row, j in zip(rows_local, chosen_flag):
                lm = flagged[int(j)]
                comp = self.rng.choice(len(lm["weights"]), p=lm["weights"])
                mu = lm["Xi"][comp]
                cov = inflate * lm["cov"]
                # , n_sam_per_local: int =10 consider more than one sample per local
                #  X_new[row] = self.rng.multivariate_normal(mu, cov, n_sam_per_local)
                X_new[row] = self.rng.multivariate_normal(mu, cov)

        logq_new = self._proposal_logpdf(
            X_new,
            flagged=flagged,
            mix_w=mix_w,
            prior_weight=prior_weight,
            inflate=inflate,
        )
        return X_new, logq_new

    def sample_targeted_batch_v1( self, flagged_local_ids, n_new=100,  inflate=1.0,
                                        prior_weight=0.10,
                                        n_rep_per_choice=100, ):
        """
        Draw new theta particles from a mixture of flagged local kernels plus a
        small prior component.

        n_new is the total number of returned particles.
        n_rep_per_choice controls how many iid Gaussian draws are taken after one
        source choice (prior or local kernel). If > 1, the method becomes more
        exploitative locally.
        """
        if n_rep_per_choice < 1:
            raise ValueError("n_rep_per_choice must be >= 1")

        flagged, mix_w = self._build_flagged_mixture(flagged_local_ids)

        # Number of source choices needed to produce about n_new samples
        n_choices = int(np.ceil(n_new / n_rep_per_choice))
        source_u = self.rng.uniform(size=n_choices)
        prior_mask = source_u < prior_weight

        X_parts = []

        # Prior draws
        n_prior_choices = int(prior_mask.sum())
        if n_prior_choices > 0:
            Xp = self._prior_rvs(n_prior_choices * n_rep_per_choice)
            X_parts.append(Xp)

        # Local draws
        n_local_choices = n_choices - n_prior_choices
        if n_local_choices > 0:
            chosen_flag = self.rng.choice(len(flagged), size=n_local_choices, p=mix_w)

            for j in chosen_flag:
                lm = flagged[int(j)]
                comp = self.rng.choice(len(lm["weights"]), p=lm["weights"])
                mu = lm["Xi"][comp]
                cov = inflate * lm["cov"]
                X_loc = self.rng.multivariate_normal(mean=mu,  cov=cov,  size=n_rep_per_choice)
                X_parts.append(np.atleast_2d(X_loc))

        if len(X_parts) == 0:
            X_new = np.empty((0, self.d_x), dtype=float)
            logq_new = np.empty((0,), dtype=float)
            return X_new, logq_new

        X_new = np.vstack(X_parts)

        # Trim to exactly n_new samples
        if X_new.shape[0] > n_new:
            X_new = X_new[:n_new]

        logq_new = self._proposal_logpdf(    X_new,
                                         flagged=flagged,   mix_w=mix_w,
                                             prior_weight=prior_weight,    inflate=inflate, )
        return X_new, logq_new


    def sample_targeted_batch(
            self,
            flagged_local_ids,
            n_new=100,            inflate=1.5,
            prior_weight=0.20,
            posterior_weight=0.50,
            local_weight=0.30,
            posterior_temp=0.70,
            posterior_cov_scale=1.0,
    ):
        """
        Stable 3-way proposal:
            q_t(theta)
            = alpha * prior
            + beta  * tempered posterior KDE
            + gamma * flagged local mixtures

        This is substantially more stable than refining only the worst balls.

        Parameters
        ----------
        flagged_local_ids : array-like
            Local IDs selected for targeted repair.
        n_new : int
            Number of new particles.
        inflate : float
            Inflation of local covariances for flagged proposals.
        prior_weight, posterior_weight, local_weight : float
            Mixture weights. They will be renormalized internally.
        posterior_temp : float
            Temperature for posterior KDE proposal.
        posterior_cov_scale : float
            Scale factor for posterior KDE covariance.
        """
        weights = np.array([prior_weight, posterior_weight, local_weight], dtype=float)
        weights = np.maximum(weights, 1e-12)
        weights /= np.sum(weights)

        alpha, beta, gamma = weights

        n_prior = self.rng.multinomial(int(n_new), weights)

        X_parts = []
        logq_parts = []

        # ----------------------------------------------------------
        # 1) Prior / global exploration
        if n_prior[0] > 0:
            Xp = self._prior_rvs(n_prior[0])
            logq_p = np.asarray(self.prior.logpdf(Xp), dtype=float)
            X_parts.append(Xp)
            logq_parts.append(logq_p)

        # ----------------------------------------------------------
        # 2) Tempered posterior KDE exploration
        if n_prior[1] > 0:
            Xpost, logq_post = self._sample_from_tempered_posterior_kde(
                n_new=n_prior[1],
                temp=posterior_temp,
                cov_scale=posterior_cov_scale,
            )
            X_parts.append(Xpost)
            logq_parts.append(logq_post)

        # ----------------------------------------------------------
        # 3) Targeted local repair
        if n_prior[2] > 0 and len(flagged_local_ids) > 0:
            flagged, mix_w = self._build_flagged_mixture(flagged_local_ids)

            Xloc = np.zeros((n_prior[2], self.d_x), dtype=float)
            chosen_flag = self.rng.choice(len(flagged), size=n_prior[2], p=mix_w)

            for row, j in enumerate(chosen_flag):
                lm = flagged[int(j)]
                comp = self.rng.choice(len(lm["weights"]), p=lm["weights"])
                mu = lm["Xi"][comp]
                cov = inflate * lm["cov"]
                Xloc[row] = self.rng.multivariate_normal(mu, cov)

            logq_loc = self._proposal_logpdf(
                Xloc,
                flagged=flagged,
                mix_w=mix_w,
                prior_weight=0.0,  # local term only here; full mixture corrected below
                inflate=inflate,
            )

            X_parts.append(Xloc)
            logq_parts.append(logq_loc)

        # ----------------------------------------------------------
        # Stack proposed particles
        if len(X_parts) == 0:
            return np.empty((0, self.d_x)), np.empty((0,))

        X_new = np.vstack(X_parts)

        # ----------------------------------------------------------
        # Full mixture proposal correction
        # q(theta) = alpha*pi + beta*q_post + gamma*q_local
        parts = []

        if alpha > 0:
            parts.append(np.log(alpha + 1e-300) + np.asarray(self.prior.logpdf(X_new), dtype=float))

        if beta > 0:
            parts.append(np.log(beta + 1e-300) +
                         self._posterior_kde_logpdf(X_new, temp=posterior_temp, cov_scale=posterior_cov_scale))

        if gamma > 0 and len(flagged_local_ids) > 0:
            flagged, mix_w = self._build_flagged_mixture(flagged_local_ids)
            parts.append(np.log(gamma + 1e-300) +
                         self._proposal_logpdf(
                             X_new,
                             flagged=flagged,
                             mix_w=mix_w,
                             prior_weight=0.0,
                             inflate=inflate,
                         ))

        logq_new = logsumexp(np.column_stack(parts), axis=1)
        return X_new, logq_new


    def sample_from_posterior_mcmc(
            self,
            n_samples=2000,
            burn_in=1000,
            thin=1,
            x0=None,
            proposal_cov=None,
            proposal_scale=1.0,
            logq_fn=None,
    ):
        """
        Random-walk Metropolis sampler for the continuous posterior surrogate.

        Parameters
        ----------
        n_samples : int    Number of retained posterior samples.
        burn_in : int    Number of burn-in iterations.
        thin : int  Keep one sample every `thin` steps after burn-in.
        x0 : array-like or None   Initial point. If None, starts from the archive posterior mode.
        proposal_cov : ndarray or None
            Proposal covariance for the Gaussian random walk.
            If None, uses the posterior-weighted archive covariance.
        proposal_scale : float
            Global multiplier on proposal_cov.
        logq_fn : callable or None
            Optional continuous proposal correction log-density.
            If None, samples from the uncorrected continuous surrogate.

        Returns -------
        X_chain : ndarray, shape (n_samples, d_x)   Posterior MCMC samples.
        info : dict      Diagnostics such as acceptance rate.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()
        if self.post_weights_db is None:
            self.compute_log_posterior_weights_on_archive()

        # Start at archive posterior mode if not provided
        if x0 is None:
            x0, _ = self.posterior_mode_from_db()

        x_curr = np.asarray(x0, dtype=float).reshape(-1)

        # Default proposal covariance from weighted archive posterior
        if proposal_cov is None:
            proposal_cov = self._weighted_cov(self.X_db, self.post_weights_db, self.ridge)

        proposal_cov = proposal_scale * np.asarray(proposal_cov, dtype=float)

        n_total = int(burn_in + n_samples * thin)
        chain = np.zeros((n_samples, self.d_x), dtype=float)

        logp_curr = self.continuous_log_posterior(x_curr[None, :], logq_fn=logq_fn)[0]

        n_accept = 0
        save_id = 0

        for t in range(n_total):
            x_prop = self.rng.multivariate_normal(mean=x_curr, cov=proposal_cov)
            logp_prop = self.continuous_log_posterior(x_prop[None, :], logq_fn=logq_fn)[0]

            log_alpha = logp_prop - logp_curr
            if np.log(self.rng.uniform()) < log_alpha:
                x_curr = x_prop
                logp_curr = logp_prop
                n_accept += 1

            if t >= burn_in and ((t - burn_in) % thin == 0):
                chain[save_id] = x_curr
                save_id += 1

        info = {
            "accept_rate": n_accept / max(n_total, 1),
            "x0": np.asarray(x0, dtype=float),
            "proposal_cov": proposal_cov,
            "burn_in": int(burn_in),
            "thin": int(thin),
            "n_total_steps": int(n_total),
        }
        return chain, info



class UniformBoxPrior:
    """
    Simple independent uniform prior on a box [low, high].
    """
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.d_x = self.low.shape[0]
        self.mean = (self.low + self.high) / 2.0
        var = (self.high - self.low) ** 2 / 12.0
        self.cov = np.diag(var)

    def pdf(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        inside = np.all((X >= self.low) & (X <= self.high), axis=1)
        vol = np.prod(self.high - self.low)
        out = np.zeros(X.shape[0], dtype=float)
        out[inside] = 1.0 / vol
        return out

    def logpdf(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        inside = np.all((X >= self.low) & (X <= self.high), axis=1)
        vol = np.prod(self.high - self.low)
        out = np.full(X.shape[0], -np.inf, dtype=float)
        out[inside] = -np.log(vol)
        return out

    def rvs(self, size=1, random_state=None):
        rng = (
            np.random.default_rng(random_state)
            if not isinstance(random_state, np.random.Generator)
            else random_state
        )
        return rng.uniform(self.low, self.high, size=(size, self.d_x))




class SimpleAdaptiveKNNABC:
    """
    Minimal adaptive KNN-ABC / KNN-KDE inverse solver.

    Core idea
    ---------
    1) Keep a shared archive of theta particles.
    2) For each design and empirical observation:
         - find K nearest simulated responses,
         - map them back to theta-space,
         - build a local weighted Gaussian mixture.
    3) Combine all local mixtures design-by-design to get posterior weights
       on the archive.
    4) Refine by sampling around the worst KNN ball.

    Assumptions
    -----------
    - model(X, design) -> array of simulated outputs
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
        self.Y_emp_by_design = {
            d: self._ensure_2d(Y_emp_by_design[d]) for d in self.designs
        }

        self.d_x = self._infer_dx()
        self.X_db = None
        self.Y_db_by_design = {}
        self.log_prior_db = None
        self.log_prop_db = None

        self.local_models = []
        self.log_post_weights_db = None
        self.post_weights_db = None

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
        denom = max(1e-12, 1.0 - np.sum(w**2))
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

                self.local_models.append({
                    "design": design,
                    "empirical_id": i,
                    "indices": idx,
                    "Xi": Xi,
                    "weights": w,
                    "cov": cov_i,
                    "radius_y": float(np.max(di)),
                    "eps": eps_i,
                })

        return self.local_models

    # ------------------------------------------------------------------
    # step 2: posterior on archive
    def _local_kernel_logpdf(self, X, lm, inflate=1.0):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        cov = inflate * lm["cov"]

        parts = []
        for wk, mu in zip(lm["weights"], lm["Xi"]):
            parts.append(np.log(wk + 1e-300) + self._safe_mvn_logpdf(X, mu, cov))
        return logsumexp(np.column_stack(parts), axis=1)

    def compute_posterior_weights(self, inflate=1.0):
        if not self.local_models:
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

    def posterior_particles(self, n_samples=1000):
        if self.post_weights_db is None:
            self.compute_posterior_weights()
        idx = self.rng.choice(
            self.X_db.shape[0],
            size=n_samples,
            replace=True,
            p=self.post_weights_db,
        )
        return self.X_db[idx]

    def posterior_pdf(self, X, cov_scale=1.0, return_log=False):
        """
        Continuous posterior density from the weighted archive support.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))

        if self.post_weights_db is None:
            self.compute_posterior_weights()

        w = np.asarray(self.post_weights_db, dtype=float)
        w = w / np.sum(w)

        # global weighted covariance of archive support
        cov = self._weighted_cov(self.X_db, w)
        cov = float(cov_scale) * cov

        log_parts = []
        for wk, mu in zip(w, self.X_db):
            log_parts.append(
                np.log(wk + 1e-300) + self._safe_mvn_logpdf(X, mu, cov)
            )

        log_pdf = logsumexp(np.column_stack(log_parts), axis=1)
        return log_pdf if return_log else np.exp(log_pdf)

    def sample_posterior_particles_smooth(self, n_samples=1000, cov_scale=1.0):
        if self.post_weights_db is None:
            self.compute_posterior_weights()

        w = np.asarray(self.post_weights_db, dtype=float)
        w = w / w.sum()

        idx = self.rng.choice(self.X_db.shape[0], size=n_samples, replace=True, p=w)

        cov = self._weighted_cov(self.X_db, w)
        cov = cov_scale * cov

        X_new = np.empty((n_samples, self.d_x), dtype=float)
        for i, j in enumerate(idx):
            X_new[i] = self.rng.multivariate_normal(self.X_db[j], cov)

        return X_new


    def posterior_mode(self):
        if self.post_weights_db is None:
            self.compute_posterior_weights()
        j = int(np.argmax(self.post_weights_db))
        return self.X_db[j]

    # ------------------------------------------------------------------
    # step 3: refinement
    def diagnostics(self):
        if not self.local_models:
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

    def sample_from_worst_ball(self, n_per_center=20, inflate=1.0):
        """
        Simplest targeted enrichment:
        - find local model with largest radius
        - sample around its K mapped-back neighbours
        """
        if not self.local_models:
            self.fit_local_models()

        diag = self.diagnostics()
        lm = self.local_models[diag["worst_id"]]

        centers = lm["Xi"]
        cov = inflate * lm["cov"]

        X_new_parts = []
        for mu in centers:
            X_new_parts.append(
                self.rng.multivariate_normal(mu, cov, size=n_per_center)
            )
        X_new = np.vstack(X_new_parts)

        # equal-weight Gaussian mixture proposal density
        log_parts = []
        mix_w = 1.0 / len(centers)
        for mu in centers:
            log_parts.append(np.log(mix_w) + self._safe_mvn_logpdf(X_new, mu, cov))
        logq_new = logsumexp(np.column_stack(log_parts), axis=1)

        return X_new, logq_new

    def append_to_archive(self, X_new, logq_new):
        X_new = np.asarray(X_new, dtype=float)
        logq_new = np.asarray(logq_new, dtype=float)

        Y_new_by_design = self._simulate_all_designs(X_new)
        logp_new = np.asarray(self.prior.logpdf(X_new), dtype=float)

        self.X_db = np.vstack([self.X_db, X_new])
        self.log_prior_db = np.concatenate([self.log_prior_db, logp_new])
        self.log_prop_db = np.concatenate([self.log_prop_db, logq_new])

        for d in self.designs:
            self.Y_db_by_design[d] = np.vstack([self.Y_db_by_design[d], Y_new_by_design[d]])

    def posterior_pdf(self, X, cov_scale=1.0):
        """
        Smooth posterior density obtained by KDE over the weighted archive posterior.

        Parameters
        ----------
        X : array-like, shape (n_eval, d_x) or (d_x,)
            Evaluation points.
        cov_scale : float
            Multiplier for the posterior covariance. Increase it if the density
            looks too spiky, decrease it if it looks too blurred.

        Returns
        -------
        pdf : ndarray, shape (n_eval,)
            Estimated posterior density at the evaluation points.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))

        if self.post_weights_db is None:
            self.compute_posterior_weights()

        # global weighted covariance of the posterior support
        cov = self._weighted_cov(self.X_db, self.post_weights_db)
        cov = float(cov_scale) * cov

        parts = []
        for w, mu in zip(self.post_weights_db, self.X_db):
            parts.append(
                np.log(w + 1e-300) + self._safe_mvn_logpdf(X, mu, cov)
            )

        return np.exp(logsumexp(np.column_stack(parts), axis=1))


    def _posterior_predictive_score(self, n_post=1000):
        """
        Observable predictive discrepancy:
            compare posterior predictive responses to empirical responses in Y-space.

        Returns
        -------
        score : float
            Average sqrt(sliced_wasserstein_2) across designs.
            Smaller is better.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()
        if self.post_weights_db is None:
            self.compute_posterior_weights()

        X_post = self.posterior_particles(n_samples=n_post)

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

    def log_posterior_surrogate(self, X, inflate=1.0):
        """
        Unnormalized log surrogate posterior at arbitrary X:
            log pi(theta) - log q(theta) + sum_e log L_e(theta)

        In the simplified setting we usually do not know log q(theta) away from
        the archive, so this version uses only log prior + surrogate factors.

        If your prior archive is all prior-drawn, this is perfectly consistent
        with the simple implementation.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))

        if len(self.local_models) == 0:
            self.fit_local_models()

        logp = np.asarray(self.prior.logpdf(X), dtype=float)
        for design in self.designs:
            logp = logp + self.evaluate_design_factor_logpdf(X, design, inflate=inflate)
        return logp

    def _local_center(self, lm):
        """
        Weighted local mean in theta-space.
        """
        return np.sum(lm["Xi"] * lm["weights"][:, None], axis=0)

    def sample_mcmc_from_flagged(
            self,
            n_chains=5,
            steps_per_chain=50,
            burnin=20,
            top_frac=0.2,
            proposal_scale=0.5,
            inflate=1.0,
            start_mode="mean",
            keep_every=5,
            return_all_samples=False,
    ):
        """
        Run short random-walk MH chains from flagged local regions.

        Parameters
        ----------
        n_chains : int
            Number of flagged local regions / chains to use.
        steps_per_chain : int
            MH steps per chain.
        burnin : int
            Number of initial iterations to discard.
        top_frac : float
            Fraction of worst local models (by radius) considered as candidates.
        proposal_scale : float
            Proposal covariance multiplier:
                theta' ~ N(theta, proposal_scale^2 * cov_local)
        inflate : float
            Inflation used in the surrogate posterior local kernels.
        start_mode : {"mean", "neighbor"}
            How to initialize each chain:
            - "mean": weighted local mean
            - "neighbor": randomly chosen mapped-back neighbor
        keep_every : int
            Thinning after burnin.
        return_all_samples : bool
            If True, return all kept MCMC samples.
            If False, return one final sample per chain.

        Returns
        -------
        X_new : ndarray, shape (n_samples, d_x)
            New candidate particles.
        logq_new : ndarray, shape (n_samples,)
            Approximate proposal log-density based on the flagged-start mixture.
            This is a rough approximation, good enough for archive bookkeeping.
        info : dict
            Diagnostics such as acceptance rates and selected local ids.
        """
        if len(self.local_models) == 0:
            self.fit_local_models()

        # ------------------------------------------------------------
        # 1) select flagged locals by radius
        radii = np.array([lm["radius_y"] for lm in self.local_models], dtype=float)
        n_pool = max(1, int(np.ceil(top_frac * len(radii))))
        cand_ids = np.argsort(radii)[-n_pool:]  # worst radii
        cand_ids = cand_ids[::-1]  # descending order

        if len(cand_ids) == 0:
            raise ValueError("No candidate flagged locals found.")

        chosen_ids = cand_ids[: min(n_chains, len(cand_ids))]
        flagged = [self.local_models[int(i)] for i in chosen_ids]

        # ------------------------------------------------------------
        # 2) starting points and proposal covariances
        starts = []
        prop_covs = []
        centers = []

        for lm in flagged:
            center = self._local_center(lm)
            centers.append(center)

            if start_mode == "mean":
                x0 = center.copy()
            elif start_mode == "neighbor":
                j = self.rng.choice(len(lm["Xi"]), p=lm["weights"])
                x0 = lm["Xi"][j].copy()
            else:
                raise ValueError("start_mode must be 'mean' or 'neighbor'.")

            starts.append(x0)
            prop_covs.append((proposal_scale ** 2) * lm["cov"])

        starts = np.asarray(starts, dtype=float)
        centers = np.asarray(centers, dtype=float)

        # ------------------------------------------------------------
        # 3) run short MH chains
        kept_samples = []
        acc_rates = []

        for c in range(len(flagged)):
            x = starts[c].copy()
            logpx = float(self.log_posterior_surrogate(x[None, :], inflate=inflate)[0])

            n_acc = 0
            chain_kept = []

            for t in range(steps_per_chain):
                x_prop = self.rng.multivariate_normal(mean=x, cov=prop_covs[c])
                logp_prop = float(self.log_posterior_surrogate(x_prop[None, :], inflate=inflate)[0])

                log_alpha = logp_prop - logpx
                if np.log(self.rng.uniform()) < min(0.0, log_alpha):
                    x = x_prop
                    logpx = logp_prop
                    n_acc += 1

                if t >= burnin and ((t - burnin) % keep_every == 0):
                    chain_kept.append(x.copy())

            acc_rates.append(n_acc / max(steps_per_chain, 1))

            if len(chain_kept) == 0:
                chain_kept = [x.copy()]

            if return_all_samples:
                kept_samples.extend(chain_kept)
            else:
                kept_samples.append(chain_kept[-1])

        X_new = np.asarray(kept_samples, dtype=float)

        # ------------------------------------------------------------
        # 4) rough proposal log-density for bookkeeping
        # We approximate q_new as a mixture over flagged starting regions:
        #
        #   q(theta) = average_j N(theta ; center_j, prop_cov_j + local_cov_j)
        #
        # This is not the exact MH path density, but it is a practical
        # approximation if you want to keep log_prop_db bookkeeping.
        log_parts = []
        mix_w = 1.0 / len(flagged)

        for j, lm in enumerate(flagged):
            approx_cov = prop_covs[j] + lm["cov"]
            log_parts.append(
                np.log(mix_w) + self._safe_mvn_logpdf(X_new, centers[j], approx_cov)
            )

        logq_new = logsumexp(np.column_stack(log_parts), axis=1)

        info = {
            "selected_local_ids": [int(i) for i in chosen_ids],
            "acceptance_rates": acc_rates,
            "mean_acceptance": float(np.mean(acc_rates)),
            "n_returned": int(X_new.shape[0]),
        }

        return X_new, logq_new, info

    # ------------------------------------------------------------------

    def adaptive_refine(
            self,
            max_iter=20,
            top_frac=0.20,
            n_new_per_iter=10,
            inflate=1.0,
            improve_tol=0.01,
            patience=4,
            min_iter=3,
            n_post_pred=1000,
            sampler_mcmcm=False, # 'mcmc' or 'ball'
            keep_best_state=True,
            true_target=None,
            verbose=True,
    ):
        """
        Simplified adaptive refinement, but with the same history keys as the
        previous richer implementation.

        History keys kept:
            iteration
            db_size
            mean_radius
            max_radius
            posterior_mode_weight
            n_flagged
            rel_improvement_radius
            mode_shift
            theta_change
            predictive_score
            best_predictive_score_so_far
            best_iter_so_far
            truth_wasserstein
            stop
            stop_reason
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
            # ----------------------------------------------
            # refresh local models and archive posterior
            self.fit_local_models()
            self.compute_posterior_weights(inflate=1.0)

            diag_df = self.diagnostics()
            radii = np.asarray(diag_df["radii"], dtype=float)

            mean_radius = float(np.mean(radii))
            max_radius = float(np.max(radii))
            mode_x = self.posterior_mode()
            mode_w = float(np.max(self.post_weights_db))

            # ----------------------------------------------
            # predictive score wasser
            pred_score = self._posterior_predictive_score(n_post=n_post_pred)

            # posterior-to-posterior change
            Xi = self.posterior_particles(n_samples=2000)
            theta_change = (
                np.nan if prev_Xi is None
                else float(np.sqrt(sliced_wasserstein_2(
                    np.asarray(prev_Xi, dtype=float),
                    np.asarray(Xi, dtype=float),
                    n_proj=100
                )))
            )

            # optional truth-based diagnostic for synthetic examples
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

            # ----------------------------------------------
            # "flagged" count: keep same semantics as old history
            n_flagged = max(1, int(np.ceil(top_frac * len(radii))))

            # ----------------------------------------------
            # best predictive checkpoint
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

            # ----------------------------------------------
            # stopping
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
                "predictive_score_W2": float(pred_score),
                "best_predictive_score_so_far_W2": float(best_pred_score),
                "best_iter_so_far": int(best_iter) if best_iter is not None else -1,
                "truth_wasserstein": Wasser_Distance,
                "stop": bool(stop),
                "stop_reason": reason,
            })

            if verbose:
                msg = (
                    f"[iter {it + 1}/{max_iter}] "
                    f"db={self.X_db.shape[0]} "
                    f"pred_W2={pred_score:.4g} "
                    f"mean_radius={mean_radius:.4g} "
                    f"max_radius={max_radius:.4g} "
                    f"flagged={n_flagged}"
                )
                if Wasser_Distance is not None:
                    msg += f" truth_W={Wasser_Distance:.4g}"
                print(msg)
            if stop:
                break

            # ----------------------------------------------
            # simple enrichment from the worst ball
            if sampler_mcmcm:
                X_new, logq_new, mcmc_info = self.sample_mcmc_from_flagged(
                    n_chains=5,
                    steps_per_chain=60,
                    burnin=20,
                    top_frac=top_frac,
                    proposal_scale=0.5,
                    inflate=inflate,
                    start_mode="mean",
                    keep_every=5,
                    return_all_samples=False,
                )

                if verbose:
                    print(
                        f"   MCMC enrich -> n_new={X_new.shape[0]} "
                        f"mean_acc={mcmc_info['mean_acceptance']:.3f}"
                    )
            else:
                sample_out = self.sample_from_worst_ball(
                    n_per_center=n_new_per_iter,
                    inflate=inflate,
                )
                # support both signatures:
                #   (X_new, logq_new)
                #   (X_new, logq_new, worst_local_id)
                if len(sample_out) == 2:
                    X_new, logq_new = sample_out
                else:
                    X_new, logq_new, _ = sample_out

            self.append_to_archive(X_new, logq_new)

            prev_mean_radius = mean_radius
            prev_mode = mode_x.copy()
            prev_Xi = Xi.copy()

        # ----------------------------------------------
        # restore best predictive state if requested
        if keep_best_state and best_state is not None:
            self.X_db = best_state["X_db"]
            self.Y_db_by_design = best_state["Y_db_by_design"]
            self.log_prior_db = best_state["log_prior_db"]
            self.log_prop_db = best_state["log_prop_db"]

        # final refresh
        self.fit_local_models()
        self.compute_posterior_weights(inflate=1.0)

        return pd.DataFrame(self.history), self.diagnostics()