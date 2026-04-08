# Adaptive KNN-KDE Inverse Problem Solver (Modular Version)
# ========================================================
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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
        """
        Simulate all designs for the same shared particles X.
        """
        X = np.asarray(X, dtype=float)
        Y_by_design = {}
        for design in self.designs:
            Y_by_design[design] = self._ensure_2d(self.model(X, design))
        return Y_by_design

    def _initialize_prior_db(self, N0=1000):
        """ Initial archive: draw theta from the prior, then simulate all designs.

        Methodological note:  This is the correct archive structure for later multiplicative
        aggregation across designs, because every row corresponds to the same  theta_k evaluated under all designs.
        """
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
        """
        Small wrapper to tolerate near-singular local covariances.
        """
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
    def diagnostics(self):
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

        qrad = max(np.quantile(radii, 0.5), 1e-12)
        qspr = max(np.quantile(spread_x, 0.75), 1e-12)

        #priority = ( radii / qrad   + spread_x / qspr   + (0.4 * self.K) / np.maximum(ess, 1e-12)  )
        priority = (    radii / qrad   )
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

    def sample_targeted_batch_v0( self,  flagged_local_ids,
                               n_new=100,  inflate=1.0,
                               prior_weight=0.10):
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

    def sample_targeted_batch(
            self,
            flagged_local_ids,
            n_new=100,
            inflate=1.0,
            prior_weight=0.10,
            n_rep_per_choice=100,
    ):
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

                X_loc = self.rng.multivariate_normal(
                    mean=mu,
                    cov=cov,
                    size=n_rep_per_choice,
                )
                X_parts.append(np.atleast_2d(X_loc))

        if len(X_parts) == 0:
            X_new = np.empty((0, self.d_x), dtype=float)
            logq_new = np.empty((0,), dtype=float)
            return X_new, logq_new

        X_new = np.vstack(X_parts)

        # Trim to exactly n_new samples
        if X_new.shape[0] > n_new:
            X_new = X_new[:n_new]

        logq_new = self._proposal_logpdf(
            X_new,
            flagged=flagged,
            mix_w=mix_w,
            prior_weight=prior_weight,
            inflate=inflate,
        )
        return X_new, logq_new






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

    def adaptive_refine(self,
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

            """
            plt.scatter(true_target[:,0],true_target[:,1])
            plt.scatter(self.X_db[self.post_weights_db>1e-4,0],self.X_db[self.post_weights_db>1e-4,1]) 
            plt.scatter(self.X_db[self.post_weights_db>0.001,0],self.X_db[self.post_weights_db>0.001,1]) 
            plt.scatter(self.X_db[self.post_weights_db>0.02,0],self.X_db[self.post_weights_db>0.02,1]) 
            plt.show()
            """

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
                f" flagged={n_flagged}")

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
            n_per_center=5,
            inflate=1.0,
            kernel="gaussian",
            use_radius_only=True,
    ):
        """
        Simpler refinement:
        1) pick the empirical response with the largest KNN ball
        2) take its K reconstructed input neighbours
        3) sample around each of those K input points
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

        elif kernel == "uniform_box":
            std = np.sqrt(np.diag(cov))
            for mu in centers:
                U = self.rng.uniform(
                    low=mu - std,
                    high=mu + std,
                    size=(n_per_center, self.d_x),
                )
                X_new_parts.append(U)

        else:
            raise ValueError("kernel must be 'gaussian' or 'uniform_box'")

        X_new = np.vstack(X_new_parts)
        return X_new, worst_local_id, lm

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



if __name__ == '__main__':
    print("=== DEMO: Posterior Progression Towards True Target ===")
    # --------------------------------------------------------------
    # Forward model with explicit design argument
    from resources.loader_usecases import prepare_case
    demo_model, _, _ = prepare_case(1, Nemp=10, Nsim=1000)

    # --------------------------------------------------------------
    # Synthetic "true posterior" used only to generate empirical data
    rng = np.random.default_rng(123)
    m = 100
    weights = np.array([0.55, 0.45])
    mus_true = np.array([[-0.5,  1.0],   [ 0.2, -1.0]])
    covs_true = np.array([[[ 0.15,  0.05], [ 0.05,  0.20]], [[ 0.20, -0.03], [-0.03,  0.10]] ])
    z = rng.choice(2, size=m, p=weights)
    X_latent = np.zeros((m, 2))
    for k in range(2):
        idx = (z == k)
        X_latent[idx] = rng.multivariate_normal(mus_true[k], covs_true[k], size=idx.sum())
    # --------------------------------------------------------------
    # One or more experimental designs
    designs = [-1.0, 2.0, 4.0]
    Y_emp_by_design = { xi: demo_model(X_latent, xi) for xi in designs }

    # --------------------------------------------------------------
    prior = UniformBoxPrior(low=[-15,-15], high=[15,15])

    # --------------------------------------------------------------
    # Instantiate the NEW solver
    solver = AdaptiveInverseKNNKDE(   model=demo_model,
                                            Y_emp_by_design=Y_emp_by_design,
                                            prior=prior,
                                            N0=700,
                                            K=50,
                                            ridge=1e-3,
                                            seed=123,
                                            )

    print("Initial archive summary:", solver.archive_summary())

    # --------------------------------------------------------------
    # Adaptive refinement
    n_new_per_iter = 100
    top_frac = 0.2

    hist, diag = solver.adaptive_refine(
        max_iter=15,
        top_frac=top_frac,
        n_new_per_iter=n_new_per_iter,
        inflate=1.0,
        prior_weight=0.10,
        improve_tol=0.05,
        patience=3,
        target_shrink=0.60,
        min_iter=5,
    )

    print("\n=== Refinement history ===")
    print(hist)

    print("\n=== Final diagnostics ===")
    print(diag.head())

    # --------------------------------------------------------------
    # Posterior summaries from the weighted shared archive
    # --------------------------------------------------------------
    post_mean = solver.posterior_mean()
    post_mode, post_mode_w = solver.posterior_mode_from_db()

    print("\nPosterior mean:", post_mean)
    print("Posterior mode (best archive particle):", post_mode)
    print("Posterior mode weight:", post_mode_w)

    # Optional posterior particle resampling
    X_post = solver.posterior_particles(n_samples=5000, replace=True)
    print("Posterior particle sample shape:", X_post.shape)