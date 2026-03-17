from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity


class Calibrator(ABC):
    """
    Abstract base class for calibration methods.

    Workflow
    --------
    1. setup(...)      → provide priors, simulator, or precomputed simulations
    2. calibrate(...)  → condition on observations, produce posterior
    3. get_posterior() → retrieve posterior representation
    """

    def __init__(self):
        self.is_ready = False

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Define priors, simulator, or precomputed simulations."""
        pass

    @abstractmethod
    def calibrate(self, observations: Any, resample_n: Optional[int] = None) -> Any:
        """Condition on observed data to produce posterior samples."""
        pass

    @abstractmethod
    def get_posterior(self) -> Any:
        """Retrieve posterior representation (samples, chains, or density)."""
        pass


class MCMCCalibrator(Calibrator):
    """  Calibration via Bayesian MCMC (e.g. Metropolis-Hastings, HMC, NUTS). """
    def __init__(self, n_chains: int = 4,
                 n_samples: int = 1000,
                 burn_in: int = 200):
        super().__init__()
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.burn_in = burn_in

        # internal state placeholders
        self._prior = None
        self._likelihood = None
        self._posterior_chain = None

    def setup(self, prior=None,
              likelihood=None,
              model=None):
        """  Define priors and likelihood (or simulator-based likelihood). """
        self._prior = prior
        self._likelihood = likelihood
        self.is_ready = True
        # TODO: implement sampler initialization (PyMC, NumPyro, etc.)

    def calibrate(self, observations: Any, resample_n: Optional[int] = None) -> Any:
        """  Run MCMC to sample posterior given observations.  """
        if not self.is_ready:
            raise RuntimeError("Call setup() before calibrate().")
        # TODO: implement actual MCMC run
        self._posterior_chain = None
        return self._posterior_chain

    def get_posterior(self) -> Any:
        """Return MCMC chain or posterior samples."""
        return self._posterior_chain



class KNNCalibrator(Calibrator):
    r"""
    Unified kNN-based calibrator for black-box models or precomputed simulations.

    Setup (unified)
    ---------------
    - If `evaluate_model=False` and `simulated_data` is provided, we **reuse** simulations and
      build a per-design kNN index by **filtering** rows with |xi - xi*| < a_tol for each ξ* in `xi_list`.
    - If `evaluate_model=True`, we **simulate** `y = model(theta, xi)` **for each** ξ in `xi_list`, using
      a **shared θ grid** drawn once from `theta_sampler(n_samples)`. Then we build per-design kNN indices.

    Calibration (single logic for single/multi-design)
    --------------------------------------------------
    For each `(y_obs, xi)`:
      1) standardize `y_obs` with the per-design scaler,
      2) get k nearest neighbors in y-space,
      3) map indices → θ for that design.
    Finally we **stack** θ across all observations/designs (or optionally tally/vote).

    Args
    ----
    knn : int
        Number of neighbors per observed row.
    a_tol : float
        Tolerance for matching `simulated_data['xi']` to a requested ξ* (when reusing).
    evaluate_model : bool
        If True, call the black-box `model` for each ξ in `xi_list` on a shared θ grid.
        If False, reuse `simulated_data` (requires y/theta/xi).
    random_state : Optional[int]
        Seed for reproducibility (affects theta_sampler and resampling).
    """

    def __init__(self,
                 knn: int = 100,
                 a_tol: float = 0.05,
                 evaluate_model: bool = False
                 ):

        super().__init__()
        self.knn = int(knn)
        self.a_tol = float(a_tol)
        self.evaluate_model = bool(evaluate_model)
        self.random_state = 42

        # Internal state
        self._theta_grid: Optional[np.ndarray] = None  # shared grid if evaluate_model=True, else unused
        self._theta_by_xi: Dict[Tuple[float, ...], np.ndarray] = {}  # per-design θ (may be shared ref)
        self._y_by_xi: Dict[Tuple[float, ...], np.ndarray] = {}  # per-design y
        self._scaler_by_xi: Dict[Tuple[float, ...], StandardScaler] = {}
        self._neigh_by_xi: Dict[Tuple[float, ...], NearestNeighbors] = {}
        self._grid_idx_by_xi: Dict[Tuple[float, ...], np.ndarray] = {}
        self._posterior: Optional[Dict[str, Any]] = None

        # Keep original sims if reusing
        self._sim_y: Optional[np.ndarray] = None
        self._sim_theta: Optional[np.ndarray] = None
        self._sim_xi: Optional[np.ndarray] = None

    # ---------- utilities ----------
    @staticmethod
    def _key_from_xi(xi) -> Tuple[float, ...]:
        """Stable tuple key for a scalar/vector design ξ."""
        return tuple(np.atleast_1d(np.asarray(xi, float)).ravel())

    # ---------- setup ----------
    def setup(self,
              model: Optional[Callable[[np.ndarray, Union[float, np.ndarray]], np.ndarray]] = None,
              theta_sampler: Optional[Callable[[int], np.ndarray]] = None,
              simulated_data: Optional[Dict[str, np.ndarray]] = None,
              xi_list: Optional[List[Union[float, np.ndarray]]] = None,
              n_samples: int = 10000):
        """
        Prepare per-design kNN structures by either reusing `simulated_data` or by simulating for each design.

        Parameters
        ----------
        model : callable
            Black-box simulator with signature `model(theta, xi) -> y` (vectorized over theta).
        theta_sampler : callable
            Sampler for θ; required when `evaluate_model=True`.
        simulated_data : dict
            Dict with keys {"y": (n, dy), "theta": (n, dθ), "xi": (n, dξ)} when reusing sims.
        xi_list : list
            List of designs; each item can be scalar or array-like. If None → [0.0].
        n_samples : int
            Number of θ samples to draw when `evaluate_model=True`.
        """
        xi_list = [0.0] if not xi_list else xi_list

        # Reset state
        self._theta_grid = None
        self._theta_by_xi.clear()
        self._y_by_xi.clear()
        self._scaler_by_xi.clear()
        self._neigh_by_xi.clear()
        self._posterior = None

        if not self.evaluate_model:
            # ---- Reuse provided simulations; filter per design ----
            if simulated_data is None:
                raise ValueError("evaluate_model=False requires `simulated_data` with keys 'y','theta','xi'.")

            self._sim_y = np.asarray(simulated_data["y"], float)
            self._sim_theta = np.asarray(simulated_data["theta"], float)
            self._sim_xi = np.asarray(simulated_data.get("xi", None), float)
            if self._sim_xi is None:
                raise ValueError("`simulated_data` must include 'xi' to filter per design.")

            for xi in xi_list:
                key = self._key_from_xi(xi)
                mask = np.all(np.abs(self._sim_xi - np.atleast_1d(xi)) < self.a_tol, axis=1)
                y_xi = self._sim_y[mask]
                theta_xi = self._sim_theta[mask]
                if y_xi.size == 0:
                    raise ValueError(f"No simulations matched design {xi} within tolerance a_tol={self.a_tol}.")
                # drop NaNs rows in y
                ok = ~np.isnan(y_xi).any(axis=1)
                y_xi, theta_xi = y_xi[ok], theta_xi[ok]
                if y_xi.size == 0:
                    raise ValueError(f"All simulations at design {xi} had NaNs in y.")
                # build scaler & kNN
                sc = StandardScaler().fit(y_xi)
                neigh = NearestNeighbors(n_neighbors=self.knn).fit(sc.transform(y_xi))
                # store
                self._theta_by_xi[key] = theta_xi
                self._y_by_xi[key] = y_xi
                self._scaler_by_xi[key] = sc
                self._neigh_by_xi[key] = neigh

        else:
            # ---- Evaluate model per design on a shared θ grid ----
            if model is None or theta_sampler is None:
                raise ValueError("evaluate_model=True requires `model` and `theta_sampler`.")
            self._theta_grid = np.asarray(theta_sampler(int(n_samples)), float)
            if self._theta_grid.ndim != 2:
                raise ValueError("theta_sampler must return a 2D array (n_samples, dθ).")

            for xi in xi_list:
                key = self._key_from_xi(xi)
                # vectorized model call over θ
                y_xi = np.asarray(model(self._theta_grid, xi), float)
                if y_xi.ndim == 1:
                    y_xi = y_xi[:, None]
                if y_xi.shape[0] != self._theta_grid.shape[0]:
                    raise ValueError("Model must return one row of y per θ sample.")
                # drop NaNs rows in y (and corresponding θ rows)
                ok = ~np.isnan(y_xi).any(axis=1)
                y_xi = y_xi[ok]
                theta_xi = self._theta_grid[ok]
                grid_idx = np.where(ok)[0]
                self._grid_idx_by_xi[key] = grid_idx  # maps local row j -> global grid index grid_idx[j]

                if y_xi.size == 0:
                    raise ValueError(f"All simulations at design {xi} had NaNs in y.")
                # build scaler & kNN
                sc = StandardScaler().fit(y_xi)
                neigh = NearestNeighbors(n_neighbors=self.knn).fit(sc.transform(y_xi))
                # store
                self._theta_by_xi[key] = theta_xi
                self._y_by_xi[key] = y_xi
                self._scaler_by_xi[key] = sc
                self._neigh_by_xi[key] = neigh
        self.is_ready = True

    # ---------- nearest ----------
    def nearest(self,
                y: Union[np.ndarray, List[float]],
                xi: Union[float, np.ndarray],
                k: Optional[int] = None,
                return_dist: bool = False):
        """
        Return k nearest neighbors for `y` at design `xi`.

        Parameters
        ----------
        y : array-like, shape (m, d_y) or (d_y,)
            Query outputs.
        xi : scalar or array-like
            Design key to select the per-design index.
        k : Optional[int]
            Number of neighbors; defaults to self.knn.
        return_dist : bool
            If True, also return distances and raw indices.

        Returns
        -------
        theta_neighbors : (m*k, dθ) stacked θ for all query rows
        distances, indices : optional
        """
        if not self.is_ready:
            raise RuntimeError("Call setup() before nearest().")
        key = self._key_from_xi(xi)
        if key not in self._neigh_by_xi:
            raise KeyError(f"Design {xi} not in index. Known: {list(self._neigh_by_xi.keys())}")
        y = np.atleast_2d(np.asarray(y, float))
        sc = self._scaler_by_xi[key]
        neigh = self._neigh_by_xi[key]
        k_eff = int(k or self.knn)
        d, idx = neigh.kneighbors(sc.transform(y), n_neighbors=k_eff)
        theta = self._theta_by_xi[key]
        theta_neighbors = np.vstack([theta[i] for i in idx])
        if return_dist:
            return theta_neighbors, d, idx
        return theta_neighbors

    # ---------- calibration ----------
    def calibrate(self,
                  observations,
                  resample_n: int | None = None,
                  combine: str = "stack",
                  combine_params: dict | None = None):
        """
        kNN calibration with two aggregation modes:
        combine:
          - 'stack'     : concatenate all kNN θ; optional de-duplication
          - 'intersect' : keep θ that occur at least 'min_count' times across all neighbor hits
        combine_params:
          - dedup: bool (default False) — only for 'stack'
          - theta_match_tol: float (default 1e-9) — rounding quantum for row matching/dup
          - min_count: int | None — minimum occurrences for 'intersect'
                       default: max(1, ceil(0.5 * total_blocks))   # “appear in about half of the lists”
          - use_kde: bool (default False) — if True, compute KDE log-scores and normalized weights
          - kde_bandwidth: float | None — optional bandwidth for KDE (Scott’s rule if None)
        Returns:
          dict with keys:
            'mode'    : 'knn'
            'theta'   : (n,dθ) posterior samples (resampled if requested)
            'weights' : None (stack/intersect) or KDE weights if use_kde=True
            'meta'    : dict with aggregation info (and KDE bandwidth if used)
        """
        if not self.is_ready:
            raise RuntimeError("Call setup() before calibrate().")

        combine_params = combine_params or {}
        dedup = bool(combine_params.get("dedup", False))
        tol = float(combine_params.get("theta_match_tol", 1e-9))
        use_kde = bool(combine_params.get("use_kde", True))
        kde_bw = combine_params.get("kde_bandwidth", 0.1)

        # ---------------- Collect θ-neighbors for every (y, ξ) ----------------
        theta_hits = []  # list of (n_i*k, dθ) blocks, one block per y-row (across all designs)
        for (y_obs, xi) in observations:
            key = self._key_from_xi(xi)
            if key not in self._neigh_by_xi:
                raise KeyError(f"Design {xi} not in index. Known: {list(self._neigh_by_xi.keys())}")
            yo = np.atleast_2d(np.asarray(y_obs, float))
            yo = yo[~np.isnan(yo).any(axis=1)]
            if yo.size == 0:
                continue
            sc, neigh = self._scaler_by_xi[key], self._neigh_by_xi[key]
            d, idx = neigh.kneighbors(sc.transform(yo), n_neighbors=self.knn, return_distance=True)
            # gather θ for this design
            th = self._theta_by_xi[key]
            # flatten all rows’ neighbors for this block
            theta_block = np.vstack([th[i] for i in idx])  # (n_rows*k, dθ)
            theta_hits.append(theta_block)

        if len(theta_hits) == 0:
            raise ValueError("No valid observations after NaN filtering.")

        # ---------------- Aggregation strategies ----------------
        if combine == "stack":
            theta_all = np.vstack(theta_hits)  # (sum n_i*k, dθ)

            if dedup:
                uniq, _ = self._round_rows(theta_all, tol)
                theta_out = uniq
            else:
                theta_out = theta_all

            # Optional KDE scoring on returned support
            weights = None
            meta = {"combine": "stack", "dedup": dedup, "theta_match_tol": tol}
            if use_kde and theta_out.shape[0] > 0:
                logp, w = self._kde_logweights(theta_out, bw=kde_bw)
                weights = w
                meta.update({"use_kde": True, "kde_bandwidth": kde_bw})

            # Optional resampling
            if resample_n and theta_out.shape[0] > 0:
                rng = np.random.default_rng(self.random_state)
                if weights is None:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True)
                else:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True, p=weights)
                theta_out = theta_out[take]

            self._posterior = {"mode": "knn", "theta": theta_out, "weights": weights, "meta": meta}
            return self._posterior

        elif combine == "intersect":
            # Build one big stack and count approximate matches
            big = np.vstack(theta_hits)  # (M, dθ)
            uniq, counts = self._round_rows(big, tol)

            # total neighbor lists (one per row across all designs)
            total_blocks = sum(b.shape[0] // self.knn for b in theta_hits)

            # Strictness knobs
            min_frac = float(combine_params.get("min_frac", 0.8))  # keep θ seen in ≥80% of lists
            min_count = combine_params.get("min_count", None)
            if min_count is None:
                min_count = max(1, int(np.ceil(min_frac * total_blocks)))

            # Filter by frequency
            keep = counts >= int(min_count)
            theta_out = uniq[keep]
            counts_sel = counts[keep].astype(float)

            # If nothing passed, fall back to TOP-FRACTION
            if theta_out.shape[0] == 0:
                top_frac = float(combine_params.get("top_frac", 0.1))  # keep top 10% by frequency
                k = max(1, int(np.ceil(top_frac * len(counts))))
                top_idx = np.argsort(counts)[::-1][:k]
                theta_out = uniq[top_idx]
                counts_sel = counts[top_idx].astype(float)
                meta = {"combine": "intersect", "theta_match_tol": tol,
                        "min_count": min_count, "min_frac": min_frac,
                        "fallback": f"top-{top_frac:.2f}"}
            else:
                meta = {"combine": "intersect", "theta_match_tol": tol,
                        "min_count": min_count, "min_frac": min_frac}

            # Frequency-based weights (sharpen with gamma)
            weights = None
            if theta_out.shape[0] > 0:
                gamma = float(combine_params.get("gamma", 1.0))  # 1.0=no sharpen, 2.0=stricter
                w_counts = counts_sel ** max(gamma, 1e-12)

                # Optional: KDE blending for smoother density
                if bool(combine_params.get("use_kde", False)):
                    kde_bw = combine_params.get("kde_bandwidth", 0.1)
                    _, w_kde = self._kde_logweights(theta_out, bw=kde_bw)
                    beta = float(combine_params.get("beta", 1.0))  # blend exponent for KDE
                    w = w_counts * (w_kde ** beta)
                    w = np.asarray(w, float)
                    w = w / (w.sum() if w.sum() > 0 else len(w))
                    weights = w
                    meta.update({"use_kde": True, "kde_bandwidth": kde_bw, "gamma": gamma, "beta": beta})
                else:
                    w = w_counts / (w_counts.sum() if w_counts.sum() > 0 else len(w_counts))
                    weights = w
                    meta.update({"gamma": gamma})

            # Optional resampling
            if resample_n and theta_out.shape[0] > 0:
                rng = np.random.default_rng(self.random_state)
                if weights is None:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True)
                else:
                    take = rng.choice(theta_out.shape[0], size=int(resample_n), replace=True, p=weights)
                theta_out = theta_out[take]

            self._posterior = {"mode": "knn", "theta": theta_out, "weights": weights, "meta": meta}
            return self._posterior

        else:
            raise ValueError("`combine` must be 'stack' or 'intersect'.")

    def _round_rows(self, A: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Round rows of A to multiples of `tol` and return (unique_rows, counts).
        If tol <= 0, exact matching is used.
        """
        import numpy as _np
        A = _np.asarray(A, float)
        if A.size == 0:
            return A.copy(), _np.array([], dtype=int)
        if tol <= 0:
            uniq, idx, counts = _np.unique(A, axis=0, return_index=True, return_counts=True)
            order = _np.sort(idx)
            uniq = A[order]
            counts = counts[_np.argsort(idx)]
            return uniq, counts
        R = _np.round(A / tol) * tol
        uniq, idx, counts = _np.unique(R, axis=0, return_index=True, return_counts=True)
        order = _np.sort(idx)
        uniq = R[order]
        counts = counts[_np.argsort(idx)]
        return uniq, counts

    def _kde_logweights(self, X, bw=0.5, n_max_exact=5000):
        """
        Compute KDE-based log-weights for posterior samples X.

        Args:
            X : ndarray (n, d)
                Posterior samples.
            bw : float
                Bandwidth for Gaussian kernel.
            n_max_exact : int
                Max n for exact pairwise KDE. Above this, fall back to sklearn.KernelDensity.

        Returns:
            logp : ndarray (n,)
                Log-density values at X.
            w : ndarray (n,)
                Normalized weights.
        """
        n, d = X.shape
        if n <= n_max_exact:
            # ---- Exact method (safe for small n) ----
            h2 = float(bw) ** 2
            d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)  # (n,n)
            K = np.exp(-0.5 * d2 / (h2 + 1e-18))
            sK = K.sum(axis=1) + 1e-300
            logp = np.log(sK)
            w = sK / sK.sum()
        else:
            # ---- Scalable method (sklearn KD-tree backend) ----
            kde = KernelDensity(kernel="gaussian", bandwidth=bw)
            kde.fit(X)
            logp = kde.score_samples(X)  # log density at each sample
            w = np.exp(logp - logp.max())
            w /= w.sum()

        return logp, w

    # ---------- posterior ----------
    def get_posterior(self) -> Any:
        """Return the last computed posterior dict; raises if calibrate() hasn't been called."""
        if self._posterior is None:
            raise RuntimeError("No posterior available. Run calibrate() first.")
        return self._posterior


class AdaptiveKNNCalibrator:
    """
    Semi-Bayesian calibration with adaptive simulation augmentation.

    Steps:
    1. Generate prior simulations D_sim = {(x_i, y_i)}.
    2. For each empirical y_emp find K nearest simulated outputs.
    3. Compute spread of the corresponding x-set (variance proxy).
    4. If spread > tau_threshold, locally resample around the K-set,
       evaluate simulator, and augment D_sim.
    5. Return calibration dataset D_cal and the non-parametric posterior.
    """

    def __init__(self,
                 simulator,  # simulation model
                 sample_prior,  # starting prior sampler
                 k=20,  # number of knn
                 tau_factor=1.5,
                 aug_size=50,
                 cov_scale=0.3,
                 max_iter=3):
        self.simulator = simulator
        self.sample_prior = sample_prior

        self.k = k
        self.tau_factor = tau_factor
        self.aug_size = aug_size
        self.cov_scale = cov_scale
        self.max_iter = max_iter

        # storage
        self.X_sim = None
        self.Y_sim = None
        self.history = []

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def compute_spread(X):
        """ Scalar measure of spread (mean variance across dims)."""
        return np.mean(np.var(X, axis=0))

    @staticmethod
    def regularize_cov(C, eps=1e-6):
        """Ensure covariance is SPD."""
        return C + eps * np.eye(C.shape[0])

    # ------------------------------------------------------------
    # Main functions
    # ------------------------------------------------------------
    def initialize_simulations(self, n_s):
        """Generate initial simulation dataset."""
        self.X_sim = self.sample_prior(n_s)
        self.Y_sim = self.simulator(self.X_sim)

    def find_K_sets(self, Y_emp):
        """Return list of K-nearest neighbor index sets for each y_emp."""
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(self.Y_sim)
        _, idx = nbrs.kneighbors(Y_emp)
        return idx

    def augment_region(self, X_region):
        """Resample around X_region and simulate."""
        mu = np.mean(X_region, axis=0)
        Cov = np.cov(X_region.T)
        Cov = self.regularize_cov(Cov) * self.cov_scale

        X_new = np.random.multivariate_normal(mu, Cov, size=self.aug_size)
        Y_new = self.simulator(X_new)
        return X_new, Y_new

    def run(self, Y_emp, n_s):
        """Run the full adaptive calibration procedure."""
        self.initialize_simulations(n_s)
        Y_emp = np.atleast_2d(Y_emp)
        for it in range(self.max_iter):
            idx_sets = self.find_K_sets(Y_emp)

            # Compute spreads
            spreads = np.array([self.compute_spread(self.X_sim[idx]) for idx in idx_sets])
            tau = self.tau_factor * np.median(spreads)

            # Identify sparse regions
            to_augment = np.where(spreads > tau)[0]
            if len(to_augment) == 0:
                break  # converged

            # Augmentation step
            X_new_all, Y_new_all = [], []
            for i in to_augment:
                X_region = self.X_sim[idx_sets[i]]
                X_new, Y_new = self.augment_region(X_region)
                X_new_all.append(X_new)
                Y_new_all.append(Y_new)

            # Add new simulations
            X_new_all = np.vstack(X_new_all)
            Y_new_all = np.vstack(Y_new_all)
            self.X_sim = np.vstack([self.X_sim, X_new_all])
            self.Y_sim = np.vstack([self.Y_sim, Y_new_all])

            # diagnostics
            self.history.append({
                "iteration": it,
                "n_sim": len(self.X_sim),
                "spreads": spreads,
                "tau": tau,
                "n_augmented_regions": len(to_augment)
            })

        # Final K-set calibration dataset
        final_idx = self.find_K_sets(Y_emp)
        D_cal = np.vstack([self.X_sim[idx] for idx in final_idx])

        return D_cal



def estimate_p_theta_knn(observed_data,
                         simulated_data,
                         xi_star,
                         knn: int = 20,
                         a_tol: float = 0.05):
    """
    Estimate the posterior distribution p(θ) of θ using a k-Nearest Neighbors (kNN)
    filter on a pre-computed simulation archive, conditioned on a design ξ*.

    This method restricts the simulation archive to runs at (or near) the
    target design ξ*, then fits a kNN model in output (y) space. For each
    observed output y_obs, it retrieves the k-nearest simulated outputs and
    returns the corresponding θ values as approximate posterior samples.
    Args:
        observed_data (np.ndarray):
            Array of observed outputs y_obs (shape: n_obs × d_y).
            Must match the dimensionality of simulated outputs.
        simulated_data (list):
            A list of arrays [y, θ, ξ], containing
                - y (n × d_y): simulation output, e.g. a transformed y with only KPIs
                - θ (n × d_theta): parameters and variables to be calibrated
                - ξ (n × d_xi):  conditioning controllable factors, e.g., design,  parameters
        knn (int):
            Number of nearest neighbors to query per observed sample.
        xi_star
            Target design ξ* at which the posterior is estimated.
        a_tol (float, optional):
            Tolerance for matching simulations to ξ*. Defaults to 0.1.
            A simulation is kept if ||xi_sim - xi_star||∞ < a_tol.

    Returns:
        np.ndarray:
            θ samples from the posterior, stacked across all observed y.
            Shape: (n_obs × knn, d_theta).

    Raises:
        ValueError: If filtering leaves no simulations at ξ*.
        RuntimeError: If kNN search fails due to inconsistent dimensions.

    Notes:
        - Scaling of outputs y is performed internally via StandardScaler
          for robustness against different KPI magnitudes.
        - The parameter `knn` acts as a smoothing parameter: higher values
          broaden the posterior but reduce sharpness.
        - The choice of `a_tol` trades off strict design conditioning vs.
          sample size. Too small → few matches; too large → weaker conditioning.

    Example:
        >>> import numpy as np
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.neighbors import NearestNeighbors
        >>> # Fake simulator archive
        >>> theta_sim = np.random.uniform(-5, 5, size=(5000, 2))
        >>> xi_sim = np.zeros((5000, 1))
        >>> y_sim = np.sum(theta_sim**2, axis=1, keepdims=True) \
        ...         + 0.1*np.random.randn(5000, 1)
        >>> simulated_data = [y_sim, theta_sim, xi_sim]
        >>> # Observed data
        >>> theta_true = np.array([1.5, -2.0])
        >>> y_obs = np.sum(theta_true**2) + 0.1*np.random.randn(1)
        >>> # Estimate posterior
        >>> theta_post = estimate_p_theta_knn(
        ...     observed_data=np.array([[y_obs]]),
        ...     simulated_data=simulated_data,
        ...     knn=50,
        ...     xi_star=0.0
        ... )
        >>> theta_post.shape
        (50, 2)
        >>> theta_post.mean(axis=0)
        array([ 1.4, -2.1])  # close to true [1.5, -2.0]
    """

    # Step 1: Filter simulated datasets based on ξ = ξ*
    if xi_star is None:
        simulated_data_xi = [s for s in simulated_data]
    else:
        xi_idx = np.all((np.abs(simulated_data[2] - xi_star) / (np.abs(xi_star) + 1e-10)) < a_tol, axis=1)
        simulated_data_xi = [s[xi_idx] for s in simulated_data]

    # Step 2: fit a kNN on the (filtered) space of y. Normalize observations
    scaler = StandardScaler()
    if np.any(np.isnan(simulated_data_xi[0])):
        simulated_data_xi[0] = simulated_data_xi[0][~np.isnan(simulated_data_xi[0]).any(axis=1)]

    scaler.fit(simulated_data_xi[0])
    neigh = NearestNeighbors(n_neighbors=knn)
    neigh.fit(scaler.transform(simulated_data_xi[0]))

    # Step 3: retrieve the kNN for each observed y_i  ...... check if there are nan values in the observed datasets
    if np.any(np.isnan(observed_data)):
        observed_data = observed_data[~np.isnan(observed_data).any(axis=1)]
    dist, knn_idx = neigh.kneighbors(scaler.transform(observed_data))
    theta_set = np.vstack([simulated_data_xi[1][idx] for idx in knn_idx])
    return theta_set




def estimate_p_theta_knn_multi(
    observed_data_list: List[np.ndarray],
    simulated_data: List[np.ndarray],
    xi_star_list: List[np.ndarray],
    knn: int = 20,
    a_tol: float = 0.05,
    return_full_cov: bool = False,
    combine_results: bool = False,
) -> Dict[str, Any]:
    """
    Multi-experiment KNN inversion in output (response) space.

    For each target design ξ* in xi_star_list and corresponding observed dataset,
    this function:
      1) filters the simulation archive near ξ*,
      2) fits kNN on scaled outputs,
      3) retrieves per-observation K nearest simulations,
      4) returns θ-sets, indices, and spread diagnostics in output and input spaces.

    Args:
        observed_data_list:
            List of observed arrays, one per experiment e.
            Each array has shape (n_obs_e, d_y).
        simulated_data:
            [y_sim, theta_sim, xi_sim] with shapes:
                - y_sim:    (n_sim, d_y)
                - theta_sim:(n_sim, d_theta)
                - xi_sim:   (n_sim, d_xi)
        xi_star_list:
            List of target designs ξ* (arrays of shape (d_xi,) or broadcastable).
            Must have the same length as observed_data_list.
        knn:
            Number of nearest neighbors per observation (clipped to available sims).
        a_tol:
            Tolerance for design filtering. If all |ξ*| ≤ 1e-12 → absolute L∞,
            else relative L∞: max|ξ-ξ*|/(|ξ*|+1e-12) ≤ a_tol.
        return_full_cov:
            If True, returns per-observation θ covariance matrices.
        combine_results:
            If True, returns a 'combined' block stacking all experiments
            (useful for downstream batching).

    Returns:
        dict with:
            - "per_experiment": list of length E, each a dict:
                {
                  "xi_star": (d_xi,),
                  "theta_samples": (n_obs_e, knn_eff_e, d_theta),
                  "knn_indices": (n_obs_e, knn_eff_e),              # within filtered subset
                  "knn_indices_global": (n_obs_e, knn_eff_e),       # indices in original archive
                  "epsilon": (n_obs_e,),                            # K-th distances (scaled space)
                  "output_spread": {
                      "distances": (n_obs_e, knn_eff_e),
                      "mean_distance": (n_obs_e,),
                      "std_distance": (n_obs_e,),
                  },
                  "input_spread": {
                      "theta_mean": (n_obs_e, d_theta),
                      "theta_std": (n_obs_e, d_theta),
                      "theta_cov": (n_obs_e, d_theta, d_theta) or None,
                  },
                  "meta": {
                      "knn_eff": int,
                      "n_sim_filtered": int,
                      "scaler_mean_": (d_y,),
                      "scaler_scale_": (d_y,),
                  },
                }
            - "meta": {
                "n_experiments": E,
                "n_sim_initial": int,
                "n_sim_after_nan_filter": int,
              }
            - "combined": optional stacked views when combine_results=True:
                {
                  "theta_samples": (sum_e n_obs_e, max_knn_eff, d_theta) with NaN pad if needed,
                  "experiment_index": (sum_e n_obs_e,),  # which experiment each row came from
                  "epsilon": (sum_e n_obs_e,),
                }

    Raises:
        ValueError / RuntimeError with informative messages on shape mismatches
        or empty filtered sets.
    """
    # Unpack and basic checks
    y_sim_all, theta_sim_all, xi_sim_all = simulated_data
    y_sim_all = np.asarray(y_sim_all)
    theta_sim_all = np.asarray(theta_sim_all)
    xi_sim_all = np.asarray(xi_sim_all)

    if not (y_sim_all.ndim == 2 and theta_sim_all.ndim == 2 and xi_sim_all.ndim == 2):
        raise RuntimeError("simulated_data arrays must be 2D.")
    n_sim0 = y_sim_all.shape[0]
    if theta_sim_all.shape[0] != n_sim0 or xi_sim_all.shape[0] != n_sim0:
        raise RuntimeError("All simulated_data arrays must share the same first dimension.")

    if len(observed_data_list) != len(xi_star_list):
        raise RuntimeError("observed_data_list and xi_star_list must have the same length.")

    # Global NaN filtering to maintain alignment
    sim_valid = (
        ~np.isnan(y_sim_all).any(axis=1)
        & ~np.isnan(theta_sim_all).any(axis=1)
        & ~np.isnan(xi_sim_all).any(axis=1)
    )
    valid_idx_global = np.nonzero(sim_valid)[0]
    y_sim = y_sim_all[sim_valid]
    theta_sim = theta_sim_all[sim_valid]
    xi_sim = xi_sim_all[sim_valid]
    n_sim_valid = y_sim.shape[0]

    per_exp_outputs: List[Dict[str, Any]] = []

    # Iterate experiments
    for e, (y_obs_e, xi_star_e) in enumerate(zip(observed_data_list, xi_star_list)):
        y_obs_e = np.asarray(y_obs_e)
        xi_star_e = np.atleast_1d(np.asarray(xi_star_e))

        # Sanity checks per experiment
        if y_obs_e.ndim != 2:
            raise RuntimeError(f"observed_data_list[{e}] must be 2D (n_obs_e, d_y).")
        if y_obs_e.shape[1] != y_sim.shape[1]:
            raise RuntimeError(f"y dimension mismatch at experiment {e}: "
                               f"observed d_y={y_obs_e.shape[1]} vs simulated d_y={y_sim.shape[1]}.")
        if xi_star_e.size != xi_sim.shape[1]:
            raise RuntimeError(f"xi_star_list[{e}] has incompatible dimensionality: "
                               f"{xi_star_e.size} vs d_xi={xi_sim.shape[1]}.")

        # Clean NaNs from observed rows
        obs_valid = ~np.isnan(y_obs_e).any(axis=1)
        y_obs_e = y_obs_e[obs_valid]
        if y_obs_e.shape[0] == 0:
            raise ValueError(f"Experiment {e}: no valid observed rows after NaN removal.")

        # Design filtering near xi_star_e
        if np.all(np.abs(xi_star_e) <= 1e-12):
            d_inf = np.max(np.abs(xi_sim - xi_star_e), axis=1)
            mask = d_inf <= a_tol
        else:
            d_inf_rel = np.max(np.abs(xi_sim - xi_star_e) / (np.abs(xi_star_e) + 1e-12), axis=1)
            mask = d_inf_rel <= a_tol

        if not np.any(mask):
            raise ValueError(f"Experiment {e}: no simulations within tolerance of xi_star.")

        # Extract filtered arrays and keep mapping to global indices
        y_sim_e = y_sim[mask]
        theta_sim_e = theta_sim[mask]
        # local->global index mapping:
        global_idx_e = valid_idx_global[mask]

        # Standardize outputs using simulated subset of this experiment
        scaler = StandardScaler()
        scaler.fit(y_sim_e)
        y_sim_e_scaled = scaler.transform(y_sim_e)
        y_obs_e_scaled = scaler.transform(y_obs_e)

        # kNN search
        knn_eff = int(min(knn, y_sim_e_scaled.shape[0]))
        if knn_eff < 1:
            raise ValueError(f"Experiment {e}: knn < 1 after filtering.")
        nbrs = NearestNeighbors(n_neighbors=knn_eff, algorithm="auto")
        nbrs.fit(y_sim_e_scaled)
        dist_e, knn_idx_e = nbrs.kneighbors(y_obs_e_scaled)  # (n_obs_e, knn_eff)

        # Retrieve θ-sets and indices
        theta_knn_e = theta_sim_e[knn_idx_e]                      # (n_obs_e, knn_eff, d_theta)
        knn_idx_global_e = global_idx_e[knn_idx_e]                # same shape, indices into original archive


        # Output-space spread (scaled)
        epsilon_e = dist_e[:, -1]                                 # K-th distance per observation
        mean_d_e = dist_e.mean(axis=1)
        std_d_e = dist_e.std(axis=1, ddof=1) if knn_eff > 1 else np.zeros_like(mean_d_e)

        # Input-space spread
        theta_mean_e = theta_knn_e.mean(axis=1)
        theta_std_e = theta_knn_e.std(axis=1, ddof=1) if knn_eff > 1 else np.zeros_like(theta_mean_e)

        theta_centered = theta_knn_e - theta_mean_e[:, None, :]      # (n_obs_e, knn_eff, d_theta)

        # Local covariance per observation (optional but useful)
        theta_cov_e = None
        if return_full_cov:
            n_obs_e, _, d_theta = theta_knn_e.shape
            theta_cov_e = np.empty((n_obs_e, d_theta, d_theta))
            for i in range(n_obs_e):
                X = theta_knn_e[i]
                theta_cov_e[i] = np.cov(X, rowvar=False) if knn_eff > 1 else np.zeros((d_theta, d_theta))


        # θ-space radius = K-th smallest Euclidean distance to the centroid (per observation)
        theta_pairwise = np.linalg.norm(theta_centered, axis=2)      # (n_obs_e, knn_eff)
        theta_eps_e = np.partition(theta_pairwise, knn_eff - 1, axis=1)[:, knn_eff - 1]


        # Optional anisotropy: ratio of largest/smallest eigenvalue of covariance
        # (skip if knn_eff==1)
        anisotropy_e = None
        if knn_eff > 1:
            # Fast eigendecomp per obs using eigh on full cov if available; else approximate via SVD on centered block
            if return_full_cov:
                # use the cov we already computed
                eigvals = np.linalg.eigvalsh(theta_cov_e)            # (n_obs_e, d_theta)
            else:
                # cheap fallback: compute cov on the fly for anisotropy only
                n_obs_e, _, d_theta = theta_knn_e.shape
                eigvals = np.empty((n_obs_e, d_theta))
                for i in range(n_obs_e):
                    Xc = theta_centered[i]
                    cov_i = (Xc.T @ Xc) / max(knn_eff - 1, 1)
                    eigvals[i] = np.linalg.eigvalsh(cov_i)
            anisotropy_e = eigvals[:, -1] / np.maximum(eigvals[:, 0], 1e-12)


        per_exp_outputs.append({
            "xi_star": xi_star_e.copy(),
            "theta_samples": theta_knn_e,
            "knn_indices": knn_idx_e,
            "knn_indices_global": knn_idx_global_e,
            "epsilon": epsilon_e,
            "output_spread": {
                "distances": dist_e,
                "mean_distance": mean_d_e,
                "std_distance": std_d_e,
            },
            "input_spread": {
                "theta_mean": theta_mean_e,
                "theta_std": theta_std_e,
                "theta_cov": theta_cov_e,     # <-- θ-space radius (diagnostic)
                "theta_radius": theta_eps_e,  # <-- θ-space radius (diagnostic)
                "theta_anisotropy": anisotropy_e  # <-- eigenvalue ratio (diagnostic)
            },
            "meta": {
                "knn_eff": knn_eff,
                "n_sim_filtered": y_sim_e.shape[0],
                "scaler_mean_": scaler.mean_.copy(),
                "scaler_scale_": scaler.scale_.copy(),
            },
        })





    result: Dict[str, Any] = {
        "per_experiment": per_exp_outputs,
        "meta": {
            "n_experiments": len(per_exp_outputs),
            "n_sim_initial": n_sim0,
            "n_sim_after_nan_filter": n_sim_valid,
        },
    }

    if combine_results:
        # Build stacked views; pad to max_knn for rectangular arrays if K varies by experiment
        max_knn = max(pe["meta"]["knn_eff"] for pe in per_exp_outputs)
        d_theta = per_exp_outputs[0]["theta_samples"].shape[-1]
        stacked_theta = []
        stacked_eps = []
        stacked_exp_idx = []
        for e, pe in enumerate(per_exp_outputs):
            th = pe["theta_samples"]  # (n_obs_e, knn_e, d_theta)
            n_obs_e, knn_e, _ = th.shape
            if knn_e < max_knn:
                pad = np.full((n_obs_e, max_knn - knn_e, d_theta), np.nan, dtype=th.dtype)
                th = np.concatenate([th, pad], axis=1)
            stacked_theta.append(th)
            stacked_eps.append(pe["epsilon"])
            stacked_exp_idx.append(np.full(n_obs_e, e, dtype=int))
        result["combined"] = {
            "theta_samples": np.concatenate(stacked_theta, axis=0),   # (sum n_obs_e, max_knn, d_theta)
            "epsilon": np.concatenate(stacked_eps, axis=0),           # (sum n_obs_e,)
            "experiment_index": np.concatenate(stacked_exp_idx, axis=0),
        }

    return result