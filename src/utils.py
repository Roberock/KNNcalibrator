from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class ARCHIVE:
    """ Container for simulated data.  Mathematical object
        -------------------
        𝒟^sim = { { (θ^{(i),e}, z_hat^{(i),e}) }_{i=1}^{N_sim^e}, ξ^e }_{e=1}^{n_e}
        ----------
        response : list[np.ndarray] -> response[e] = z_hat^e with shape (N_sim^e, n_z)
        designs : list[np.ndarray]  -> designs[e] = ξ^e with shape (n_xi,) or (1, n_xi)
        input : list[np.ndarray]    -> input[e] = θ^e with shape (N_sim^e, n_theta)
    """
    response: List[np.ndarray] = field(default_factory=list)   # z_hat
    designs: List[np.ndarray] = field(default_factory=list)    # xi
    input: List[np.ndarray] = field(default_factory=list)      # theta

    def add_design_block( self, theta: np.ndarray, z_hat: np.ndarray, xi: ArrayLike, ) -> None:
        theta = np.asarray(theta, dtype=float)
        z_hat = np.asarray(z_hat, dtype=float)
        xi = np.asarray(xi, dtype=float).reshape(-1)
        if theta.ndim != 2:
            raise ValueError("theta must have shape (N, n_theta).")
        if z_hat.ndim != 2:
            raise ValueError("z_hat must have shape (N, n_z).")
        if theta.shape[0] != z_hat.shape[0]:
            raise ValueError("theta and z_hat must have the same number of rows.")
        self.input.append(theta)
        self.response.append(z_hat)
        self.designs.append(xi)

    @property
    def n_designs(self) -> int:
        return len(self.designs)

    @property
    def size(self) -> int:
        return int(sum(block.shape[0] for block in self.input))

    def flatten(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return flattened arrays:
            Z_hat : (N_tot, n_z);  Theta : (N_tot, n_theta); Xi    : (N_tot, n_xi) """
        if self.n_designs == 0:
            raise ValueError("Archive is empty.")
        z_hat = np.vstack(self.response)
        theta = np.vstack(self.input)
        xi = np.vstack([ np.tile(np.asarray(self.designs[e]).reshape(1, -1), (self.input[e].shape[0], 1))
                         for e in range(self.n_designs) ])
        return z_hat, theta, xi

    def get_design_block(self, e: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return (z_hat^e, theta^e, xi^e). """
        return self.response[e], self.input[e], self.designs[e]

@dataclass
class OBSERVATIONS:
    """ Container for empirical observations.
        -------------------
        𝒟^emp = { { z^{(i),e} }_{i=1}^{N_e}, ξ^e }_{e=1}^{n_e}
        ----------
        response : list[np.ndarray] -> response[e] = z^e with shape (N_e, n_z)
        designs : list[np.ndarray] -> designs[e] = ξ^e with shape (n_xi,) or (1, n_xi)  """
    response: List[np.ndarray] = field(default_factory=list)   # z
    designs: List[np.ndarray] = field(default_factory=list)    # xi

    def add_design_block(self, z: np.ndarray, xi: ArrayLike,) -> None:
        z = np.asarray(z, dtype=float)
        xi = np.asarray(xi, dtype=float).reshape(-1)
        if z.ndim != 2:
            raise ValueError("z must have shape (N, n_z).")
        self.response.append(z)
        self.designs.append(xi)

    @property
    def n_designs(self) -> int:
        return len(self.designs)

    def get_design_block(self, e: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Return (z^e, xi^e). """
        return self.response[e], self.designs[e]


@dataclass
class SIMULATOR:
    """ Wrapper for the simulation model M(theta, xi).
        The wrapped model must accept:  model(theta, xi) -> z_hat
        with
            theta : (N, n_theta) or (n_theta,)
            xi    : scalar, (n_xi,), or compatible """
    model: Callable[[np.ndarray, ArrayLike], np.ndarray]

    def simulate(self, theta: np.ndarray, design: ArrayLike) -> np.ndarray:
        """ Simulate responses for a batch of theta at a given design xi.
            theta : np.ndarray shape (N, n_theta) or (n_theta,)
            design : array-like  ξ
            z_hat : np.ndarray shape (N, n_z) """
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)

        z_hat = np.asarray(self.model(theta, design), dtype=float)
        if z_hat.ndim == 1:
            z_hat = z_hat.reshape(1, -1)
        return z_hat

    def mc_simulator(
        self,
        N: int,
        design: ArrayLike,
        pdf: Optional[Callable[[int], np.ndarray]] = None,
        bounds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Sample theta and simulate z_hat = M(theta, xi).
        N : int  Number of Monte Carlo samples.
        design : array-like ξ
        pdf : callable, optional Sampling rule for theta. Must return (N, n_theta).
        bounds : np.ndarray, optional  Uniform bounds of shape (n_theta, 2), used if pdf is None.
        Returns
        -------
        theta : np.ndarray shape (N, n_theta)
        z_hat : np.ndarray shape (N, n_z)
        """
        if pdf is not None:
            theta = np.asarray(pdf(N), dtype=float)
        else:
            if bounds is None:
                raise ValueError("If pdf is None, bounds must be provided.")
            bounds = np.asarray(bounds, dtype=float)
            low = bounds[:, 0]
            high = bounds[:, 1]
            theta = np.random.default_rng().uniform(low, high, size=(N, bounds.shape[0]))

        z_hat = self.simulate(theta, design)
        return theta, z_hat



def build_archive(
    simulator: SIMULATOR,
    designs: Sequence[ArrayLike],
    N_sim: int,
    pdf: Optional[Callable[[int], np.ndarray]] = None,
    bounds: Optional[np.ndarray] = None,
) -> ARCHIVE:
    archive = ARCHIVE()
    for xi in designs:
        theta_e, z_hat_e = simulator.mc_simulator(N=N_sim, design=xi, pdf=pdf, bounds=bounds)
        archive.add_design_block(theta=theta_e, z_hat=z_hat_e, xi=xi)
    return archive



def estimate_theta_knn_xi(observed_data, simulated_y, simulated_theta, K = 10):
    scaler = StandardScaler()
    y_sim_scaled = scaler.fit_transform(simulated_y)
    y_obs_scaled = scaler.transform(observed_data)

    neigh = NearestNeighbors(n_neighbors=min(K, simulated_y.shape[0]), metric="euclidean")
    neigh.fit(y_sim_scaled)
    dist, knn_idx = neigh.kneighbors(y_obs_scaled)

    theta_post = np.vstack([simulated_theta[idx] for idx in knn_idx])
    response_set = np.vstack([simulated_y[idx] for idx in knn_idx])

    # Per-observation local radii in response space
    # q95 = np.quantile(dist, 0.95, axis=1)
    # q100 = np.max(dist, axis=1)   # same as dist[:, -1] since kneighbors sorts distances
    radii = np.max(dist, axis=1)

    return theta_post, response_set, radii