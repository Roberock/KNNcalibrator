from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol, Union, Callable
import numpy as np


# =========================
# Aleatoric model
# =========================
class AleatoricModel(ABC):
    """
    Aleatoric uncertainty model: defines a pdf for x_a.

    Examples:
      - x_a ~ Normal(mu, sigma)
      - x_a ~ MVN(mu, Sigma)
      - x_a ~ BetaMixtureCopula(...)
    """

    name: str = "aleatoric model"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters: Dict[str, Any] = {}
        if parameters is not None:
            self.update(parameters)

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of x_a."""
        raise NotImplementedError

    @abstractmethod
    def sample_xa(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw x_a samples. Returns shape (n, dim)."""
        raise NotImplementedError

    def logpdf_xa(self, xa: np.ndarray) -> np.ndarray:
        """Optional: log p(x_a | phi). Returns shape (n,)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement logpdf_xa().")

    def update(self, parameters: Dict[str, Any]) -> None:
        """Update aleatoric parameters phi and refresh caches."""
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dict.")
        self.parameters = dict(parameters)
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild caches if needed (e.g., Cholesky factors)."""
        return


# Example concrete aleatoric model (2D iid Normal)
class NormalIID2D(AleatoricModel):
    name = "NormalIID2D"

    @property
    def dim(self) -> int:
        return 2

    def _refresh(self) -> None:
        self.mu = float(self.parameters.get("mu", 0.0))
        self.sigma = float(self.parameters.get("sigma", 1.0))
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

    def sample_xa(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=(n, 2))


# =========================
# Epistemic model
# =========================
class EpistemicModel(ABC):
    """
    Epistemic uncertainty model: defines a belief/set over fixed unknown quantities.

    Typical epistemic quantities include:
      - x_e (fixed-but-unknown simulator parameters)
      - phi = parameters of the aleatoric model
      - discrepancy params, hyperparams, etc.

    This base class supports two common representations:
      A) Set-based: parameter bounds / feasible set (no probabilities)
      B) Particle-based: weighted samples approximating a distribution
    """

    name: str = "epistemic model"
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters: Dict[str, Any] = {}
        if parameters is not None:
            self.update(parameters)

    @abstractmethod
    def update(self, parameters: Dict[str, Any]) -> None:
        """Update epistemic belief representation."""
        raise NotImplementedError

    @abstractmethod
    def sample_theta_e(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample epistemic parameters theta_e (e.g., [phi, x_e]).
        If set-based, sampling is from the set (e.g., uniform).
        Returns shape (n, d_e).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of theta_e."""
        raise NotImplementedError


@dataclass
class BoxSetEpistemic(EpistemicModel):
    """
    Set-based epistemic model: theta_e lies in a box [lb, ub].
    Optionally you can later refine this box (model updating).

    parameters:
      - lb: (d,)
      - ub: (d,)
      - names: optional list of length d
    """
    name: str = "box-set epistemic"
    def __init__(self, lb: np.ndarray, ub: np.ndarray, names: Optional[list[str]] = None):
        lb = np.asarray(lb, float).reshape(-1)
        ub = np.asarray(ub, float).reshape(-1)
        if lb.shape != ub.shape:
            raise ValueError("lb/ub must have same shape")
        if np.any(ub <= lb):
            raise ValueError("Need ub > lb componentwise")
        self.lb = lb
        self.ub = ub
        self.names = names or [f"theta_e{i}" for i in range(lb.size)]
        self.parameters = {"lb": self.lb, "ub": self.ub, "names": self.names}

    @property
    def dim(self) -> int:
        return int(self.lb.size)

    def update(self, parameters: Dict[str, Any]) -> None:
        # simplest update: replace bounds
        lb = np.asarray(parameters.get("lb", self.lb), float).reshape(-1)
        ub = np.asarray(parameters.get("ub", self.ub), float).reshape(-1)
        if lb.shape != ub.shape:
            raise ValueError("lb/ub shape mismatch")
        self.lb, self.ub = lb, ub
        self.parameters = {"lb": self.lb, "ub": self.ub, "names": self.names}

    def sample_theta_e(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=(n, self.dim))
        return self.lb + u * (self.ub - self.lb)


class ParticleEpistemic(EpistemicModel):
    """
    Particle-based epistemic model: theta_e has weighted particles.

    parameters:
      - particles: (M, d)
      - weights:   (M,) nonnegative, sums to 1
      - names: optional
    """
    name = "particle epistemic"

    def __init__(self, particles: np.ndarray, weights: Optional[np.ndarray] = None, names: Optional[list[str]] = None):
        particles = np.asarray(particles, float)
        if particles.ndim != 2:
            raise ValueError("particles must be (M, d)")
        self.particles = particles
        M = particles.shape[0]
        if weights is None:
            weights = np.ones(M) / M
        weights = np.asarray(weights, float).reshape(-1)
        if weights.shape[0] != M:
            raise ValueError("weights length mismatch")
        weights = np.clip(weights, 0, np.inf)
        weights = weights / (weights.sum() if weights.sum() > 0 else M)
        self.weights = weights
        self.names = names or [f"theta_e{i}" for i in range(particles.shape[1])]
        self.parameters = {"particles": self.particles, "weights": self.weights, "names": self.names}

    @property
    def dim(self) -> int:
        return int(self.particles.shape[1])

    def update(self, parameters: Dict[str, Any]) -> None:
        # allow updating particles/weights after calibration
        if "particles" in parameters:
            self.particles = np.asarray(parameters["particles"], float)
        if "weights" in parameters and parameters["weights"] is not None:
            w = np.asarray(parameters["weights"], float).reshape(-1)
            w = np.clip(w, 0, np.inf)
            w = w / (w.sum() if w.sum() > 0 else len(w))
            self.weights = w
        self.parameters = {"particles": self.particles, "weights": self.weights, "names": self.names}

    def sample_theta_e(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(self.particles.shape[0], size=n, replace=True, p=self.weights)
        return self.particles[idx]


# =========================
# Glue: full uncertainty state
# =========================


@dataclass
class UncertaintyState:
    aleatoric: Any        # AleatoricModel
    epistemic: Any        # EpistemicModel
    extract_phi: Callable[[np.ndarray], Dict[str, np.ndarray]]
    # extract_phi maps theta_e -> dict of aleatoric parameters (phi)

    def sample_world(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample ONE epistemic 'world' (fixed parameters)."""
        if rng is None:
            rng = np.random.default_rng()
        theta_e = self.epistemic.sample_theta_e(1, rng=rng)[0]   # (d_e,)
        return theta_e

    def sample_replicates(self, theta_e: np.ndarray, n_rep: int,
                          rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        """Sample x_a replicates conditional on the same theta_e."""
        if rng is None:
            rng = np.random.default_rng()

        phi = self.extract_phi(theta_e[None, :])  # dict of arrays shape (1, ...)
        # update aleatoric with that world’s phi
        self.aleatoric.update({k: float(np.asarray(v).reshape(-1)[0]) for k, v in phi.items()})
        xa = self.aleatoric.sample_xa(n_rep, rng=rng)            # (n_rep, d_a)

        return {"theta_e": np.repeat(theta_e[None, :], n_rep, axis=0),
                "xa": xa}
