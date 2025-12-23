
## Black-Box UQ, Calibration, and Design of Experiments (DOE)

A compact Python toolkit for uncertainty quantification (UQ), calibration (stochastic model updating), and design of experiments (DOE) for **black-box computational models** using **empirical data** and (optionally) a large **simulation archive**.
This repository is structured to remain model-agnostic:
the computational model is treated as a callable that maps inputs to outputs, and the inference/DOE machinery operates on standardized interfaces.

 
`pip install -e ".[plot,notebook]"`



### Key ideas

- **Black-box simulator interface**: standardize calls of the form  
  $$(x_a, x_e, x_c) \rightarrow y$$
  where:
  - $x_a$: aleatoric (repeat-to-repeat variability)
  - $x_e$: epistemic (fixed-but-unknown quantities)
  - $x_c$: design / controllable variables
  - $y$: observed or simulated response (can be multi-dimensional, time series, trajectories)

- **Uncertainty models**:
  - `AleatoricModel`: probability distribution for $x_a$
  - `EpistemicModel`: belief (set or distribution) over epistemic quantities (including parameters of aleatoric model)
  - `NestedAEModel`: hierarchical combination: epistemic over $(x_e,\phi)$ and aleatoric $x_a\sim p(x_a\mid\phi)$

- **Calibration**:
  Update epistemic belief using empirical observations and either:
  - a simulation archive (given-data / kNN conditioning / ABC-like)
  - a direct simulator (if available for on-demand simulations)

- **DOE**:
  Select new design points $x_c$ to reduce uncertainty, e.g. via Expected Information Gain (EIG) or other criteria.


### Repository layout

* resources/  # data and models
* src/ # core library code (models, calibration, DOE, plotting)
* examples/ # tutorial notebooks and workflows (kept minimal and separate)

### Workflow overview

Typical workflow (sequential or batch):

1. **Define / wrap the simulator** as a `Simulator`:
   - Ensure consistent input/output shapes.
2. **Specify uncertainty** using:
   - an `AleatoricModel` and/or an `EpistemicModel` and/or  a `NestedAEModel`. 
3. **Load/build a simulation archive**:
   - Large set of tuples $\{(x_a, x_e, x_c, y)\}$ for fast conditioning / reuse.
4. **Calibrate** using empirical observations:
   - Obtain an updated epistemic belief (posterior-like representation).
5. **DOE selection**:
   - Score candidate designs using a criterion (e.g., EIG).
   - Choose next $x_c$ (sequential) or a batch of designs.
6. **Acquire new empirical data**:
   - Add new observations.
7. **Repeat** until the experimental budget is exhausted.


## Data conventions

All internal components should use consistent array shapes.

- Designs:
  - `xc`: shape `(n, d_c)` (or `(d_c,)` for a single design)
- Aleatoric samples:
  - `xa`: shape `(n, d_a)`
- Epistemic samples:
  - `xe`: shape `(n, d_e)`
  - `theta_e`: generic epistemic vector, shape `(n, d_theta)`
- Outputs:
  - `y`: shape `(n, d_y)` for feature vectors, or `(n, T, d_y)` for trajectories  
    (the simulator wrapper decides the canonical form, but the rest of the library expects consistent shapes)


## Core modules (src)

### `src/simulator.py`
Defines the black-box simulator interface:
- `Simulator.evaluate(xa, xe, xc) -> y`

### `src/uncertaintymodel.py`
Defines uncertainty abstractions:
- `AleatoricModel`: `sample(n) -> xa`, `update(params) -> None`
- `EpistemicModel`: `sample(n) -> theta_e`, `get_set(alpha) -> set_repr`, `update(params) -> None`
- `NestedAEModel`: `nested_samples(ne, na) -> {theta_e, xe, xa}`

### `src/forward.py`
Uncertainty propagation:
- `forward_propagation(model, simulator, xc, ...) -> {y, meta, raw}`

### `src/backward.py`
Abstract classes for input calibration:
- `Calibrator`  

Specific class that run given-data calibration using kNN conditioning on a simulation archive:
- `KNNCalibrator` 
 
### `src/doe.py`
Design of experiments functionality: 
- `DOEEngine.select_next(state, ...) -> xc_next`
- `DOEEngine.select_batch(state, n_batch, ...) -> xc_batch`

### `src/scores.py`
Scores for calibration, doe, etc: 
 

### `src/plotting.py`
Visualization utilities (all optional, no hard dependency for core algorithms).

### `src/types.py`, `src/utils.py`
Shared types, shape utilities, timing tools, hashing designs, etc.


## What is “posterior” in this repository?

This library supports different posterior-like representations depending on calibration method:

- **Particle posterior**: samples (and optional weights) approximating $p(\theta_e \mid D)$
- **Set posterior**: an updated feasible set $E(D)$ for epistemic quantities
- **Hybrid**: weighted particles + derived credible sets

Calibration code should return a standardized dictionary with:
- `posterior`: posterior representation object or dict
- `meta`: method metadata and diagnostics


## Supported problem types

- Scalar outputs, vector outputs, trajectories
- Multiple replicate observations per design
- Sequential (adaptive) experiments and batch design selection
- Archive-based calibration (fast) and simulator-based calibration (general)


## Installation

This project uses `pyproject.toml` (recommended: `pip install -e .` in a virtual environment).

Dependencies are kept minimal. Optional dependencies may be required for:
- kNN (scikit-learn)
- KDE / density tools (scipy)
- plotting (matplotlib / seaborn)


## Development guidelines

- Keep core logic in `src/` and notebooks in `examples/`.
- Keep use-case-specific assets under `resources/`.
- Avoid hard-coding any simulator-specific assumptions into `src/`.
- Enforce shape conventions at module boundaries (simulator wrapper and data loaders).


## License

See `LICENSE`.
