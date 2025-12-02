from typing import NamedTuple
import jax.numpy as jnp

class SimulationParams(NamedTuple):
    """Parameters in simulation that are never updated"""
    STEP_SIZE: float
    MAXIMUM_DT_INTEGRATION: float
    dt_integration: float
    n_integration: int
    n_steps: int
    t0: float
    tf: float
    reestimate_window_size: int
    prediction_horizon: int
    optimisation_params: tuple
    dual_inner_loops: int
    param_transition_magic_number: float # Scale process noise on parameters to larger values

class DisturbanceParams(NamedTuple):
    times: jnp.ndarray
    disturbances: jnp.ndarray

class PlantParams(NamedTuple):
    """Parameters for the plant"""
    model_params: tuple
    disturbance_params: tuple
    noise_params: tuple

class PlantState(NamedTuple):
    """State of the plant"""
    x: jnp.ndarray

class OptimisationParams(NamedTuple):
    """Parameters for optimisation"""
    n_generations: int
    population_size: int
    mutation_rate: float
    mutation_scale: float
    mask: tuple
    mask_indices: tuple

class OptimisationConstraints(NamedTuple):
    """
    Constraints for optimisation
    minval: jnp.ndarray
        For clipping parameters to constraints
    maxval: jnp.ndarray
        For clipping parameters to constraints
    replacement_params: jnp.ndarray
        For if a parameter is masked out
    """
    minval: jnp.ndarray
    maxval: jnp.ndarray
    replacement_params: jnp.ndarray

class ControlSystemParams(NamedTuple):
    """Parameters for the controller and observer"""
    controller_params: tuple
    observer_params: tuple
    Q_k: jnp.ndarray
    R_k: jnp.ndarray
    nis_avg_lb: float
    nis_avg_ub: float
    opt_constraints: tuple
    ukf_params: tuple
    Q_param: jnp.ndarray
    R_param: jnp.ndarray

class ControlSystemState(NamedTuple):
    """State of the controller and observer"""
    x_hat: jnp.ndarray
    P: jnp.ndarray
    nis_history: jnp.ndarray
    model_params_hat: jnp.ndarray
    model_params_P_hat: jnp.ndarray
    z_history: jnp.ndarray
    u_history: jnp.ndarray
    x_hat_history: jnp.ndarray
    controller_state: tuple
    mhe_activation: int
    dual_counter: int
