"""Observers for closed-loop coevolution simulations"""
from typing import NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES as ES
from .plant_model import Plant
from .numerical_methods import reRK4, rk4
from .types import ControlSystemState, SimulationParams, ControlSystemParams

class DUKFObserver:
    """
    Dual Unscented Kalman Filter for joint state and parameter estimation
    """
    def _compute_ukf_weights(n, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        Returns (Wm, Wc, lambda_, c) where:
        - Wm: weights for mean (shape (2n+1,))
        - Wc: weights for covariance
        - lambda_: scaling
        - c: n + lambda_
        """
        lambda_ = alpha**2 * (n + kappa) - n
        c = n + lambda_
        Wm0 = lambda_ / c
        Wc0 = Wm0 + (1 - alpha**2 + beta)
        W = 1.0 / (2.0 * c)
        Wm = jnp.concatenate([jnp.array([Wm0]), jnp.full(2 * n, W)])
        Wc = jnp.concatenate([jnp.array([Wc0]), jnp.full(2 * n, W)])
        return Wm, Wc, lambda_, c

    def _sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        Generate 2n+1 sigma points for state mean x and covariance P.
        Return shape: (2n+1, n)
        """
        n = x.shape[0]
        Wm, Wc, lambda_, c = DUKFObserver._compute_ukf_weights(n, alpha, beta, kappa)
        # scaled cholesky
        jitter = 1e-6
        scaled_P = (c) * P
        L = jnp.linalg.cholesky(scaled_P + jitter * jnp.eye(n))
        # first sigma is mean
        SP0 = x
        # other sigma points: x +/- columns of L
        L_cols = L.T  # shape (n, n) columns are rows of L.T
        plus = x + L_cols
        minus = x - L_cols
        sigma = jnp.vstack([SP0, plus, minus])
        return sigma, Wm, Wc

    def _ukf_predict(x, P, theta, f, u, Q, simulation_params,
                     alpha=1e-3, beta=2.0, kappa=0.0):
        """
        UKF prediction step.
        - x: state mean (n,)
        - P: state covariance (n,n)
        - f: dynamics function f(x) -> x_pred (n,)
        - Q: process noise covariance (n,n)
        returns (x_pred, P_pred, transformed_sigma_points)
        """
        n = x.shape[0]
        sigma, Wm, Wc = DUKFObserver._sigma_points(x, P, alpha, beta, kappa)

        # propagate sigma points through dynamics
        f_vmap = jax.vmap(f, in_axes=(0, None, None, None))  # Vectorize over sigma points, keep u fixed
        Xp = f_vmap(sigma, theta, u, simulation_params)  # Pass both sigma and u

        x_pred = jnp.sum(Wm[:, None] * Xp, axis=0)
        dx = Xp - x_pred[None, :]
        P_pred = jnp.einsum('i,ij,ik->jk', Wc, dx, dx) + Q
        return x_pred, P_pred, Xp, Wm, Wc

    def _ukf_correct(x_pred, P_pred, theta,
                     Xp, z, h, u, R, Wm, Wc, simulation_params):
        """
        UKF update step with measurement z.
        - x_pred: predicted mean (n,)
        - P_pred: predicted covariance (n,n)
        - Xp: propagated sigma points (2n+1, n)
        - z: measurement vector (m,)
        - h: measurement function h(x) -> z (m,)
        - R: measurement noise covariance (m,m)
        - Wm, Wc: weights from sigma generation (both length 2n+1)
        returns (x_upd, P_upd)
        """
        # compute predicted measurements for each sigma
        h_vmap = jax.vmap(h, in_axes=(0, None, None))  # Vectorize over sigma points, keep u fixed
        Zp = h_vmap(Xp, theta, u)  # Pass both Xp and u
        z_pred = jnp.sum(Wm[:, None] * Zp, axis=0)

        # innovations
        dz = Zp - z_pred[None, :]

        # measurement covariance Pzz
        Pzz = jnp.einsum('i,ij,ik->jk', Wc, dz, dz) + R

        # cross-covariance Pxz
        dx = Xp - x_pred[None, :]
        Pxz = jnp.einsum('i,ij,ik->jk', Wc, dx, dz)

        # Kalman gain
        # Solve Pzz.T * K.T = Pxz.T  for K.T  (better numeric stability than explicit inverse)
        Kt = jnp.linalg.solve(Pzz.T, Pxz.T)
        K = Kt.T

        # update
        y = z - z_pred
        x_upd = x_pred + jnp.matmul(K, y).squeeze() # shape (n,) instead of (n,1)
        P_upd = P_pred - K @ Pzz @ K.T
        
        # ensure symmetry
        P_upd = 0.5 * (P_upd + P_upd.T)
        return x_upd, P_upd

    def _ukf_param_correct(x_pred,
                           P_pred,
                           Xp, z, h, R, Wm,  Wc,
                           control_system_state,
                           control_system_params,
                           simulation_params):
        """
        UKF update step with measurement z.
        - x_pred: predicted mean (n,)
        - P_pred: predicted covariance (n,n)
        - Xp: propagated sigma points (2n+1, n)
        - z: measurement vector (m,)
        - h: measurement function h(x) -> z (m,)
        - R: measurement noise covariance (m,m)
        - Wm, Wc: weights from sigma generation (both length 2n+1)
        returns (x_upd, P_upd)

        Same function as _ukf_correct but h_param_dukf has different 
        signature:
        h_param_dukf(theta_hat,
                     control_system_state,
                     control_system_params,
                     simulation_params)
        """
        # compute predicted measurements for each sigma
        h_vmap = jax.vmap(h, in_axes=(0, None, None))  # Vectorize over sigma points
        Zp = h_vmap(Xp, control_system_state, simulation_params)
        z_pred = jnp.sum(Wm[:, None] * Zp, axis=0)

        # innovations
        dz = Zp - z_pred[None, :]

        # measurement covariance Pzz
        Pzz = jnp.einsum('i,ij,ik->jk', Wc, dz, dz) + R

        # cross-covariance Pxz
        dx = Xp - x_pred[None, :]
        Pxz = jnp.einsum('i,ij,ik->jk', Wc, dx, dz)

        # Kalman gain
        # Solve Pzz.T * K.T = Pxz.T  for K.T  (better numeric stability than explicit inverse)
        Kt = jnp.linalg.solve(Pzz.T, Pxz.T)
        K = Kt.T

        # update
        y = z - z_pred
        x_upd = x_pred + jnp.matmul(K, y).squeeze() # shape (n,) instead of (n,1)
        P_upd = P_pred - K @ Pzz @ K.T
        
        # ensure symmetry
        P_upd = 0.5 * (P_upd + P_upd.T)
        # jax.debug.breakpoint()
        return x_upd, P_upd

    def _ukf_state_update(control_system_state, f_state, h_state, u, z, control_system_params, simulation_params):
        """
        UKF update step for state only, keeping parameters fixed.
            - control_system_state: ControlSystemState named tuple
            - f: dynamics function f(x) -> x_pred (n,)
            - h: measurement function h(x) -> z (m,)
            - u: control input
            - z: measurement vector (m,)
            - control_system_params: ControlSystemParams named tuple
        returns updated control_system_state with new x_hat and P_hat
        """
        # UKF update for state
        x_pred, P_pred, Xp, Wm, Wc = DUKFObserver._ukf_predict(control_system_state.x_hat,
                                                               control_system_state.P,
                                                    control_system_state.model_params_hat,
                                                    f_state,
                                                    u, control_system_params.Q_k,
                                                    simulation_params,
                                                    control_system_params.ukf_params[0],
                                                    control_system_params.ukf_params[1],
                                                    control_system_params.ukf_params[2])
        # UKF update with measurement
        x_hat_ukf, P_hat_ukf = DUKFObserver._ukf_correct(x_pred, P_pred,
                                                         control_system_state.model_params_hat,
                                                         Xp, z, h_state, u,
                                                         control_system_params.R_k,
                                                         Wm, Wc,
                                                         simulation_params)

        # Repackage
        control_system_state = control_system_state._replace(
            x_hat=x_hat_ukf,
            P=P_hat_ukf)
        return control_system_state

    def _ukf_param_update(f_param, h_param, u, z, control_system_params, simulation_params, control_system_state):
        """
        UKF update step for parameters only, keeping state fixed.
            - f: dynamics function f(x) -> x_pred (n,)
            - h: measurement function h(x) -> z (m,)
            - u: control input
            - z: measurement vector (m,)
            - dukf_params: DualUKFParams named tuple
            - dukf_state: DualUKFState named tuple NOTE wild order in order to use in partial
        returns updated dukf_state with new x_hat and P_hat
        """
        # UKF update for state
        theta_pred, P_theta_pred, Xp, Wm, Wc = DUKFObserver._ukf_predict(control_system_state.model_params_hat,
                                                control_system_state.model_params_P_hat,
                                                control_system_state.x_hat,
                                                f_param,
                                                u,
                                                control_system_params.Q_param,
                                                simulation_params,
                                                control_system_params.ukf_params[0],
                                                control_system_params.ukf_params[1],
                                                control_system_params.ukf_params[2])
        # UKF update with measurement
        theta_hat_ukf, P_theta_hat_ukf = DUKFObserver._ukf_param_correct(theta_pred,
                                                                P_theta_pred,
                                                                Xp, z, h_param,
                                                                control_system_params.R_param,
                                                                Wm, Wc,
                                                                control_system_state,
                                                                control_system_params,
                                                                simulation_params)

        # theta_hat_ukf = jnp.clip(theta_hat_ukf, # OPTION TO CLIP PARAMETERS
        #                 jnp.array([50.0, jnp.log(1e-12), 0.1, 0.1]),
        #                 jnp.array([200.0, jnp.log(1e-6), 2.0, 3.0]))
        theta_hat_ukf = jnp.clip(theta_hat_ukf, # OPTION TO CLIP PARAMETERS IF USING FILTERED NOISE TRANSITION
                        jnp.array([90.0, jnp.log(1e-12), 0.499, 0.1, -1e3, -1e3, -1e3, -1e3]),
                        jnp.array([110.0, jnp.log(1e-6), 0.501, 3.0, 1e3, 1e3, 1e3, 1e3]))

        control_system_state = control_system_state._replace(model_params_hat=theta_hat_ukf,
                                                             model_params_P_hat=P_theta_hat_ukf)
        return control_system_state

    def dukf_update(control_system_state, z, f_state, h_state, f_param, h_param,
                    u, control_system_params, simulation_params):
        """
        Dual UKF update step for joint state and parameter estimation.
            - control_system_state: ControlSystemState NamedTuple
            - z: measurement vector (m,)
            - h: measurement function h(x) -> z (m,)
            - u: control input
            - control_system_params: ControlSystemParams NamedTuple
            - simulation_params: SimulationParams NamedTuple
        returns updated control_system_state with new x_hat, P_hat, theta_hat, P_theta_hat
        """
        # UKF update with measurement
        control_system_state = DUKFObserver._ukf_state_update(control_system_state, f_state,
                                                              h_state, u, z, control_system_params,
                                                              simulation_params)

        # If dual_counter is 0, perform full UKF update
        parameter_update_partial = partial(DUKFObserver._ukf_param_update,
                                        f_param, h_param, u, z,
                                        control_system_params, simulation_params)
        condition = control_system_state.dual_counter == 0
        control_system_state = jax.lax.cond(condition,
                        parameter_update_partial,
                        lambda x: x,
                        control_system_state)

        # Advance dual counter and wrap around
        dual_counter_new = (control_system_state.dual_counter + 1) % simulation_params.dual_inner_loops
        control_system_state = control_system_state._replace(dual_counter=dual_counter_new)

        return control_system_state
    
    def f_state_dukf(x_hat, theta_hat, u, simulation_params):
        """Transition function for state UKF"""
        _, traj = rk4(
            Plant.sllos_dukf,
            [0.0, simulation_params.STEP_SIZE],
            x_hat,
            N=simulation_params.n_integration,
            args=(u, jnp.zeros_like(x_hat), theta_hat),
        )
        return traj[-1, :]
    def f_param_dukf(theta_hat, x_hat, u, simulation_params):
        """Transition function for parameter UKF
        Currently implicitly a random walk with Gaussian noise
        Gaussian noise enters through Q_param in the UKF prediction step
        """
        new_theta_hat_params = theta_hat[:4] + theta_hat[4:] \
            * simulation_params.param_transition_magic_number
        return jnp.concatenate([new_theta_hat_params, theta_hat[4:]])
    def h_state_dukf(x_hat, theta_hat, u):
        """Observation function for state UKF"""
        return Plant.sensor_one_species(0., x_hat, u, 0.)
    def h_param_dukf(theta_hat,
                     control_system_state,
                     simulation_params):
        """Observation function for parameter UKF
        Need to integrate forward from previous state under theta_hat
        Number of integration steps is dual_inner_loops
        """
        def scan_step(x_prev, index):
            u = jnp.take(control_system_state.u_history, index)
            theta_hat_interp = theta_hat[:4] + theta_hat[4:] \
                * simulation_params.param_transition_magic_number * index
            x_next = DUKFObserver.f_state_dukf(x_prev, theta_hat_interp, u, simulation_params)
            return x_next, None
        init_carry = control_system_state.x_hat_history[simulation_params.dual_inner_loops - 1, :]
        x_hat, _ = jax.lax.scan(scan_step,
                             init_carry,
                             xs=jnp.arange(simulation_params.dual_inner_loops - 1, -1, -1))

        z = Plant.sensor_one_species(0., x_hat, control_system_state.u_history[0], 0.)
        return z

class Observer:
    def update(key,
               t,
               u,
               z,
               control_system_state,
               control_system_params,
               simulation_params):
        """
        Wrapper for EKF update and MHE update
        Note observation is 1 element 1D array

        t: time
        u: control input
        z: jnp.array (1D, with 1 element)
            observation
        observer_state: tuple (x, P)
            x: estimated state
            P: state covariance
        observer_params: tuple (f, h, Q, R, dt)
            f: Plant model
            h: Plant sensor
            Q: process noise covariance
            R: observation noise covariance
            dt: time step
        model_params: tuple
            Passed to Plant model f
        """
        # State observer update
        # x_hat, P, nis = Observer.ekf_update(t,
        #                                 u,
        #                                 z,
        #                                 (control_system_state.x_hat, control_system_state.P), # ekf state
        #                                 control_system_state.model_params_hat, # current estimate for linearisation
        #                                 control_system_params,
        #                                 simulation_params)
        # Alternatively, use UKF
        # x_hat, P, nis = Observer.ukf_update(t,
        #                                     u,
        #                                     z,
        #                                     (control_system_state.x_hat, control_system_state.P),
        #                                     control_system_state.model_params_hat,
        #                                     control_system_params,
        #                                     simulation_params)
        # Alternatively, use DUKF
        # NOTE need to comment out nis history, mhe, and some repackaging below for DUKF
        control_system_state = DUKFObserver.dukf_update(control_system_state,
                                                    z,
                                                    DUKFObserver.f_state_dukf,
                                                    DUKFObserver.h_state_dukf,
                                                    DUKFObserver.f_param_dukf,
                                                    DUKFObserver.h_param_dukf,
                                                    u,
                                                    control_system_params,
                                                    simulation_params)

        # Update histories
        # Circular buffer of Normalised Innovations Squared - FOR EKF, UKF
        # Not used for DUKF currently
        # nis_history = jnp.roll(control_system_state.nis_history, 1)
        # nis_history = nis_history.at[0].set(nis) # nis is 1 element 2D array
        # nis_sum = jnp.sum(nis_history)
        # Circular buffer of observations
        z_history = jnp.roll(control_system_state.z_history, 1)
        z = z.squeeze() # z is 1 element array
        z_history = z_history.at[0].set(z)
        # Circular buffer of inputs
        u_history = jnp.roll(control_system_state.u_history, 1)
        u_history = u_history.at[0].set(u)
        # Circular buffer of state estimates
        x_hat_history = jnp.roll(control_system_state.x_hat_history, 1, axis=0)
        # x_hat_history = x_hat_history.at[0, :].set(x_hat) # FOR EKF, UKF
        x_hat_history = x_hat_history.at[0, :].set(control_system_state.x_hat) # FOR DUKF

        # Repacking state
        # Repackage in ControlSystemState for EKF, UKF
        # control_system_state = control_system_state._replace(x_hat=x_hat,
        #                                                     P=P,
        #                                                     nis_history=nis_history,
        #                                                     z_history=z_history,
        #                                                     u_history=u_history,
        #                                                     x_hat_history=x_hat_history)
        # Repackage in ControlSystemState for DUKF
        control_system_state = control_system_state._replace(z_history=z_history,
                                                            u_history=u_history,
                                                            x_hat_history=x_hat_history)

        # MHE update condition - FOR EKF, UKF
        # condition = nis_sum > control_system_params.nis_avg_ub # only care about nis over bounds
        # # Additional condition that MHE only activates once - DOES NOT HELP
        # # condition = jnp.logical_and(condition,
        # #                             control_system_state.mhe_activation < 1)
        # # Uncomment to use MHE with evosax
        # mhe_es_partial = partial(Observer._mhe_es_masked,
        #                          control_system_params=control_system_params,
        #                          simulation_params=simulation_params,
        #                          key=key)
        # control_system_state = jax.lax.cond(condition,
        #                  mhe_es_partial,
        #                  lambda x: x,
        #                  control_system_state)

        return control_system_state

    def ekf_update(t,
                   u,
                   z,
                   ekf_state,
                   model_params_hat,
                   control_system_params,
                   simulation_params):
        """
        Discrete-time Extended Kalman Filter (EKF) for JAX
        Stateless implementation

        Uses notation of Optimal State Estimation by Dan Simon

        Parameters
        ----------
        t: time
        u: control input
        z: observation
        observer_state: tuple (x, P)
            x: estimated state
            P: state covariance
        observer_params: tuple (f, h, Q, R, dt)
            f: Plant model
            h: Plant sensor
            Q_k_minus_1: process noise covariance 
                If it's not updated, why does it have to be indexed?
                I assume it's because it becomes relevant when system is time-varying
            R_k: observation noise covariance
            dt: time step
        model_params: tuple
            Passed to Plant model f
        Returns
        ---------
        ekf_state: tuple (x_hat_plus, P_k_plus, nis)
            x_hat_plus: updated state estimate
            P_k_plus: updated state covariance
            nis: normalised innovation
        
        """

        def ode_to_integrate(t, y, args) -> jnp.array:
            """
            Wrapper function for the system dynamics
            """
            u, w, model_params = args
            return Plant.ode_model_one_species(t, y, u, w, model_params)

        def state_transition_function(t, y, u, w, model_params):
            """
            State transition function
            """
            _, x_hat_minus_k = reRK4(ode_to_integrate,
                                              (t, t+simulation_params.STEP_SIZE),
                                              x_hat_plus_k_minus_1,
                                              N=simulation_params.n_integration,
                                              args=((u, w, model_params),))
            x_hat_minus_k = x_hat_minus_k[-1, :] # Get the last value
            return x_hat_minus_k

        # Unpack the state and parameters
        x_hat_plus_k_minus_1, P_plus_k_minus_1 = ekf_state
        # Linearise the system (state transition function, not ode model!) at current point
        w  = jnp.array([0.])
        v  = jnp.array([0.])
        F_k_minus_1 = jax.jacfwd(state_transition_function, 1)(t,
                                                               x_hat_plus_k_minus_1,
                                                               u,
                                                               w,
                                                               model_params_hat)
        # Size should be (n_states, n_states)
        L_k_minus_1 = jax.jacfwd(state_transition_function, 3)(t,
                                                               x_hat_plus_k_minus_1,
                                                               u,
                                                               w,
                                                               model_params_hat).T
        # Size should be (1, n_states)

        # Predictor (Time update of state estimate and estimation-error covariance)
        P_minus_k = F_k_minus_1 @ P_plus_k_minus_1 @ F_k_minus_1.T + \
            L_k_minus_1 @ control_system_params.Q_k @ L_k_minus_1.T
        _, x_hat_minus_k = reRK4(ode_to_integrate,
                                          (t, t+simulation_params.STEP_SIZE),
                                          x_hat_plus_k_minus_1,
                                          N=simulation_params.n_integration,
                                          args=((u, w, model_params_hat),))
        # Update using system dynamics from last estimate
        x_hat_minus_k = x_hat_minus_k[-1, :].squeeze() # Get the last value and make 1D array

        # Linearise noise dynamics
        H_k = jax.jacfwd(Plant.sensor_one_species, 1)(t, x_hat_minus_k, u, v) # Should be (1, n_states)
        M_k = jax.jacfwd(Plant.sensor_one_species, 3)(t, x_hat_minus_k, u, v) # Should be (1, 1)
        # Corrector
        innovation_k = z - Plant.sensor_one_species(t, x_hat_minus_k, u, v)

        # Innovation covariance
        S = H_k @ P_minus_k @ H_k.T + M_k @ control_system_params.R_k @ M_k.T
        # Calculate NIS using Cholesky decomposition (more stable)
        # S = L @ L.T
        L = jnp.linalg.cholesky(S)
        # Solve L @ y = innovation_k
        y = jnp.linalg.solve(L, innovation_k)
        # NIS = innovation_k.T @ inv(S) @ innovation_k = y.T @ y
        nis = jnp.sum(y**2)

        # Kalman gain calculation without explicitly computing inverse
        # Instead of K_k = P_minus_k @ H_k.T @ inv(S)
        # 1. Solve L @ L.T @ x = H_k @ P_minus_k.T for x
        # 2. Then K_k = P_minus_k @ H_k.T @ inv(S) = x
        K_part1 = jnp.linalg.solve(L, H_k @ P_minus_k.T)
        K_k = jnp.linalg.solve(L.T, K_part1).T
    
        x_hat_plus = x_hat_minus_k + K_k @ innovation_k

        # Enforce non-negativity - NOTE DEVIATION FROM CLASSICAL EKF
        x_hat_plus = jnp.maximum(0., x_hat_plus)

        # P_k_plus = (jnp.eye(jnp.shape(P_minus_k)[0]) - K_k@H_k) @ P_minus_k # NUMERICALLY UNSTABLE
        I_KH = (jnp.eye(jnp.shape(P_minus_k)[0]) - K_k @ H_k)
        P_k_plus = I_KH @ P_minus_k @ I_KH.T + K_k @ control_system_params.R_k @ K_k.T # Joseph form

        ekf_state = (x_hat_plus, P_k_plus, nis)
        return ekf_state

    def _compute_ukf_weights(n, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        Returns (Wm, Wc, lambda_, c) where:
        - Wm: weights for mean (shape (2n+1,))
        - Wc: weights for covariance
        - lambda_: scaling
        - c: n + lambda_
        """
        lambda_ = alpha**2 * (n + kappa) - n
        c = n + lambda_
        Wm0 = lambda_ / c
        Wc0 = Wm0 + (1 - alpha**2 + beta)
        W = 1.0 / (2.0 * c)
        Wm = jnp.concatenate([jnp.array([Wm0]), jnp.full(2 * n, W)])
        Wc = jnp.concatenate([jnp.array([Wc0]), jnp.full(2 * n, W)])
        return Wm, Wc, lambda_, c

    def _sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        Generate 2n+1 sigma points for state mean x and covariance P.
        Return shape: (2n+1, n)
        """
        n = x.shape[0]
        Wm, Wc, lambda_, c = Observer._compute_ukf_weights(n, alpha, beta, kappa)
        # scaled cholesky
        jitter = 1e-6
        scaled_P = (c) * P
        L = jnp.linalg.cholesky(scaled_P + jitter * jnp.eye(n))
        # first sigma is mean
        SP0 = x
        # other sigma points: x +/- columns of L
        L_cols = L.T  # shape (n, n) columns are rows of L.T
        plus = x + L_cols
        minus = x - L_cols
        sigma = jnp.vstack([SP0, plus, minus])
        return sigma, Wm, Wc

    def _ukf_predict(x, P, f, u, Q, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        UKF prediction step.
        - x: state mean (n,)
        - P: state covariance (n,n)
        - f: dynamics function f(x) -> x_pred (n,)
        - Q: process noise covariance (n,n)
        returns (x_pred, P_pred, transformed_sigma_points)
        """
        n = x.shape[0]
        sigma, Wm, Wc = Observer._sigma_points(x, P, alpha, beta, kappa)

        # propagate sigma points through dynamics
        f_vmap = jax.vmap(f, in_axes=(0, None))  # Vectorize over sigma points, keep u fixed
        Xp = f_vmap(sigma, u)  # Pass both sigma and u

        x_pred = jnp.sum(Wm[:, None] * Xp, axis=0)
        dx = Xp - x_pred[None, :]
        P_pred = jnp.einsum('i,ij,ik->jk', Wc, dx, dx) + Q
        return x_pred, P_pred, Xp, Wm, Wc

    def _ukf_correct(x_pred, P_pred, Xp, z, h, u, R, Wm, Wc):
        """
        UKF update step with measurement z.
        - x_pred: predicted mean (n,)
        - P_pred: predicted covariance (n,n)
        - Xp: propagated sigma points (2n+1, n)
        - z: measurement vector (m,)
        - h: measurement function h(x) -> z (m,)
        - R: measurement noise covariance (m,m)
        - Wm, Wc: weights from sigma generation (both length 2n+1)
        returns (x_upd, P_upd)
        """
        # compute predicted measurements for each sigma
        h_vmap = jax.vmap(h, in_axes=(0, None))  # Vectorize over sigma points, keep u fixed
        Zp = h_vmap(Xp, u)  # Pass both Xp and u
        z_pred = jnp.sum(Wm[:, None] * Zp, axis=0)

        # innovations
        dz = Zp - z_pred[None, :]

        # measurement covariance Pzz
        Pzz = jnp.einsum('i,ij,ik->jk', Wc, dz, dz) + R

        # cross-covariance Pxz
        dx = Xp - x_pred[None, :]
        Pxz = jnp.einsum('i,ij,ik->jk', Wc, dx, dz)

        # Kalman gain
        # Solve Pzz.T * K.T = Pxz.T  for K.T  (better numeric stability than explicit inverse)
        Kt = jnp.linalg.solve(Pzz.T, Pxz.T)
        K = Kt.T

        # update
        y = z - z_pred
        x_upd = x_pred + jnp.matmul(K, y).squeeze() # shape (n,) instead of (n,1)
        P_upd = P_pred - K @ Pzz @ K.T
        
        # ensure symmetry
        P_upd = 0.5 * (P_upd + P_upd.T)

        L = jnp.linalg.cholesky(Pzz)
        # Solve L @ y = innovation_k
        innovation_normalised = jnp.linalg.solve(L, y)
        # NIS = innovation_k.T @ inv(S) @ innovation_k = y.T @ y
        nis = jnp.sum(innovation_normalised**2)

        return x_upd, P_upd, nis

    def ukf_update(t,
                   u,
                   z,
                   ukf_state,
                   model_params_hat,
                   control_system_params,
                   simulation_params):
        """Unscented Kalman Filter update step."""
        def ode_to_integrate(t, y, args) -> jnp.array:
            """
            Wrapper function for the system dynamics
            """
            u, w, model_params = args
            return Plant.ode_model_one_species(t, y, u, w, model_params)
        def f_for_ukf(x, u):
            _, traj = reRK4(
                ode_to_integrate,
                [0.0, simulation_params.STEP_SIZE],
                x,
                N=simulation_params.n_integration,
                args=((u, jnp.zeros_like(x), model_params_hat),)
            )
            return traj[-1, :]
        def h_for_ukf(x, u):
            return jnp.array([Plant.sensor_one_species(0., x, u, 0.)])
        
        x_hat_ukf, P_hat_ukf = ukf_state

        # 1. Predictor
        x_pred, P_pred, Xp, Wm, Wc = Observer._ukf_predict(x_hat_ukf, P_hat_ukf, f_for_ukf,
                                                 u,  control_system_params.Q_k,
                                                 control_system_params.ukf_params[0],
                                                 control_system_params.ukf_params[1],
                                                 control_system_params.ukf_params[2])
        # 2. Corrector
        x_hat_ukf, P_hat_ukf, nis = Observer._ukf_correct(x_pred, P_pred, Xp, z, h_for_ukf, u, control_system_params.R_k, Wm, Wc)

        x_hat_ukf = jnp.nan_to_num(x_hat_ukf, nan=0.0)
        P_hat_ukf = jnp.nan_to_num(P_hat_ukf, nan=1.0)
        nis = jnp.nan_to_num(nis, nan=0.0)

        x_hat_ukf = jnp.maximum(0., x_pred) # enforce non-negativity
        # Ensure P is positive semidefinite, if an issue
        # eigvals, eigvecs = jnp.linalg.eigh(P_hat_ukf)
        # eigvals = jnp.maximum(eigvals, 1e-6)  # Enforce non-negative eigenvalues
        # P_hat_ukf = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        return (x_hat_ukf, P_hat_ukf, nis)

    def _clip_population(params_opt: jnp.ndarray,
                        params_true: jnp.ndarray,
                        mask_tpl: tuple,
                        minval: jnp.ndarray,
                        maxval: jnp.ndarray
                        ):
        """
        Constructs a full parameter array by replacing entries of params_true with params_opt
        according to mask_tpl, then clips both the full parameter array and the optimized parameters.

        Args:
            params_opt: jnp.ndarray of shape (B, K), the batch of parameters to be optimized.
            params_true: jnp.ndarray of shape (M,), the nominal values for the full set of parameters.
            mask_tpl: tuple of booleans of length M. For each index:
                    - True: the value should be optimized (and replaced by a value from params_opt).
                    - False: the nominal value from params_true is kept.
            minval: jnp.ndarray of shape (M,) with the lower bounds for each parameter.
            maxval: jnp.ndarray of shape (M,) with the upper bounds for each parameter.

        Returns:
            full_params: jnp.ndarray of shape (B, M) with the parameters inserted and then clipped.
            params_opt_clipped: jnp.ndarray of shape (B, K) with the optimized parameters clipped.
        """
        B = params_opt.shape[0]
        # Convert mask_tpl (a tuple of booleans) into a tuple of concrete indices where the mask is True.
        opt_indices = tuple(i for i, flag in enumerate(mask_tpl) if flag)
        # Expand params_true into a (B, M) array where B is the batch size.
        full_params = jnp.tile(params_true, (B, 1))
        # Replace the entries in full_params corresponding to optimisable parameters with params_opt,
        # using the concrete indices.
        full_params = full_params.at[:, opt_indices].set(params_opt)
        # Clip full_params with the global bounds.
        full_params = jnp.clip(full_params, minval, maxval)
        # Clip the optimisable parameters individually
        params_opt_clipped = jnp.clip(params_opt,
                                      minval[jnp.array(opt_indices)],
                                      maxval[jnp.array(opt_indices)])
        return full_params, params_opt_clipped

    def _loss_function(model_params_hat,
                      x_hat,
                      u_history,
                      z_history,
                      n_integration,
                      reestimate_window_size,
                      STEP_SIZE):
        """MSE loss function for MHE"""
        # Define a function for scan to apply at each step
        def scan_step(carry, input_idx):
            x_hat_f, _ = carry
            current_u = jnp.take(u_history, input_idx)
            _, x_hat_f = reRK4(Plant.ode_model_to_integrate_one_species,
                                    (0., STEP_SIZE), # time is arbitrary here
                                    x_hat_f,
                                    N=n_integration,
                                    args=((current_u, jnp.array([0.]), model_params_hat),))
            x_hat_f = x_hat_f[-1, :] # Get the last value
            y_estimated = Plant.sensor_one_species(0.,
                                        x_hat_f,
                                        current_u,
                                        0.) # time is arbitrary here

            # Store the estimated measurement
            current_z = jnp.take(z_history, input_idx)
            abs_error = jnp.abs(y_estimated - current_z)

            return (x_hat_f, y_estimated), abs_error

        # Initialize with starting state
        init_carry = (x_hat, 0.)
        # Create array of indices for the scan
        # NOTE u_history[0] is the most recent value and u_history[-1] is the oldest
        # Therefore we want to scan from the oldest to the most recent
        indices = jnp.arange(reestimate_window_size-2, -1, -1)

        # Scan through the window
        (x_hat_f, _), abs_errors = jax.lax.scan(
            scan_step, init_carry, indices)

        # Calculate MAE
        error = jnp.mean(abs_errors)

        return error, x_hat_f

    def _mhe_es_masked(control_system_state,
                      control_system_params,
                      simulation_params,
                      key):
        """MHE using Evolution Strategies"""
        
        loss_function_partial = partial(Observer._loss_function,
                                                x_hat=control_system_state.x_hat_history[-1],
                                                u_history=control_system_state.u_history,
                                                z_history=control_system_state.z_history,
                                                n_integration=simulation_params.n_integration,
                                                reestimate_window_size=simulation_params.reestimate_window_size,
                                                STEP_SIZE=simulation_params.STEP_SIZE
                                                )

        def es_step (state_input, tmp):
            """
            Helper es step to lax.scan through.
            Taken from evosax documentation
            """
            key, es_state = state_input
            key, key_ask, key_tell = jax.random.split(key, 3)
            population, es_state = es.ask(key_ask,
                                        es_state,
                                        es_params)
            params_full, population_clipped = Observer._clip_population(population,
                                                                        control_system_params.opt_constraints.replacement_params,
                                                                        simulation_params.optimisation_params.mask,
                                                                        control_system_params.opt_constraints.minval,
                                                                        control_system_params.opt_constraints.maxval)
            vectorized_loss_function = jax.vmap(loss_function_partial)
            fitness, x_hat_f = vectorized_loss_function(params_full)
            es_state, metrics = es.tell(key_tell, population_clipped, fitness, es_state, es_params)
            # Return the best fitness value from this generation
            # (Minimising so want lowest fitness)
            best_idx = jnp.argmin(fitness)
            best_fitness = fitness[best_idx]
            best_params = params_full[best_idx]
            best_x_hat_f = x_hat_f[best_idx]

            return (key, es_state), (best_fitness, best_params, best_x_hat_f)

        key, key_init, key_pop = jax.random.split(key, 3)
        # Convert the boolean mask to indices
        mask_indices = simulation_params.optimisation_params.mask_indices
        # Extract the parameters to optimize
        params_opt = control_system_state.model_params_hat[jnp.array(mask_indices)]
        # Create initialisation that starts over entire range
        mean = (control_system_params.opt_constraints.minval + control_system_params.opt_constraints.maxval) / 2
        std = (control_system_params.opt_constraints.maxval - control_system_params.opt_constraints.minval) / 6 # 3 standard deviations to cover the range
        mean_masked = mean[jnp.array(mask_indices)]
        std_masked = std[jnp.array(mask_indices)]
        # minval_masked = control_system_params.opt_constraints.minval[jnp.array(mask_indices)]
        # maxval_masked = control_system_params.opt_constraints.maxval[jnp.array(mask_indices)]

        # Intialise ES
        es = ES(population_size=simulation_params.optimisation_params.population_size,
                solution=params_opt)
        es_params = es.default_params

        # For Population based ES:
        # key, key_init, key_pop = jax.random.split(key, 3)
        # population = jax.random.uniform(key_pop,
        #                          shape=(hyperparameters['population_size'], mean_masked.shape[0]),
        #                          minval=mean_masked - 2 * std_masked,
        #                          maxval=mean_masked + 2 * std_masked)
        # fitness, _ = jax.vmap(loss_function_partial)(population)
        # es_state = es.init(key_init, population, fitness, es_params)

        # For Distribution-based ES:
        key, key_init = jax.random.split(key, 2)
        es_state = es.init(key_init, params_opt, es_params) # key, solution, params
        es_state = es_state.replace(mean=mean_masked, std=std_masked)

        num_generations = simulation_params.optimisation_params.n_generations

        # Calculate loss on current parameters
        current_fitness, x_hat_f_current = loss_function_partial(control_system_state.model_params_hat)

        _, (fitness_history, best_params_history, x_hat_f_history) = jax.lax.scan(
                                                es_step, # function
                                                (key, es_state), # initial carry
                                                jnp.zeros(num_generations) # input
                                            )

        # Get final best parameters (from the last generation)
        best_idx = jnp.argmin(fitness_history)
        best_params_opt = best_params_history[best_idx]

        condition = fitness_history[best_idx] < current_fitness
        model_params_hat, training_loss, x_hat_f = jax.lax.cond(condition,
                                        lambda _: (best_params_opt, fitness_history[best_idx], x_hat_f_history[best_idx]),
                                        lambda _: (control_system_state.model_params_hat, current_fitness, x_hat_f_current),
                                        None)

        mhe_activation = control_system_state.mhe_activation + 1 # Increment activation

        # Repackage in ControlSystemState
        control_system_state = control_system_state._replace(x_hat=x_hat_f,
                                                             nis_history=jnp.zeros_like(control_system_state.nis_history), # resets NIS to stop multiple triggers
                                                            model_params_hat=model_params_hat,
                                                            mhe_activation=mhe_activation)

        return control_system_state
