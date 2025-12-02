"""Compare observers on open loop simulation"""
import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import vmap
from math import ceil
from typing import NamedTuple
from functools import partial
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True) # Detect NaNs
# jax.config.update("jax_disable_jit", True) # for debugging

class SimulationTools():
    def _reRK4Step(func, t, y, h, args):
        """One step of the Runge-Kutta 4th order method"""
        k1 = h*func(t, y, *args)
        k2 = h*func(t + 0.5*h, y + 0.5*k1, *args)
        k3 = h*func(t + 0.5*h, y + 0.5*k2, *args)
        k4 = h*func(t + h, y + k3, *args)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        # return jnp.maximum(y, 0.)  # Ensure non-negative values
        return y

    def reRK4(func, t_span: list[float], y0, N=100, args=()):
        """
        Runge-Kutta 4th order method for solving ODEs, RK4
        With rectification
        Constant step size

        Both SciPy solve_ivp and MATLAB ode45 use the Dormand-Prince 
        4th order Runge Kutta method, which is slightly different, 
        being a 6-term, 5th order method with adaptive step size control

        Args:
        func: function to solve, with arguments (t, y, *args)
        t_span: time span, (t0, tf)
        y0: initial conditions
        N: number of steps - N+1 equally spaced in (t0, tf)
        args: additional arguments to pass to func

        Returns:
        t: time array
        y: solution array
        """
        t0, tf = t_span
        h = (tf - t0)/N
        t = jnp.linspace(t0, tf, N+1)
        y = jnp.zeros((N+1, len(y0)))
        y = y.at[0, :].set(y0)

        def body_fun(i, y):
            return y.at[i].set(SimulationTools._reRK4Step(func, t[i - 1], y[i - 1, :], h, args))

        y = jax.lax.fori_loop(1, N + 1, body_fun, y)
        return t, y

class Plant():
    def simplified_levin_lenski_one_species(t, y, u, w_noise, model_params) -> jnp.array:
        """
        2 Bacterial strains, 2 phage strains
        No dependence on Volume

        Inputs:
        t: time
        y: state variables, S, I, P
        args: parameters (model_params, control_signal)
        model_params: 
        model_params[0]: burst size, B1
        model_params[1]: adsorption rate, delta1-1 (Phage 1 to Bacteria 1)
        model_params[2]: time delay, tau1
        model_params[3]: carrying capacity, K (monod constant)
        model_params[4]: concentration of glucose in input media, c
        model_params[5]: maximum growth rate of bacteria 1, mu_max1
        model_params[6]: removal rate of nutrients, e
        model_params[7]: monodOn: bool
        model_params[8]: burst size, B2
        model_params[9]: time delay, tau2
        model_params[10]: adsorption rate, delta2-1 (Phage 2 to Bacteria 1)
        model_params[11]: adsorption rate, delta2-2 (Phage 2 to Bacteria 2)
        model_params[12]: adsorption rate, delta1-2 (Phage 1 to Bacteria 2)
        model_params[13]: maximum growth rate of bacteria 2, mu_max2
        control_signal: turnover rate, omega
        
        Outputs:
        dy: time derivatives of state variables 
        """
        M = 5
        control_signal = u
        # Ensure no negative values in y
        y = jnp.maximum(y, 0.)
        delta_clipped = jnp.clip(jnp.exp(model_params[1]), 1e-12, 1e-6) # NOTE added clipping to avoid overflow
        # delta_clipped = jnp.exp(model_params[1]) # OPTION TO REMOVE DELTA CLIPPING

        S1, I11, I12, I13, I14, I15, P1  = y

        dS1 = model_params[5] * S1 - delta_clipped * S1 * P1 \
                                    - control_signal * S1 + w_noise[0]
        dI11 = delta_clipped * S1 * P1   - M / model_params[2] * I11 \
            - control_signal * I11
        dI12 = M / model_params[2] * (I11 - I12) - control_signal * I12
        dI13 = M / model_params[2] * (I12 - I13) - control_signal * I13
        dI14 = M / model_params[2] * (I13 - I14) - control_signal * I14
        dI15 = M / model_params[2] * (I14 - I15) - control_signal * I15
        dP1 = model_params[0] * M / model_params[2] * I15 \
            - delta_clipped * S1 * P1 - control_signal * P1

        dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1])
        return dy
    
    def sllos_dukf(t, y, u, w_noise, model_params) -> jnp.array:
        """
        Simplified Levin-Lenski Dynamics for DUKF Observer
        1 Bacterial strain, 1 phage strains

        Inputs:
        t: time
        y: state variables, S, I, P
        args: parameters (model_params, control_signal)
        model_params: 
        model_params[0]: burst size, B1
        model_params[1]: adsorption rate, delta1-1 (Phage 1 to Bacteria 1)
        model_params[2]: time delay, tau1
        model_params[3]: maximum growth rate of bacteria 1, mu_max1 NOTE change
        control_signal: turnover rate, omega
        
        Outputs:
        dy: time derivatives of state variables 
        """
        M = 5
        control_signal = u
        # Ensure no negative values in y
        y = jnp.maximum(y, 0.)
        delta_clipped = jnp.clip(jnp.exp(model_params[1]), 1e-12, 1e-6)
        # delta_clipped = jnp.exp(model_params[1]) # OPTION TO REMOVE DELTA CLIPPING

        S1, I11, I12, I13, I14, I15, P1  = y

        dS1 = model_params[3] * S1 - delta_clipped * S1 * P1 \
                                    - control_signal * S1 + w_noise[0]
        dI11 = delta_clipped * S1 * P1   - M / model_params[2] * I11 \
            - control_signal * I11
        dI12 = M / model_params[2] * (I11 - I12) - control_signal * I12
        dI13 = M / model_params[2] * (I12 - I13) - control_signal * I13
        dI14 = M / model_params[2] * (I13 - I14) - control_signal * I14
        dI15 = M / model_params[2] * (I14 - I15) - control_signal * I15
        dP1 = model_params[0] * M / model_params[2] * I15 \
            - delta_clipped * S1 * P1 - control_signal * P1

        dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1])
        return dy
    
    def theta_transition(x_state, theta, u, w_noise) -> jnp.array:
        """Parameter transition model"""
        dtheta = jnp.zeros_like(theta) + w_noise
        return dtheta
    
    def sllos_jukf(t, x_aug, u, w_noise) -> jnp.array:
        """
        Simplified Levin-Lenski Dynamics for JUKF Observer
        1 Bacterial strain, 1 phage strains

        Inputs:
        x_aug: augmented state [y, model_params]
            y: state variables, S, I, P
            model_params:
            model_params[0]: burst size, B1
            model_params[1]: adsorption rate, delta1-1 (Phage 1 to Bacteria 1)
            model_params[2]: time delay, tau1
            model_params[3]: maximum growth rate of bacteria 1, mu_max1 NOTE change
        u : control input
        w_noise: process noise
        Outputs:
        dx_aug : time derivatives of augmented state variable
        """
        dy = Plant.sllos_dukf(0., x_aug[:7], u, w_noise[:7], x_aug[7:])
        dtheta = Plant.theta_transition(x_aug[:7], x_aug[7:], u, w_noise[7:])
        dx_aug = jnp.concatenate([dy, dtheta])
        return dx_aug

    def sensor_model9_one_species(t, x, u, v, args=()):
        """
        Complies with the observer function signature
        h: observation function, h(t, x, u, v, args)

        Inputs:
        t: time
        x: state variables: S1, I11, I12, I13, I14, I15, P1  NOTE CHANGE
        u: control input
        v: float NOTE change from model7
            observation noise
        """
        observation_true = x[0] + x[1] + x[2] + x[3] + x[4] + x[5]
        observation_made = observation_true + v
        # Return as 1D array with single element
        return jnp.array([observation_made]) # shape (1,) NOTE change from model8.py and model9.py

class DUKFObserver():
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

    def _ukf_predict(x, P, theta, f, u, Q, alpha=1e-3, beta=2.0, kappa=0.0):
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
        f_vmap = vmap(f, in_axes=(0, None, None))  # Vectorize over sigma points, keep u fixed
        Xp = f_vmap(sigma, theta, u)  # Pass both sigma and u

        x_pred = jnp.sum(Wm[:, None] * Xp, axis=0)
        dx = Xp - x_pred[None, :]
        P_pred = jnp.einsum('i,ij,ik->jk', Wc, dx, dx) + Q
        return x_pred, P_pred, Xp, Wm, Wc

    def _ukf_correct(x_pred, P_pred, theta, Xp, z, h, u, R, Wm, Wc):
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
        h_vmap = vmap(h, in_axes=(0, None, None))  # Vectorize over sigma points, keep u fixed
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
    
    def _ukf_param_correct(x_pred, P_pred, dukf_state,
                           Xp, z, h, R, Wm, Wc):
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
        h_vmap = jax.vmap(h, in_axes=(0, None))  # Vectorize over sigma points
        Zp = h_vmap(Xp, dukf_state)
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

    def _ukf_state_update(dukf_state, f_state, h_state, u, z, dukf_params):
        """
        UKF update step for state only, keeping parameters fixed.
            - dukf_state: DualUKFState named tuple
            - f: dynamics function f(x) -> x_pred (n,)
            - h: measurement function h(x) -> z (m,)
            - u: control input
            - z: measurement vector (m,)
            - dukf_params: DualUKFParams named tuple
        returns updated dukf_state with new x_hat and P_hat
        """
        # UKF update for state
        x_pred, P_pred, Xp, Wm, Wc = DUKFObserver._ukf_predict(dukf_state.x_hat, dukf_state.P_hat,
                                                    dukf_state.theta_hat, f_state,
                                                    u, dukf_params.Q_state, dukf_params.alpha,
                                                    dukf_params.beta, dukf_params.kappa)
        # UKF update with measurement
        x_hat_ukf, P_hat_ukf = DUKFObserver._ukf_correct(x_pred, P_pred, dukf_state.theta_hat, Xp,
                                                         z, h_state, u, dukf_params.R_state, Wm, Wc)

        # Repackage
        dukf_state = dukf_state._replace(x_hat=x_hat_ukf, P_hat=P_hat_ukf)

        return dukf_state

    def _ukf_param_update(f_param, h_param, u, z, dukf_params, dukf_state):
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
        theta_pred, P_theta_pred, Xp, Wm, Wc = DUKFObserver._ukf_predict(dukf_state.theta_hat,
                                                dukf_state.P_theta_hat, dukf_state.x_hat, f_param,
                                                u, dukf_params.Q_param, dukf_params.alpha,
                                                dukf_params.beta, dukf_params.kappa)
        # UKF update with measurement
        theta_hat_ukf, P_theta_hat_ukf = DUKFObserver._ukf_param_correct(theta_pred, P_theta_pred, dukf_state,
                                                         Xp, z, h_param, dukf_params.R_param, Wm, Wc)

        theta_hat_ukf = jnp.clip(theta_hat_ukf, # OPTION TO CLIP PARAMETERS
                         jnp.array([50.0, jnp.log(1e-12), 0.1, 0.1]),
                         jnp.array([200.0, jnp.log(1e-6), 2.0, 3.0]))

        dukf_state = dukf_state._replace(theta_hat=theta_hat_ukf,
            P_theta_hat=P_theta_hat_ukf)
        return dukf_state

    def dukf_update(dukf_state, z, f_state, h_state, f_param, h_param, u, dukf_params):
        """
        Dual UKF update step for joint state and parameter estimation.
            - dukf_state: DualUKFState named tuple
            - z: measurement vector (m,)
            - h: measurement function h(x) -> z (m,)
            - u: control input
            - dukf_params: DualUKFParams named tuple
        """
        # UKF update with measurement
        dukf_state = DUKFObserver._ukf_state_update(dukf_state, f_state, h_state, u, z, dukf_params)

        # If inner_counter is 0, perform full UKF update
        parameter_update_partial = partial(DUKFObserver._ukf_param_update,
                                        f_param, h_param, u, z, dukf_params)
        condition = dukf_state.inner_counter == 0
        dukf_state = jax.lax.cond(condition,
                        parameter_update_partial,
                        lambda x: x,
                        dukf_state)

        # Update buffers NOTE careful
        # Do I update x_history and u_history after state update or parameter update?
        # NOTE this could be different to model9,py
        new_x_history = jnp.roll(dukf_state.x_hat_history, 1, axis=0)
        new_x_history = new_x_history.at[0].set(dukf_state.x_hat)
        dukf_state = dukf_state._replace(x_hat_history=new_x_history)
        new_u_history = jnp.roll(dukf_state.u_history, 1, axis=0)
        new_u_history = new_u_history.at[0].set(u)
        dukf_state = dukf_state._replace(u_history=new_u_history)

        # Advance inner counter and wrap around
        inner_counter_new = (dukf_state.inner_counter + 1) % dukf_params.inner_loops
        dukf_state = dukf_state._replace(inner_counter=inner_counter_new)

        return dukf_state

class DEKFObserver():
    def _ekf_state_update(u,
                   z,
                   f_state_fn,
                   h_state_fn, 
                   dekf_params,
                   dekf_state):
        """
        Discrete-time Extended Kalman Filter (EKF) for JAX
        Stateless implementation

        Uses notation of Optimal State Estimation by Dan Simon        
        """
        x_hat_plus_k_minus_1 = dekf_state.x_hat
        P_plus_k_minus_1 = dekf_state.P_hat
        # Linearise the system (state transition function, not ode model!) at current point
        w  = jnp.array([0.])
        v  = jnp.array([0.])
        F_k_minus_1 = jax.jacfwd(f_state_fn, 0)(x_hat_plus_k_minus_1, dekf_state.theta_hat, u, w) # Size should be (n, n)
        L_k_minus_1 = jax.jacfwd(f_state_fn, 3)(x_hat_plus_k_minus_1, dekf_state.theta_hat, u, w).T # Size should be (1, n)

        # Predictor (Time update of state estimate and estimation-error covariance)
        P_minus_k = F_k_minus_1 @ P_plus_k_minus_1 @ F_k_minus_1.T + \
            L_k_minus_1 @ dekf_params.Q_state @ L_k_minus_1.T
        x_hat_minus_k = f_state_fn(x_hat_plus_k_minus_1, dekf_state.theta_hat, u, w)
        # Linearise noise dynamics
        H_k = jax.jacfwd(h_state_fn, 0)(x_hat_minus_k, dekf_state.theta_hat, u, v) # Should be (1, n)
        M_k = jax.jacfwd(h_state_fn, 3)(x_hat_minus_k, dekf_state.theta_hat, u, v) # Should be (1, 1)
        # Corrector
        innovation_k = z - h_state_fn(x_hat_minus_k, dekf_state.theta_hat, u, v)

        # Innovation covariance
        S = H_k @ P_minus_k @ H_k.T + M_k @ dekf_params.R_state @ M_k.T
        # Calculate NIS using Cholesky decomposition (more stable)
        # S = L @ L.T
        L = jnp.linalg.cholesky(S)
        # Solve L @ y = innovation_k
        y = jnp.linalg.solve(L, innovation_k)
        # NIS = innovation_k.T @ inv(S) @ innovation_k = y.T @ y
        nis = jnp.sum(y**2)

        K_part1 = jnp.linalg.solve(L, H_k @ P_minus_k.T)
        K_k = jnp.linalg.solve(L.T, K_part1).T
        
        x_hat_plus = x_hat_minus_k + jnp.matmul(K_k, innovation_k).squeeze()

        # Joseph form for numerical stability:
        I_KH = (jnp.eye(jnp.shape(P_minus_k)[0]) - K_k @ H_k)
        P_k_plus = I_KH @ P_minus_k @ I_KH.T + K_k @ dekf_params.R_state @ K_k.T

        dekf_state = dekf_state._replace(x_hat=x_hat_plus,
                                   P_hat=P_k_plus)
        return dekf_state
    
    def _ekf_param_update(u,
                   z,
                   f_param_fn,
                   h_param_fn, 
                   dekf_params,
                   dekf_state):
        """
        Discrete-time Extended Kalman Filter (EKF) for JAX
        Stateless implementation

        Uses notation of Optimal State Estimation by Dan Simon        
        """
        theta_hat_plus_k_minus_1 = dekf_state.theta_hat # Use parameter estimate as state
        P_theta_plus_k_minus_1 = dekf_state.P_theta_hat
        # Linearise the system (state transition function, not ode model!) at current point
        w  = jnp.array([0.])
        v  = jnp.array([0.])
        F_k_minus_1 = jax.jacfwd(f_param_fn, 1)(dekf_state.x_hat, theta_hat_plus_k_minus_1, u, w) # Size should be (n, n)
        L_k_minus_1 = jax.jacfwd(f_param_fn, 3)(dekf_state.x_hat, theta_hat_plus_k_minus_1, u, w).T # Size should be (1, n)

        # Predictor (Time update of state estimate and estimation-error covariance)
        P_theta_minus_k = F_k_minus_1 @ P_theta_plus_k_minus_1 @ F_k_minus_1.T + \
            L_k_minus_1 @ dekf_params.Q_param @ L_k_minus_1.T
        theta_hat_minus_k = f_param_fn(dekf_state.x_hat, theta_hat_plus_k_minus_1, u, w)
        # Linearise noise dynamics
        H_k = jax.jacfwd(h_param_fn, 1)(dekf_state.x_hat, theta_hat_minus_k, u, v) # Should be (1, n)
        M_k = jax.jacfwd(h_param_fn, 3)(dekf_state.x_hat, theta_hat_minus_k, u, v) # Should be (1, 1)
        # Corrector
        innovation_k = z - h_param_fn(dekf_state.x_hat, theta_hat_minus_k, u, v)

        # Innovation covariance
        S = H_k @ P_theta_minus_k @ H_k.T + M_k @ dekf_params.R_param @ M_k.T
        # Calculate NIS using Cholesky decomposition (more stable)
        # S = L @ L.T
        L = jnp.linalg.cholesky(S)
        # Solve L @ y = innovation_k
        y = jnp.linalg.solve(L, innovation_k)
        # NIS = innovation_k.T @ inv(S) @ innovation_k = y.T @ y
        nis = jnp.sum(y**2) 

        K_part1 = jnp.linalg.solve(L, H_k @ P_theta_minus_k.T)
        K_k = jnp.linalg.solve(L.T, K_part1).T

        theta_hat_plus = theta_hat_minus_k + jnp.matmul(K_k, innovation_k).squeeze()

        # Joseph form for numerical stability:
        I_KH = (jnp.eye(jnp.shape(P_theta_minus_k)[0]) - K_k @ H_k)
        P_theta_k_plus = I_KH @ P_theta_minus_k @ I_KH.T + K_k @ dekf_params.R_param @ K_k.T


        theta_hat_plus = jnp.clip(theta_hat_plus, # OPTION TO CLIP PARAMETERS
                         jnp.array([50.0, jnp.log(1e-12), 0.1, 0.1]), 
                         jnp.array([200.0, jnp.log(1e-6), 2.0, 3.0]))
        dekf_state = dekf_state._replace(theta_hat=theta_hat_plus,
                                   P_theta_hat=P_theta_k_plus)
        
        return dekf_state

    def dekf_update(dekf_state, z, f_state, h_state, f_param, h_param, u, dekf_params):
        """
        Dual UKF update step for joint state and parameter estimation.
            - dukf_state: DualUKFState named tuple
            - z: measurement vector (m,)
            - h: measurement function h(x) -> z (m,)
            - u: control input
            - dukf_params: DualUKFParams named tuple
        """
        # UKF update with measurement
        dekf_state = DEKFObserver._ekf_state_update(u, z, f_state, h_state, dekf_params, dekf_state) 
        # Enforce non-negativity on state estimate
        dekf_state = dekf_state._replace(x_hat=jnp.maximum(dekf_state.x_hat, 0.))

        # If inner_counter is 0, perform full UKF update
        parameter_update_partial = partial(DEKFObserver._ekf_param_update,
                                        u, z, f_param, h_param, dekf_params)
        condition = dekf_state.inner_counter == 0
        dekf_state = jax.lax.cond(condition,
                        parameter_update_partial,
                        lambda x: x,
                        dekf_state)

        # Advance inner counter and wrap around
        inner_counter_new = (dekf_state.inner_counter + 1) % dekf_params.inner_loops
        dekf_state = dekf_state._replace(inner_counter=inner_counter_new)
        return dekf_state

class JUKFObserver():
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
        Wm, Wc, lambda_, c = JUKFObserver._compute_ukf_weights(n, alpha, beta, kappa)
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
        sigma, Wm, Wc = JUKFObserver._sigma_points(x, P, alpha, beta, kappa)

        # propagate sigma points through dynamics
        f_vmap = vmap(f, in_axes=(0, None))  # Vectorize over sigma points, keep u fixed
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
        h_vmap = vmap(h, in_axes=(0, None))  # Vectorize over sigma points, keep u fixed
        Zp = h_vmap(Xp, u)  # Pass both Xp and u NOTE changed to only pass Xp and u
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

    def jukf_update(jukf_state, z, f, h, u, jukf_params):
        """
        Joint UKF update step for joint state and parameter estimation.
            - jukf_state: JointUKFState named tuple
            - z: measurement vector (m,)
            - h: measurement function h(x) -> z (m,)
            - u: control input
            - jukf_params: JointUKFParams named tuple
        returns updated jukf_state with new x_hat and P_hat
        """
        # UKF update for augmented state
        x_pred, P_pred, Xp, Wm, Wc = JUKFObserver._ukf_predict(jukf_state.x_hat, jukf_state.P_hat,
                                                    f,
                                                    u, jukf_params.Q_joint, jukf_params.alpha,
                                                    jukf_params.beta, jukf_params.kappa)
        # UKF update with measurement
        x_hat_ukf, P_hat_ukf = JUKFObserver._ukf_correct(x_pred, P_pred, Xp,
                                                         z, h, u, jukf_params.R_joint, Wm, Wc)
        
        # Enforce non-negativity on state estimate and bounds on parameters
        parameters = jnp.clip(x_hat_ukf[7:], # OPTION TO CLIP PARAMETERS
                         jnp.array([50.0, jnp.log(1e-12), 0.1, 0.1]), 
                         jnp.array([200.0, jnp.log(1e-6), 2.0, 3.0]))
        parameters = x_hat_ukf[7:]
        x_state = jnp.maximum(0., x_hat_ukf[:7])
        x_hat_ukf = jnp.concatenate([x_state, parameters])

        # Repackage
        jukf_state = JointUKFState(
            x_hat=x_hat_ukf,
            P_hat=P_hat_ukf
        )
        return jukf_state

class JEKFObserver():
    def jekf_update(jekf_state,
                u,
                z,
                f_fn,
                h_fn, 
                jekf_params):
        """
        Discrete-time Extended Kalman Filter (EKF) for JAX
        Stateless implementation

        Uses notation of Optimal State Estimation by Dan Simon        
        """
        # Unpack the state and parameters
        x_hat_plus_k_minus_1 =  jekf_state.x_hat
        P_plus_k_minus_1 = jekf_state.P_hat
        # Linearise the system (state transition function, not ode model!) at current point
        w  = jnp.zeros_like(x_hat_plus_k_minus_1) # NOTE change to match state dimension
        v  = jnp.array([0.])
        F_k_minus_1 = jax.jacfwd(f_fn, 0)(x_hat_plus_k_minus_1, u, w) # Size should be (n, n)
        L_k_minus_1 = jax.jacfwd(f_fn, 2)(x_hat_plus_k_minus_1, u, w).T # Size should be (1, n)

        # Predictor (Time update of state estimate and estimation-error covariance)
        P_minus_k = F_k_minus_1 @ P_plus_k_minus_1 @ F_k_minus_1.T + \
            L_k_minus_1 @ jekf_params.Q_joint @ L_k_minus_1.T
        x_hat_minus_k = f_fn(x_hat_plus_k_minus_1, u, w)
        # Linearise noise dynamics
        H_k = jax.jacfwd(h_fn, 0)(x_hat_minus_k, u, v) # Should be (1, n)
        M_k = jax.jacfwd(h_fn, 2)(x_hat_minus_k, u, v) # Should be (1, 1)
        # Corrector
        innovation_k = z - h_fn(x_hat_minus_k, u, v)

        # Innovation covariance
        S = H_k @ P_minus_k @ H_k.T + M_k @ jekf_params.R_joint @ M_k.T
        # Calculate NIS using Cholesky decomposition (more stable)
        # S = L @ L.T
        L = jnp.linalg.cholesky(S)
        # Solve L @ y = innovation_k
        y = jnp.linalg.solve(L, innovation_k)
        # NIS = innovation_k.T @ inv(S) @ innovation_k = y.T @ y
        nis = jnp.sum(y**2) 

        K_part1 = jnp.linalg.solve(L, H_k @ P_minus_k.T)
        K_k = jnp.linalg.solve(L.T, K_part1).T
        
        x_hat_plus = x_hat_minus_k + jnp.matmul(K_k, innovation_k).squeeze()

        # Enforce non-negativity - NOTE deviation from standard EKF
        x_state = jnp.maximum(0, x_hat_plus[:7])
        parameters = x_hat_plus[7:]
        parameters = jnp.clip(x_hat_plus[7:], # OPTION TO CLIP PARAMETERS
                         jnp.array([50.0, jnp.log(1e-12), 0.1, 0.1]), 
                         jnp.array([200.0, jnp.log(1e-6), 2.0, 3.0]))
        x_hat_plus = jnp.concatenate([x_state, parameters])

        # Joseph form for numerical stability:
        I_KH = (jnp.eye(jnp.shape(P_minus_k)[0]) - K_k @ H_k)
        P_k_plus = I_KH @ P_minus_k @ I_KH.T + K_k @ jekf_params.R_joint @ K_k.T

        jekf_state = JointEKFState(x_hat=x_hat_plus,
                                   P_hat=P_k_plus)
        return jekf_state

class DualUKFState(NamedTuple):
    """State of the Dual UKF filter"""
    x_hat: jnp.ndarray  # State estimate
    P_hat: jnp.ndarray  # State covariance
    theta_hat: jnp.ndarray  # Parameter estimate
    P_theta_hat: jnp.ndarray  # Parameter covariance
    inner_counter: int  # Number of inner loop iterations
    x_hat_history: jnp.ndarray  # History of state estimates
    u_history: jnp.ndarray  # History of control inputs

class DualUKFParams(NamedTuple):
    """Parameters for the Dual UKF filter"""
    Q_state: jnp.ndarray  # Process noise covariance for state
    R_state: jnp.ndarray  # Measurement noise covariance
    Q_param: jnp.ndarray  # Process noise covariance for parameters
    R_param: jnp.ndarray  # Measurement noise covariance for parameters
    inner_loops: int  # Number of inner loop iterations
    alpha: float  # UKF alpha parameter
    beta: float  # UKF beta parameter
    kappa: float  # UKF kappa parameter

class SimulationParams(NamedTuple):
    """Fixed parameters for the simulation"""
    dual_inner_loops: int  # Number of inner loop iterations for dual filters

class DualEKFState(NamedTuple):
    """State of the Dual EKF filter"""
    x_hat: jnp.ndarray  # State estimate
    P_hat: jnp.ndarray  # State covariance
    theta_hat: jnp.ndarray  # Parameter estimate
    P_theta_hat: jnp.ndarray  # Parameter covariance
    inner_counter: int  # Number of inner loop iterations

class DualEKFParams(NamedTuple):
    """Parameters for the Dual EKF filter"""
    Q_state: jnp.ndarray  # Process noise covariance for state
    R_state: jnp.ndarray  # Measurement noise covariance
    Q_param: jnp.ndarray  # Process noise covariance for parameters
    R_param: jnp.ndarray  # Measurement noise covariance for parameters
    inner_loops: int  # Number of inner loop iterations

class JointUKFState(NamedTuple):
    """State of the Dual UKF filter"""
    x_hat: jnp.ndarray  # State estimate
    P_hat: jnp.ndarray  # State covariance

class JointUKFParams(NamedTuple):
    """Parameters for the Joint UKF filter"""
    Q_joint: jnp.ndarray  # Process noise covariance for state
    R_joint: jnp.ndarray  # Measurement noise covariance
    alpha: float  # UKF alpha parameter
    beta: float  # UKF beta parameter
    kappa: float  # UKF kappa parameter

class JointEKFState(NamedTuple):
    """State of the Joint EKF filter"""
    x_hat: jnp.ndarray  # State estimate
    P_hat: jnp.ndarray  # State covariance

class JointEKFParams(NamedTuple):
    """Parameters for the Joint EKF filter"""
    Q_joint: jnp.ndarray  # Process noise covariance for state
    R_joint: jnp.ndarray  # Measurement noise covariance

def example():
    # Simulation parameters
    t0 = 0. # hours
    tf = 50. # hours

    step_size = 0.1 # hours, time step for simulation
    max_dt_integration = 0.01 # hours, max time step for integration within RK4
    n_integration = int(ceil(step_size / max_dt_integration))
    dt_integration = step_size / n_integration # hours, actual time step in integration
    n_steps = int(ceil((tf - t0)/step_size)) # number of simulation steps

    # Initial state
    y0 = jnp.array([7.50502485e+05, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.53812442e+07])
    # NOTE glucose removed
    # initial state estimate
    x_hat_ukf = y0.copy()
    # x_hat_0 = jnp.array([7.50502485e+05, 0.00000000e+00, 0.00000000e+00,
    #                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.53812442e+07]) 
    # NOTE glucose removed

    # Model parameters
    theta_dict = {'B': 98, # burst size, number of new viruses released per infected cell
        'log_delta': jnp.log(3.12e-8), # virus absorption rate, 1/hour
        'tau': 0.5, # latent period of virus, hours
        'K': 4, # Monod constant, concentration of substrate at which 
        'c': 100, # ug / mL, concentration of glucose in M9 minimal medium
        'muMax': 1.8, # 0.738, # Growth rate of bacteria, 1/hour
        'e':  1.92e-6, # removal rate of nutrients at maximal bacterial growth, ug / cell / hour
        'monodOn': 0, # Monod equation on (1) or off (0)
        'b2': 98, # burst size of phage2, number of new viruses released per infected cell
        'tau2': 0.5, # latent period of phage2, hours
        'delta2-1': 3.12e-8, # adsorption rate of phage 2 to bacteria 1
        'delta2-2': 3.12e-8, # adsorption rate of phage 2 to bacteria 2
        'delta1-2': 0, # adsorption rate of phage 1 to bacteria 2
        'mu_max2': 1.8 # Maximum growth rate of bacteria 2
        }
    theta_hat_dict = {'B': 98, # burst size, number of new viruses released per infected cell
        'log_delta': jnp.log(3.12e-8), # virus absorption rate, 1/hour
        'tau': 0.5, # latent period of virus, hours
        'muMax': 1.8, # 0.738, # Growth rate of bacteria, 1/hour
        }
    model_params = jnp.array(list(theta_dict.values()), dtype=jnp.float64)
    model_params_hat = jnp.array(list(theta_hat_dict.values()), dtype=jnp.float64)

    # Observer parameters
    Q_0 = jnp.eye(x_hat_ukf.shape[0]) * 1e-6  # Small positive values on the diagonal
    Q_0 = Q_0.at[0, 0].set(1.0e4)  # Set one large variance
    R_0 = jnp.array([[1.0e4]]) # Measurement noise covariance. R>0 to avoid singular matrix
    P0_ukf = jnp.diag(0.001 * jnp.abs(x_hat_ukf) + 1.0e-6) # NOTE change from model8.py
    P_theta_hat_0 = jnp.diag(0.0001 * jnp.abs(model_params_hat) + 1.0e-6)
    # P_theta_hat_0 = jnp.eye(model_params_hat.shape[0]) * 1e-3 # Initial large uncertainty in parameters
    Q_0_theta = jnp.eye(model_params_hat.shape[0]) * 1e-6 # Small process noise for parameters
    Q_0_aug = jnp.block([[Q_0, jnp.zeros((Q_0.shape[0], P_theta_hat_0.shape[0]))],
                           [jnp.zeros((P_theta_hat_0.shape[0], Q_0.shape[0])), Q_0_theta]])
    P_0_aug = jnp.block([[P0_ukf, jnp.zeros((P0_ukf.shape[0], P_theta_hat_0.shape[0]))],
                         [jnp.zeros((P_theta_hat_0.shape[0], P0_ukf.shape[0])), P_theta_hat_0]])

    # Model noise parameters, matching Q_0 and R_0
    Q_true = Q_0.copy()
    R_true = R_0.copy()

    inner_loops = 10
    dukf_params = DualUKFParams(
        Q_state=Q_0,
        R_state=R_0,
        Q_param=Q_0_theta, # Small process noise for parameters
        R_param=R_0,
        inner_loops=inner_loops, # Number of inner loop iterations before parameter update
        alpha=1., # 1e-3,
        beta=0., # 2.0,
        kappa= -1. # 0.0
    )
    dukf_state = DualUKFState(
        x_hat=x_hat_ukf,
        P_hat=P0_ukf,
        theta_hat=model_params_hat,
        P_theta_hat=P_theta_hat_0, # Initial parameter
        inner_counter=1, # NOTE change - start at 1 so first update is state update only - IMPROVEMENT
        x_hat_history=jnp.tile(x_hat_ukf, (inner_loops,1)), # History of state estimates
        u_history=1.25*jnp.ones((inner_loops, 1))  # History NOTE hardcoding input to 1.25 ug/mL
    )
    dekf_params = DualEKFParams(
        Q_state=Q_0,
        R_state=R_0,
        Q_param=Q_0_theta, # Small process noise for parameters
        R_param=R_0,
        inner_loops=inner_loops # Number of inner loop iterations before parameter update
    )
    dekf_state = DualEKFState(
        x_hat=x_hat_ukf,
        P_hat=P0_ukf,
        theta_hat=model_params_hat,
        P_theta_hat=P_theta_hat_0, # Initial parameter
        inner_counter=0
    )
    jukf_state = JointUKFState(
        x_hat=jnp.concatenate([x_hat_ukf, model_params_hat]),
        P_hat=P_0_aug
    )
    jukf_params = JointUKFParams(
        Q_joint=Q_0_aug,
        R_joint=R_0,
        alpha=1, 
        beta=0.,
        kappa=-1.
    )
    jekf_state = JointEKFState(
        x_hat=jnp.concatenate([x_hat_ukf, model_params_hat]),
        P_hat=P_0_aug
    )
    jekf_params = JointEKFParams(
        Q_joint=Q_0_aug,
        R_joint=R_0
    )
    simulation_params = SimulationParams(
        dual_inner_loops=inner_loops
    )

    # Transitions and observations for simulation
    f_fn = Plant.simplified_levin_lenski_one_species
    h_fn = Plant.sensor_model9_one_species
    
    # Transitions and observations for Dual UKF
    def f_state_dukf(x_hat, theta_hat, u):
        _, traj = SimulationTools.reRK4(
            Plant.sllos_dukf,
            [0.0, step_size],
            x_hat,
            N=n_integration,
            args=(u, jnp.zeros_like(x_hat), theta_hat),
        )
        return traj[-1, :]
    def f_param_dukf(theta_hat, x_hat, u):
        return theta_hat # TODO needs to be a random walk with noise - noise input needs to be present
    def h_state_dukf(x_hat, theta_hat, u):
        return Plant.sensor_model9_one_species(0., x_hat, u, 0.)
    # def h_param_dukf(theta_hat, x_hat, u): # Original - worked okay but was wrong conceptually
    #     return Plant.sensor_model9_one_species(0., x_hat, u, 0.)
    @partial(jax.jit,
         static_argnames=("dual_inner_loops"))
    def h_param_dukf(theta_hat,
                     dukf_state: DualUKFState,
                     dual_inner_loops: int = 10):
        """Observation function for parameter UKF
        Need to integrate forward from previous state under theta_hat
        Number of integration steps is dual_inner_loops
        """
        def scan_step(x_prev, index):
            u = jnp.take(dukf_state.u_history, index)
            x_next = f_state_dukf(x_prev, theta_hat, u)
            return x_next, None
        init_carry = dukf_state.x_hat_history[dual_inner_loops - 1, :]
        x_hat, _ = jax.lax.scan(scan_step,
                             init_carry,
                             xs=jnp.arange(dual_inner_loops - 1, -1, -1))

        z = Plant.sensor_model9_one_species(0., x_hat, dukf_state.u_history[0], 0.)
        return z
    f_state_dukf = jax.jit(f_state_dukf)  # JIT compile for better performance
    h_state_dukf = jax.jit(h_state_dukf)  # JIT compile for better performance
    f_param_dukf = jax.jit(f_param_dukf)  # JIT compile for better performance
    h_param_dukf = jax.jit(h_param_dukf)  # JIT compile for better performance
    
    # Transitions and observations for Dual EKF
    def f_state_dekf(x_hat, theta_hat, u, w):
        _, traj = SimulationTools.reRK4(
            Plant.sllos_dukf,
            [0.0, step_size],
            x_hat,
            N=n_integration,
            args=(u, w, theta_hat),
        )
        return traj[-1, :]
    def f_param_dekf(x_hat, theta_hat, u, w):
        """Persistence model for parameters"""
        return theta_hat + w
    def h_state_dekf(x_hat, theta_hat, u, v):
        return Plant.sensor_model9_one_species(0., x_hat, u, v).squeeze(axis=0)
    def h_param_dekf(x_hat, theta_hat, u, v):
        return Plant.sensor_model9_one_species(0., x_hat, u, v).squeeze(axis=0)
    f_state_dekf = jax.jit(f_state_dekf)  # JIT compile for better performance
    h_state_dekf = jax.jit(h_state_dekf)  # JIT compile for better performance
    f_param_dekf = jax.jit(f_param_dekf)  # JIT compile for better performance
    h_param_dekf = jax.jit(h_param_dekf)  # JIT compile for better performance
    
    # Transitions and observations for Joint UKF/EKF
    def f_jukf(x_aug, u):
        _, traj = SimulationTools.reRK4(
            Plant.sllos_jukf,
            [0.0, step_size],
            x_aug,
            N=n_integration,
            args=(u, jnp.zeros_like(x_aug)), # No theta_hat argument here
        )
        return traj[-1, :]
    def h_jukf(x_aug, u):
        x = x_aug[:7]
        return Plant.sensor_model9_one_species(0., x, u, 0.)
    
    # Transitions and observations for Joint EKF
    def f_jekf(x_aug, u, w):
        _, traj = SimulationTools.reRK4(
            Plant.sllos_jukf,
            [0.0, step_size],
            x_aug,
            N=n_integration,
            args=(u, w), # No theta_hat argument here
        )
        return traj[-1, :]
    def h_jekf(x_aug, u, v):
        x = x_aug[:7]
        # Squeeze required to ensure not (1,1,n) but (1,n)
        # NOTE Unclear where the extra dimension comes from
        return Plant.sensor_model9_one_species(0., x, u, v).squeeze(axis=0)

    @jax.jit
    def simulation_scan_step(carry, inputs):
        """
        Inputs:
        x_state: current state
        inputs: list (system_input, system_noise, measurement_noise)
        Outputs:
        x_state
        ys: (x_state, z)
        """
        (x_state, dukf_state, dekf_state, jukf_state, jekf_state) = carry
        system_noise, measurement_noise = inputs
        u = 1.25 # Assume constant input for now - can connect to controller in different script

        # Simulate system forward one step
        t_span = [0., step_size] # time is not used in dynamics
        _, y = SimulationTools.reRK4(f_fn, t_span, x_state,
                                     N=n_integration, args=(u, system_noise, model_params))
        x_state = y[-1, :] # discard intermediate steps
        z = h_fn(0., x_state, u, measurement_noise)

        # DUKF update
        dukf_state = DUKFObserver.dukf_update(dukf_state, z, f_state_dukf,
                                              h_state_dukf, f_param_dukf, h_param_dukf, u, dukf_params)

        # DEKF update
        dekf_state = DEKFObserver.dekf_update(dekf_state, z, f_state_dekf,
                                              h_state_dekf, f_param_dekf, h_param_dekf, u, dekf_params)

        # JUKF update
        jukf_state = JUKFObserver.jukf_update(jukf_state, z, f_jukf, h_jukf, u, jukf_params)

        # JEKF update
        jekf_state = JEKFObserver.jekf_update(jekf_state, u, z, f_jekf, h_jekf, jekf_params)

        new_carry = (x_state, dukf_state, dekf_state, jukf_state, jekf_state)

        outputs = (x_state, z,
                   dukf_state.x_hat, dukf_state.P_hat, dukf_state.theta_hat, dukf_state.P_theta_hat,
                   dekf_state.x_hat, dekf_state.P_hat, dekf_state.theta_hat, dekf_state.P_theta_hat,
                   jukf_state.x_hat, jukf_state.P_hat,
                   jekf_state.x_hat, jekf_state.P_hat)
        return new_carry, outputs

    inputs_all = (jax.random.normal(jax.random.PRNGKey(0),
                                    (n_steps, Q_true.shape[0])) * jnp.sqrt(Q_true.diagonal()),
                  jax.random.normal(jax.random.PRNGKey(1),
                                    (n_steps, R_true.shape[0])) * jnp.sqrt(R_true.diagonal()))
    initial_carry = (y0, dukf_state, dekf_state, jukf_state, jekf_state) # initial state and initial dukf_state
    # Scan forward
    _, outputs = jax.lax.scan(simulation_scan_step,
                            initial_carry,
                            inputs_all)
    x_states, zs, x_hat_dukf, P_hat_dukf, theta_hat_dukf, P_theta_hat_dukf, x_hat_dekf, P_hat_dekf, theta_hat_dekf, P_theta_hat_dekf, x_hat_jukf, P_hat_jukf, x_hat_jekf, P_hat_jekf = outputs

    # Convert to numpy arrays
    x_states = np.array(x_states)
    zs = np.array(zs)
    x_hats_dukf = np.array(x_hat_dukf)
    P_hat_dukf = np.array(P_hat_dukf)
    theta_hat_dukf = np.array(theta_hat_dukf)
    P_theta_hat_dukf = np.array(P_theta_hat_dukf)
    x_hats_dekf = np.array(x_hat_dekf)
    P_hat_dekf = np.array(P_hat_dekf)
    theta_hat_dekf = np.array(theta_hat_dekf)
    P_theta_hat_dekf = np.array(P_theta_hat_dekf)
    x_hats_jukf = np.array(x_hat_jukf)
    P_hat_jukf = np.array(P_hat_jukf)
    x_hats_jekf = np.array(x_hat_jekf)
    P_hat_jekf = np.array(P_hat_jekf)

    simulation_output = (x_states, zs,
                         x_hats_dukf, P_hat_dukf, theta_hat_dukf, P_theta_hat_dukf,
                         x_hats_dekf, P_hat_dekf, theta_hat_dekf, P_theta_hat_dekf,
                         x_hats_jukf, P_hat_jukf,
                         x_hats_jekf, P_hat_jekf)
    return simulation_output

def plot_results(x_states, zs, x_hats_dukf, P_hat_dukf, theta_hat_dukf, P_theta_hat_dukf, x_hats_dekf, P_hat_dekf, theta_hat_dekf, P_theta_hat_dekf, x_hats_jukf, P_hat_jukf, x_hats_jekf, P_hat_jekf): # NOTE terrible graphs at the moment
    time = np.arange(x_states.shape[0]) * 0.1 # hours

    max_var = np.max([np.max(P_hat_dukf.diagonal()), np.max(P_hat_dekf.diagonal()), np.max(P_hat_jukf.diagonal()), np.max(P_hat_jekf.diagonal())])
    if max_var < 1e8:
        clip = False
    else:
        print(f'Warning: large variance in estimates: {max_var}.')
        print('Clipping graphs to improve visibility.')
        clip = True

    # Define consistent colors for each filter
    color_dukf = 'tab:blue'
    color_dekf = 'tab:purple'
    color_jukf = 'tab:orange'
    color_jekf = 'tab:green'
    color_true = 'black'
    color_meas = 'red'
    # For parameter plots
    color_dukf_param = color_dukf
    color_dekf_param = color_dekf
    color_jukf_param = color_jukf
    color_jekf_param = color_jekf
    

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    total_true = x_states[:, 0] + x_states[:, 1] + x_states[:, 2] + x_states[:, 3] + x_states[:, 4] + x_states[:, 5]
    total_dukf = x_hats_dukf[:, 0] + x_hats_dukf[:, 1] + x_hats_dukf[:, 2] + x_hats_dukf[:, 3] + x_hats_dukf[:, 4] + x_hats_dukf[:, 5]
    total_dekf = x_hats_dekf[:, 0] + x_hats_dekf[:, 1] + x_hats_dekf[:, 2] + x_hats_dekf[:, 3] + x_hats_dekf[:, 4] + x_hats_dekf[:, 5]
    total_jukf = x_hats_jukf[:, 0] + x_hats_jukf[:, 1] + x_hats_jukf[:, 2] + x_hats_jukf[:, 3] + x_hats_jukf[:, 4] + x_hats_jukf[:, 5]
    total_jekf = x_hats_jekf[:, 0] + x_hats_jekf[:, 1] + x_hats_jekf[:, 2] + x_hats_jekf[:, 3] + x_hats_jekf[:, 4] + x_hats_jekf[:, 5]
    plt.plot(time, total_true, label='True S&I', color=color_true)
    plt.plot(time, total_dukf, label='DUKF S&I', color=color_dukf, linestyle='--')
    plt.plot(time, total_dekf, label='DEKF S&I', color=color_dekf, linestyle='--')
    plt.plot(time, total_jukf, label='JUKF S&I', color=color_jukf, linestyle='--')
    plt.plot(time, total_jekf, label='JEKF S&I', color=color_jekf, linestyle='--')
    plt.fill_between(time,
                     total_dukf - 2 * np.sqrt(P_hat_dukf[:, 0, 0]),
                     total_dukf + 2 * np.sqrt(P_hat_dukf[:, 0, 0]),
                     color=color_dukf, alpha=0.3)
    plt.fill_between(time,
                     total_dekf - 2 * np.sqrt(P_hat_dekf[:, 0, 0]),
                     total_dekf + 2 * np.sqrt(P_hat_dekf[:, 0, 0]),
                     color=color_dekf, alpha=0.3)
    plt.fill_between(time,
                     total_jukf - 2 * np.sqrt(P_hat_jukf[:, 0, 0]),
                     total_jukf + 2 * np.sqrt(P_hat_jukf[:, 0, 0]),
                     color=color_jukf, alpha=0.3)
    plt.fill_between(time,
                     total_jekf - 2 * np.sqrt(P_hat_jekf[:, 0, 0]),
                     total_jekf + 2 * np.sqrt(P_hat_jekf[:, 0, 0]),
                     color=color_jekf, alpha=0.3)
    plt.scatter(time, zs[:, 0], label='Measurements', color=color_meas, s=10)
    plt.xlabel('Time (hours)')
    plt.ylabel('Bacteria S1')
    plt.title('Bacteria Population and Filter Estimates')
    if clip:
        plt.ylim(0, 1e7)
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, x_states[:, 6], label='True P1', color=color_true)
    plt.plot(time, x_hats_dukf[:, 6], label='DUKF P1', color=color_dukf, linestyle='--')
    plt.plot(time, x_hats_dekf[:, 6], label='DEKF P1', color=color_dekf, linestyle='--')
    plt.plot(time, x_hats_jukf[:, 6], label='JUKF P1', color=color_jukf, linestyle='--')
    plt.plot(time, x_hats_jekf[:, 6], label='JEKF P1', color=color_jekf, linestyle='--')
    plt.fill_between(time,
                     x_hats_dukf[:, 6] - 2 * np.sqrt(P_hat_dukf[:, 6, 6]),
                     x_hats_dukf[:, 6] + 2 * np.sqrt(P_hat_dukf[:, 6, 6]),
                     color=color_dukf, alpha=0.3)
    plt.fill_between(time,
                     x_hats_dekf[:, 6] - 2 * np.sqrt(P_hat_dekf[:, 6, 6]),
                     x_hats_dekf[:, 6] + 2 * np.sqrt(P_hat_dekf[:, 6, 6]),
                     color=color_dekf, alpha=0.3)
    plt.fill_between(time,
                     x_hats_jukf[:, 6] - 2 * np.sqrt(P_hat_jukf[:, 6, 6]),
                     x_hats_jukf[:, 6] + 2 * np.sqrt(P_hat_jukf[:, 6, 6]),
                     color=color_jukf, alpha=0.3)
    plt.fill_between(time,
                     x_hats_jekf[:, 6] - 2 * np.sqrt(P_hat_jekf[:, 6, 6]),
                     x_hats_jekf[:, 6] + 2 * np.sqrt(P_hat_jekf[:, 6, 6]),
                     color=color_jekf, alpha=0.3)
    plt.xlabel('Time (hours)')
    plt.ylabel('Phage P1')
    plt.title('Phage Population and Filter Estimates')
    plt.legend()
    plt.grid()
    if clip:
        plt.ylim(0, 1e8)
    plt.tight_layout()
    plt.savefig('supplementary/supplementary/outputs/param_state_comparison_state.png')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.suptitle('Parameter Estimates and Uncertainty')
    param_names = ['B', 'delta', 'tau', 'muMax']
    for i in range(theta_hat_dukf.shape[1]):
        plt.subplot(2, 2, 1 + i)
        plt.plot(time, theta_hat_dukf[:, i], label=f'DUKF {param_names[i]}', color=color_dukf)
        plt.plot(time, theta_hat_dekf[:, i], label=f'DEKF {param_names[i]}', color=color_dekf)
        plt.plot(time, x_hats_jukf[:, 7 + i], label=f'JUKF {param_names[i]}', color=color_jukf)
        plt.plot(time, x_hats_jekf[:, 7 + i], label=f'JEKF {param_names[i]}', color=color_jekf)
        plt.fill_between(time,
                         theta_hat_dukf[:, i] - 2 * np.sqrt(P_theta_hat_dukf[:, i, i]),
                         theta_hat_dukf[:, i] + 2 * np.sqrt(P_theta_hat_dukf[:, i, i]),
                         alpha=0.3, color=color_dukf)
        plt.fill_between(time,
                         theta_hat_dekf[:, i] - 2 * np.sqrt(P_theta_hat_dekf[:, i, i]),
                         theta_hat_dekf[:, i] + 2 * np.sqrt(P_theta_hat_dekf[:, i, i]),
                         alpha=0.3, color=color_dekf)
        plt.fill_between(time,
                         x_hats_jukf[:, 7 + i] - 2 * np.sqrt(P_hat_jukf[:, 7 + i, 7 + i]),
                         x_hats_jukf[:, 7 + i] + 2 * np.sqrt(P_hat_jukf[:, 7 + i, 7 + i]),
                         alpha=0.3, color=color_jukf)
        plt.fill_between(time,
                         x_hats_jekf[:, 7 + i] - 2 * np.sqrt(P_hat_jekf[:, 7 + i, 7 + i]),
                         x_hats_jekf[:, 7 + i] + 2 * np.sqrt(P_hat_jekf[:, 7 + i, 7 + i]),
                         alpha=0.3, color=color_jekf)
        plt.legend()
        plt.grid()
        if clip:
            plt.ylim(0, np.max(theta_hat_dukf[:, i]) * 1.5)
        plt.xlabel('Time (hours)')
        plt.ylabel('Parameter Estimates')
    plt.tight_layout()
    plt.savefig('supplementary/outputs/param_state_comparison_param.png')
    plt.show()

if __name__ == "__main__":
    os.makedirs('supplementary/outputs', exist_ok=True)
    x_states, zs, x_hats_dukf, Ps_dukf, theta_hat_dukf, P_theta_hat_dukf, x_hats_dekf, P_hat_dekf, theta_hat_dekf, P_theta_hat_dekf, x_hats_jukf, P_hat_jukf, x_hats_jekf, P_hat_jekf = example()
    plot_results(x_states, zs, x_hats_dukf, Ps_dukf, theta_hat_dukf, P_theta_hat_dukf, x_hats_dekf, P_hat_dekf, theta_hat_dekf, P_theta_hat_dekf, x_hats_jukf, P_hat_jukf, x_hats_jekf, P_hat_jekf)