"""ODE models of phage-bacteria ecology"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .numerical_methods import reRK4

class Plant:
    def ode_model(t, y, u, w, model_params) -> jnp.array:
        """
        Wrapper function for the system dynamics
        """
        # return Plant.ode_model_nonlinearity(t, y, u, w, model_params)
        # return Plant.ode_model_moi(t, y, u, w, model_params)
        return Plant.ode_model9(t, y, u, w, model_params)

    def ode_model_to_integrate(t, y, args) -> jnp.array:
        """
        Wrapper function for the system dynamics
        """
        u, w, model_params = args
        return Plant.ode_model(t, y, u, w, model_params)

    def ode_model_to_integrate_one_species(t, y, args) -> jnp.array:
        """
        Wrapper function for the system dynamics
        """
        u, w, model_params = args
        return Plant.ode_model_one_species(t, y, u, w, model_params)

    def sensor(t, x, u, v, args=()):
        """
        Wrapper function for the sensor model
        """
        return Plant.sensor_model9(t, x, u, v, args)

    def manual_disturbance(t, y: jnp.ndarray, disturbance_params: NamedTuple):
        """
        Wrapper function for the disturbance dynamics

        Can be used to model emergence of a second strain
        """
        # Default disturbance (no disturbance)
        default_disturbance = jnp.zeros_like(y)
        # Check if the current time matches any predefined times
        def disturbance_fn(idx):
            return disturbance_params.disturbances[idx]
        def no_disturbance_fn(_):
            return default_disturbance
        # Iterate over the times and apply the corresponding disturbance
        def apply_disturbance(carry, time_idx):
            pred = jnp.squeeze(jnp.isclose(t, disturbance_params.times[time_idx], atol=1e-6))
            current_disturbance = jax.lax.cond(
                pred,
                lambda _: disturbance_fn(time_idx),
                lambda _: no_disturbance_fn(None),
                None
            )
            return carry + current_disturbance, None
        # Conditionally scan only if disturbance_params.times is non-empty
        def scan_disturbances(_):
            total_disturbance, _ = jax.lax.scan(
                apply_disturbance,
                default_disturbance,
                jnp.arange(jnp.size(disturbance_params.times))
            )
            return total_disturbance
        # If times is empty, return default disturbance
        total_disturbance = jax.lax.cond(
            disturbance_params.times.size > 0,
            lambda _: scan_disturbances(None),
            lambda _: default_disturbance,
            None
        )
        return total_disturbance

    def update(t,
               x,
               control_signal,
               plant_noise,
               plant_params,
               simulation_params):
        """
        Parameters:
        t: float
            time, hours
        x: array
            state variables
        control_signal: float
            control signal
        plant_noise: float
            noise
        plant_params: tuple
            parameters for the system
        simulation_params: tuple
            parameters for the simulation
        Returns:
        x: updated state variables
        """
        _, x_traj = reRK4(Plant.ode_model_to_integrate,
                                   (t, t+simulation_params.STEP_SIZE),
                                   x,
                                   N=simulation_params.n_integration,
                                   args=((control_signal, plant_noise, plant_params.model_params),))
        x_temp = x_traj[-1, :]
        x = x_temp + Plant.manual_disturbance(t, x_temp, plant_params.disturbance_params)
        return x

    def sensor_model9(t, x, u, v, args=()):
        """
        Complies with the observer function signature
        h: observation function, h(t, x, u, v, args)

        Inputs:
        t: time
        x: state variables: S1, I11, I12, I13, I14, I15, P1, S2, I21, I22, I23, I24, I25, P2
        u: control input
        v: float NOTE change from model7
            observation noise
        """
        observation_true = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[7] + x[8] +\
              x[9] + x[10] + x[11] + x[12]
        observation_made = observation_true + v
        return observation_made # shape (1,)

    def sensor_one_species(t, x, u, v, args=()):
        """
        Complies with the observer function signature
        h: observation function, h(t, x, u, v, args)

        Inputs:
        t: time
        x: state variables: S1, I11, I12, I13, I14, I15, P1, S2, I21, I22, I23, I24, I25, P2
        u: control input
        v: float NOTE change from model7
            observation noise
        """
        observation_true = x[0] + x[1] + x[2] + x[3] + x[4] + x[5]
        observation_made = observation_true + v
        # print('observation_made:', jnp.shape(observation_made)) # debugging
        return jnp.array([observation_made]) # shape (1,)

    def ode_model_one_species_extended(t, y, u, w_noise, model_params) -> jnp.array:
        """
        1 Bacterial strains, 1 phage strains
        No dependence on Volume

        Inputs:
        t: time
        y: state variables, S, I, P
        args: parameters (model_params, control_signal)
        model_params: 
        model_params[0]: burst size, B1
        model_params[1]: log delta, adsorption rate, delta1-1 (Phage 1 to Bacteria 1)
        model_params[2]: time delay, tau1
        model_params[3]: carrying capacity, K (monod constant)
        model_params[4]: concentration of glucose in input media, c
        model_params[5]: maximum growth rate of bacteria 1, mu_max1
        model_params[6]: removal rate of nutrients, e
        model_params[7]: monodOn: bool NOT NEEDED
        model_params[8]: burst size, B2 NOT USED
        model_params[9]: time delay, tau2 NOT USED
        model_params[10]: log adsorption rate, delta2-1 (Phage 2 to Bacteria 1) NOT USED
        model_params[11]: log adsorption rate, delta2-2 (Phage 2 to Bacteria 2) NOT USED
        model_params[12]: log adsorption rate, delta1-2 (Phage 1 to Bacteria 2) NOT USED
        model_params[13]: maximum growth rate of bacteria 2, mu_max2 NOT USED
        control_signal: turnover rate, omega
        
        Outputs:
        dy: time derivatives of state variables 
        """
        M = 5
        control_signal = u
        # Ensure no negative values in y
        y = jnp.maximum(y, 0.)

        S1, I11, I12, I13, I14, I15, P1, mp0, mp1, mp2, mp3, mp4, mp5, mp6,  = y
        delta1 = jnp.clip(jnp.exp(mp1), 1e-12, 1e-6)

        dS1 = mp5 * S1 \
            - delta1 * S1 * P1  - control_signal * S1 + w_noise[0]
        dI11 = delta1 * S1 * P1  - M / mp2 * I11 - control_signal * I11 + w_noise[1]
        dI12 = M / mp2 * (I11 - I12) - control_signal * I12 + w_noise[2]
        dI13 = M / mp2 * (I12 - I13) - control_signal * I13 + w_noise[3]
        dI14 = M / mp2 * (I13 - I14) - control_signal * I14 + w_noise[4]
        dI15 = M / mp2 * (I14 - I15) - control_signal * I15 + w_noise[5]
        dP1 = mp0 * M / mp2 * I15 - delta1 * S1 * P1 - control_signal * P1 + w_noise[6]
        dmp0, dmp1, dmp2, dmp3, dmp4, dmp5, dmp6 = (0., 0., 0., 0., 0., 0., 0.)

        dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1, dmp0, \
                        dmp1, dmp2, dmp3, dmp4, dmp5, dmp6])
        return dy

    def ode_model_one_species(t, y, u, w_noise, model_params) -> jnp.array:
        """
        1 Bacterial strains, 1 phage strains
        No dependence on Volume

        Inputs:
        t: time
        y: state variables, S, I, P
        args: parameters (model_params, control_signal)
        model_params: 
        model_params[0]: burst size, B1
        model_params[1]: log adsorption rate, delta1-1 (Phage 1 to Bacteria 1)
        model_params[2]: time delay, tau1
        model_params[3]: carrying capacity, K (monod constant)
        model_params[4]: concentration of glucose in input media, c
        model_params[5]: maximum growth rate of bacteria 1, mu_max1
        model_params[6]: removal rate of nutrients, e
        model_params[7]: monodOn: bool
        model_params[8]: burst size, B2
        model_params[9]: time delay, tau2
        model_params[10]: log adsorption rate, delta2-1 (Phage 2 to Bacteria 1)
        model_params[11]: log adsorption rate, delta2-2 (Phage 2 to Bacteria 2)
        model_params[12]: log adsorption rate, delta1-2 (Phage 1 to Bacteria 2)
        model_params[13]: maximum growth rate of bacteria 2, mu_max2
        control_signal: turnover rate, omega
        
        Outputs:
        dy: time derivatives of state variables 
        """
        M = 5
        control_signal = u
        # Ensure no negative values in y
        y = jnp.maximum(y, 0.)
        delta11 = jnp.clip(jnp.exp(model_params[1]), 1e-12, 1e-6)

        S1, I11, I12, I13, I14, I15, P1  = y

        dS1 = model_params[5] * S1 - delta11 * S1 * P1 \
                                      - control_signal * S1 + w_noise[0]
        dI11 = delta11 * S1 * P1  - M / model_params[2] * I11 - control_signal * I11 + w_noise[1]
        dI12 = M / model_params[2] * (I11 - I12) - control_signal * I12 + w_noise[2]
        dI13 = M / model_params[2] * (I12 - I13) - control_signal * I13 + w_noise[3]
        dI14 = M / model_params[2] * (I13 - I14) - control_signal * I14 + w_noise[4]
        dI15 = M / model_params[2] * (I14 - I15) - control_signal * I15 + w_noise[5]
        dP1 = model_params[0] * M / model_params[2] * I15 - delta11 * S1 * P1 \
            - control_signal * P1 + w_noise[6]

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
            - control_signal * I11 + w_noise[1]
        dI12 = M / model_params[2] * (I11 - I12) - control_signal * I12 + w_noise[2]
        dI13 = M / model_params[2] * (I12 - I13) - control_signal * I13 + w_noise[3]
        dI14 = M / model_params[2] * (I13 - I14) - control_signal * I14 + w_noise[4]
        dI15 = M / model_params[2] * (I14 - I15) - control_signal * I15 + w_noise[5]
        dP1 = model_params[0] * M / model_params[2] * I15 \
            - delta_clipped * S1 * P1 - control_signal * P1 + w_noise[6]

        dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1])
        return dy

    def ode_model9(t, y, u, w_noise, model_params) -> jnp.array:
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
        delta11 = jnp.clip(jnp.exp(model_params[1]), 1e-12, 1e-6)
        delta21 = jnp.clip(jnp.exp(model_params[10]), 1e-12, 1e-6)
        delta22 = jnp.clip(jnp.exp(model_params[11]), 1e-12, 1e-6)
        delta12 = jnp.clip(jnp.exp(model_params[12]), 1e-12, 1e-6)

        S1, I11, I12, I13, I14, I15, P1, S2, I21, I22, I23, I24, I25, P2  = y

        dS1 = model_params[5] * S1 - delta11 * S1 * P1 \
                                 - delta21 * S1 * P2  - control_signal * S1 + w_noise[0]
        dI11 = delta11 * S1 * P1 + delta12 * S2 * P1  - M / model_params[2] * I11 \
            - control_signal * I11 + w_noise[1]
        dI12 = M / model_params[2] * (I11 - I12) - control_signal * I12 + w_noise[2]
        dI13 = M / model_params[2] * (I12 - I13) - control_signal * I13 + w_noise[3]
        dI14 = M / model_params[2] * (I13 - I14) - control_signal * I14 + w_noise[4]
        dI15 = M / model_params[2] * (I14 - I15) - control_signal * I15 + w_noise[5]
        dP1 = model_params[0] * M / model_params[2] * I15 \
            - delta11 * S1 * P1 - delta12 * S2 * P1 - control_signal * P1 + w_noise[6]
        dS2 = model_params[13] * S2 \
                - delta12 * S2 * P1 - delta22 * S2 * P2 - control_signal * S2 + w_noise[7]
        dI21 = delta21 * S1 * P2 + delta22 * S2 * P2  \
            - M / model_params[9] * I21 - control_signal * I21 + + w_noise[8]
        dI22 = M / model_params[9] * (I21 - I22) - control_signal * I22 + w_noise[9]
        dI23 = M / model_params[9] * (I22 - I23) - control_signal * I23 + w_noise[10]
        dI24 = M / model_params[9] * (I23 - I24) - control_signal * I24 + w_noise[11]
        dI25 = M / model_params[9] * (I24 - I25) - control_signal * I25 + w_noise[12]
        dP2 = model_params[8] * M / model_params[9] * I25 - delta21 * S1 * P2 \
            - delta22 * S2 * P2  - control_signal * P2 + w_noise[13]

        dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1, \
                        dS2, dI21, dI22, dI23, dI24, dI25, dP2])
        return dy

    def ode_model_moi(t, y, u, w_noise, model_params) -> jnp.array:
        """
        Alternative plant model for testing control with model mismatch.
        Introduces MOI-dependent phage adsorption; phage can bind to 
        already infected bacteria.

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
        delta11 = jnp.clip(jnp.exp(model_params[1]), 1e-12, 1e-6)
        delta21 = jnp.clip(jnp.exp(model_params[10]), 1e-12, 1e-6)
        delta22 = jnp.clip(jnp.exp(model_params[11]), 1e-12, 1e-6)
        delta12 = jnp.clip(jnp.exp(model_params[12]), 1e-12, 1e-6)

        S1, I11, I12, I13, I14, I15, P1, S2, I21, I22, I23, I24, I25, P2  = y

        dS1 = model_params[5] * S1 - delta11 * S1 * P1 \
                                - model_params[11] * S1 * P2  \
                                - control_signal * S1 \
                                + w_noise[0]
        dI11 = delta11 * S1 * P1 + model_params[12] * S2 * P1  - M / model_params[2] * I11 \
            - control_signal * I11 + w_noise[1]
        dI12 = M / model_params[2] * (I11 - I12) - control_signal * I12 + w_noise[2]
        dI13 = M / model_params[2] * (I12 - I13) - control_signal * I13 + w_noise[3]
        dI14 = M / model_params[2] * (I13 - I14) - control_signal * I14 + w_noise[4]
        dI15 = M / model_params[2] * (I14 - I15) - control_signal * I15 + w_noise[5]
        dP1 = model_params[0] * M / model_params[2] * I15 \
            - delta11 * (S1 + I11 + I12 + I13 + I14 + I15) * P1 \
            - model_params[12] * (S2 + I21 + I22 + I23 + I24 + I25) * P1 \
            - control_signal * P1 + w_noise[6]
        dS2 = model_params[13] * S2 \
                - delta12 * S2 * P1 - delta22 * S2 * P2 \
                - control_signal * S2 + w_noise[7]
        dI21 = delta21 * S1 * P2 + delta22 * S2 * P2  \
            - M / model_params[9] * I21 - control_signal * I21 + w_noise[8]
        dI22 = M / model_params[9] * (I21 - I22) - control_signal * I22 + w_noise[9]
        dI23 = M / model_params[9] * (I22 - I23) - control_signal * I23 + w_noise[10]
        dI24 = M / model_params[9] * (I23 - I24) - control_signal * I24 + w_noise[11]
        dI25 = M / model_params[9] * (I24 - I25) - control_signal * I25 + w_noise[12]
        dP2 = model_params[8] * M / model_params[9] * I25 \
            - delta21 * (S1 + I11 + I12 + I13 + I14 + I15) * P2 \
            - delta22 * (S2 + I21 + I22 + I23 + I24 + I25) * P2  \
            - control_signal * P2 + w_noise[13]

        dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1, \
                        dS2, dI21, dI22, dI23, dI24, dI25, dP2])
        return dy

    def ode_model_arbitrary_nonlinearity(t, y, u, w_noise, model_params) -> jnp.array:
        """
        Alternative plant model for testing control with model mismatch.

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
        delta11 = jnp.clip(jnp.exp(model_params[1]), 1e-12, 1e-6)
        delta21 = jnp.clip(jnp.exp(model_params[10]), 1e-12, 1e-6)
        delta22 = jnp.clip(jnp.exp(model_params[11]), 1e-12, 1e-6)
        delta12 = jnp.clip(jnp.exp(model_params[12]), 1e-12, 1e-6)

        S1, I11, I12, I13, I14, I15, P1, S2, I21, I22, I23, I24, I25, P2  = y

        dS1 = model_params[5] * S1 - delta11 * S1 * (P1 + 0.0000001*(P1-1.5e7)**2) \
                                 - delta21 * S1 * P2  - control_signal * S1 + w_noise[0]
        dI11 = delta11 * S1 * (P1 + 0.0000001*(P1-1.5e7)**2) + delta12 * S2 * P1  - M / model_params[2] * I11 \
            - control_signal * I11 + w_noise[1]
        dI12 = M / model_params[2] * (I11 - I12) - control_signal * I12 + w_noise[2]
        dI13 = M / model_params[2] * (I12 - I13) - control_signal * I13 + w_noise[3]
        dI14 = M / model_params[2] * (I13 - I14) - control_signal * I14 + w_noise[4]
        dI15 = M / model_params[2] * (I14 - I15) - control_signal * I15 + w_noise[5]
        dP1 = model_params[0] * M / model_params[2] * I15 \
            - delta11 * S1 * P1 - delta12 * S2 * P1 - control_signal * P1 + w_noise[6]
        dS2 = model_params[13] * S2 \
                - delta12 * S2 * P1 - delta22 * S2 * P2 - control_signal * S2 + w_noise[7]
        dI21 = delta21 * S1 * P2 + delta22 * S2 * P2  \
            - M / model_params[9] * I21 - control_signal * I21 + w_noise[8]
        dI22 = M / model_params[9] * (I21 - I22) - control_signal * I22 + w_noise[9]
        dI23 = M / model_params[9] * (I22 - I23) - control_signal * I23 + w_noise[10]
        dI24 = M / model_params[9] * (I23 - I24) - control_signal * I24 + w_noise[11]
        dI25 = M / model_params[9] * (I24 - I25) - control_signal * I25 + w_noise[12]
        dP2 = model_params[8] * M / model_params[9] * I25 - delta21 * S1 * P2 \
            - delta22 * S2 * P2  - control_signal * P2 + w_noise[13]

        dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1, \
                        dS2, dI21, dI22, dI23, dI24, dI25, dP2])
        
        return dy
