"""Functions to construct and run simulations"""
from functools import partial
import jax
import jax.numpy as jnp
from jax_tqdm import loop_tqdm
from .plant_model import Plant
from .observer import Observer
from .controller import Controller
from .types import PlantState

class Simulation:
    """Functions for simulation"""
    @partial(jax.jit,
         static_argnames=("simulation_params"))
    def simulation_step(temp_states: tuple,
                        control_system_params: tuple,
                        plant_params: tuple,
                        simulation_params: tuple):
        """
        Simulate the system
        """
        # Unpack temporary states
        t, key, plant_state, control_system_state, control_signal = temp_states

        # 1. Run plant
        key, subkey = jax.random.split(key)
        plant_noise = jax.random.multivariate_normal(subkey, jnp.zeros(jnp.shape(plant_params.noise_params[1])[0]), plant_params.noise_params[1])
        x = Plant.update(t,
                         plant_state.x,
                         control_signal,
                         plant_noise,
                         plant_params,
                         simulation_params)
        # 2. Measure plant
        key, subkey = jax.random.split(key)
        sensor_noise = plant_params.noise_params[0] * jax.random.normal(subkey, (1,))
        z = Plant.sensor(t,
                         x,
                         control_signal,
                         sensor_noise)
        # 3. Update observer - needs to use measurements from the updated plant
        key, subkey = jax.random.split(key)
        control_system_state = Observer.update(subkey,
                                               t, # time - redundant currently
                                               control_signal, # u
                                               z, # observation
                                               control_system_state,
                                               control_system_params,
                                               simulation_params)
        z_hat = Plant.sensor_one_species(t,
                                         control_system_state.x_hat,
                                         control_signal,
                                         0.)
        # 4. Generate control signal for next time step
        # Using PID controller for SISO control of OD
        # control_signal, controller_state = Controller.pid_controller(z_hat,
        #                                                              control_system_state.controller_state,
        #                                                              control_system_params.controller_params,
        #                                                              simulation_params.STEP_SIZE
        #                                                              )
        # Or a Smith predictor
        control_signal, control_system_state = Controller.smith(control_system_state,
                                                            control_system_params,
                                                            simulation_params)

        # Prevent negative values, and ensure it doesn't track negligibly small values
        x = jnp.where(x < 1, 0., x)
        x_hat = jnp.where(control_system_state.x_hat < 1, 0., control_system_state.x_hat)
        control_system_state = control_system_state._replace(x_hat=x_hat)

        t = t + simulation_params.STEP_SIZE

        plant_state = PlantState(x)
        temp_states = (t, key, plant_state, control_system_state, control_signal)
        return temp_states

    @partial(jax.jit,
            static_argnames=("simulation_params"))
    def run_simulation(plant_state,
                       control_system_state,
                       simulation_params,
                       control_system_params,
                       plant_params,
                       random_key,
                       verbose=False):
        """
        Run the simulation

        Parameters:
        key: 
            PRNG key
        y0: jnp.ndarray
            Initial state
        y_hat_0: jnp.ndarray
            Initial state estimate
        t_span: tuple
            t0: float
                start time
            tf: float
                end time
        model_params: tuple
        controller_state: 
        """
        x_history = jnp.zeros((simulation_params.n_steps, jnp.shape(plant_state.x)[0]))
        x_hat_history = jnp.zeros((simulation_params.n_steps,
                                   jnp.shape(control_system_state.x_hat)[0]))
        z_history = jnp.zeros((simulation_params.n_steps, 1))
        nis_history = jnp.zeros((simulation_params.n_steps, 1))
        param_history = jnp.zeros((simulation_params.n_steps,
                                   jnp.shape(control_system_state.model_params_hat)[0]))
        u_history = 1.25*jnp.ones((simulation_params.n_steps, 1))
        x_history = x_history.at[0].set(plant_state.x)
        x_hat_history = x_hat_history.at[0].set(control_system_state.x_hat)
        z_history = z_history.at[0].set(0.)
        nis_history = nis_history.at[0].set(0.)
        param_history = param_history.at[0].set(control_system_state.model_params_hat)
        state_history = (x_history, x_hat_history, z_history, nis_history, u_history, param_history)

        t0 = simulation_params.t0
        control_signal = 0. # Initial guess
        temp_states = (t0, random_key, plant_state, control_system_state, control_signal)

        @loop_tqdm(simulation_params.n_steps-1) # REMOVE IF PROGRESS BAR NOT NEEDED
        def body_fun(i, y):
            state_history, temp_states = y
            temp_states = Simulation.simulation_step(temp_states,
                                                    control_system_params,
                                                    plant_params,
                                                    simulation_params)

            # Save to history
            x_history, x_hat_history, z_history, nis_history, u_history, param_history = state_history
            # all are multi-element 1D arrays
            x_history = x_history.at[i+1].set(temp_states[2].x)
            x_hat_history = x_hat_history.at[i+1].set(temp_states[3].x_hat)
            z_history = z_history.at[i+1].set(temp_states[3].z_history[0])
            nis_history = nis_history.at[i+1].set(jnp.sum(temp_states[3].nis_history)) # AVG NIS
            u_history = u_history.at[i+1].set(temp_states[4])
            param_history = param_history.at[i+1].set(temp_states[3].model_params_hat)
            state_history = (x_history, x_hat_history, z_history, nis_history, u_history, param_history)

            y = (state_history, temp_states)
            return y

        y = jax.lax.fori_loop(0,
                              simulation_params.n_steps-1,
                              body_fun,
                              (state_history, temp_states))
        state_history, temp_states = y

        return state_history
