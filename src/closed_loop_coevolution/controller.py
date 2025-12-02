"""Controller module for closed-loop coevolution simulations"""
import jax
import jax.numpy as jnp
from .numerical_methods import reRK4
from .plant_model import Plant

class Controller:
    def pid_controller(measured_value: float,
                       controller_state: tuple,
                       params: tuple,
                       dt):
        """
        PID controller function

        Parameters:
        measured_value: float 
            the value of the process variable that is being controlled
        controller_state: list
            the state of the controller
            [integral, previous_error, set_point]
        params: tuple
            the controller parameters
            (kp, ki, kd, offset)
        dt: float
            the time step
        
        Returns:
        control_signal: float
            the control signal to be sent to the plant
        new_state: list
            the updated controller state
        """
        integral, previous_error, set_point = controller_state
        kp, ki, kd, offset = params

        error = measured_value - set_point # Careful with the sign
        integral += error * dt
        derivative = (error - previous_error) / dt
        control_signal = kp * error + ki * integral + kd * derivative + offset
        control_signal = jnp.maximum(0.0, control_signal)  # Ensure control signal is non-negative
        new_state = (integral, error, set_point)

        return control_signal, new_state
    def smith(control_system_state: tuple,
              control_system_params: tuple,
              simulation_params: tuple):
        """
        Smith predictor controller

        Note that if using, need to change the controller_state
        declaration, and will want to use different PID parameters
        Will also want to change the way control_system_state is updated
        in Simulation.simulation_step
        """
        def integrate_forward(x_hat_0, model_params_hat):
            """Integrate state estimate forward through the window"""
            def scan_step(x_hat, idx):
                # Assumes constant control signal
                current_u = control_system_params.controller_params[3]
                # Can use previous control signal, but this is unstable - BEWARE
                # current_u = control_system_state.u_history[0]
                _, x_hat = reRK4(Plant.sllos_dukf,
                                        (0., simulation_params.STEP_SIZE), # time is arbitrary here
                                        x_hat,
                                        N=simulation_params.n_integration,
                                        args=(current_u, jnp.array([0.]), model_params_hat))
                x_hat = x_hat[-1, :] # Get the last value
                y_estimated = Plant.sensor_one_species(0.,
                                           x_hat,
                                           current_u,
                                           jnp.array([0.])) # time is arbitrary here
                return x_hat, y_estimated
            # Use scan with a simpler signature that just passes the state forward
            _, y_fut = jax.lax.scan(scan_step,
                                    x_hat_0,
                                    jnp.arange(simulation_params.prediction_horizon))
            y_predicted = y_fut[-1].squeeze()
            return y_predicted
        low_level_state, smith_predictions, filter_state = control_system_state.controller_state

        smith_predictions = jnp.roll(smith_predictions, 1) # Predicted total bacteria
        z_predicted = integrate_forward(control_system_state.x_hat,
                                            control_system_state.model_params_hat)
        smith_predictions = smith_predictions.at[0].set(z_predicted)

        # difference between the actual output and the output that was predicted previously
        d_p_unfiltered = control_system_state.z_history[0] - smith_predictions[-1]
        d_p, filter_state = Controller.filter(d_p_unfiltered, filter_state)
        signal = smith_predictions[0] + d_p

        # signal = control_system_state.z_history[0] # Uncomment to disable Smith predictor

        # Apply a simple proportional control based on signal
        control_signal, low_level_state = Controller.low_level_controller(signal,
                                                     low_level_state,
                                                     control_system_params.controller_params,
                                                     simulation_params.STEP_SIZE)

        # repackage state
        controller_state = (low_level_state, smith_predictions, filter_state)
        control_system_state = control_system_state._replace(controller_state=controller_state)

        return control_signal, control_system_state
    def low_level_controller(signal, state, params, dt):
        """
        Low level controller for the Smith predictor

        P or PI preferable, no derivative control
        """
        kp, ki, kd, offset = params
        integral, previous_error, setpoint = state
        error = setpoint - signal # Opposite to the other PID controller
        integral += error * dt
        derivative = (error - previous_error) / dt
        control_signal = kp * error + ki * integral + kd * derivative + offset
        # Ensure control signal is within reasonable bounds
        control_signal = jnp.clip(control_signal, 0.0, 10.0)
        new_state = (integral, error, setpoint)
        return control_signal, new_state
    def filter(u: float, filter_state: tuple):
        """
        EWMA filter for the Smith predictor
        Parameters:
        u: float
            input to be filtered
        filter_state: tuple
            (alpha, u_prev)
        Returns:
        u_filtered: float
            the filtered value
        filter_state: tuple
            (alpha, u_prev)
        """
        alpha, y_prev = filter_state
        u_filtered = alpha * u + (1 - alpha) * y_prev
        y_prev = u_filtered
        filter_state = (alpha, y_prev)
        return u_filtered, filter_state
