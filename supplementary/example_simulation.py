"""Example simulation script for ecological control simulation"""
import os
from math import ceil
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from closed_loop_coevolution.plant_model import Plant
from closed_loop_coevolution.simulation import Simulation
from closed_loop_coevolution.numerical_methods import newton_dynamical2
from closed_loop_coevolution.types import ControlSystemState, SimulationParams, PlantParams 
from closed_loop_coevolution.types import PlantState, DisturbanceParams, OptimisationParams
from closed_loop_coevolution.types import OptimisationConstraints, ControlSystemParams


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_disable_jit", True) # DEBUGGING
# Use JAX_PLATFORMS=cpu python model9.py if want to run on CPU
# jax.config.update("jax_debug_nans", True) # Detect NaNs


def main():
    """Example simulation script for ecological control simulation"""
    random_key = jax.random.key(0)

    # Steps to run
    CHECK_OBSERVABILITY = False
    RUN_SIMULATION = True
    SAVE_RESULTS = True
    LOAD_RESULTS = False
    PLOT_RESULTS = True
    PLOT_INNOVATIONS = False
    REDUCED_GRAPH = False
    PLOT_PARAMETERS = False
    PLOT_PARAMETERS_AND_DERIVATIVES = False
    SHOW_GRAPHS = True

    VERBOSE = True

    # Simulation hyperparameters
    NIS_WINDOW_SIZE = 10 # sample length, number of steps (not hours!)
    PREDICTION_HORIZON = 5 # sample length, number of steps (not hours!)
    REESTIMATE_WINDOW_SIZE = 21 # sample length, number of steps (not hours!)
    t0 = 0. # hours
    tf = 200. # hours
    MAXIMUM_DT_INTEGRATION = 0.01 # hours, based on previous trials
    STEP_SIZE = 0.1 # hours
    parameters_to_optimise = (True, False, False, False, False, True, False, False)
    optimisation_params = OptimisationParams(n_generations=16,
                                            population_size=100,
                                            mutation_rate=None,
                                            mutation_scale=None,
                                            mask=parameters_to_optimise,
                                            mask_indices=tuple(i for i, flag in enumerate(parameters_to_optimise) if flag))

    # Model parameters
    model_params_dict = {'B': 98, # burst size, number of new viruses released per infected cell
        'logdelta': np.log(4.0e-8), # virus absorption rate, 1/hour
        'tau': 0.5, # latent period of virus, hours
        'K': 4, # Monod constant, concentration of substrate at which DEPRECIATED
        'c': 100, # ug / mL, concentration of glucose in M9 minimal medium DEPRECIATED
        'muMax': 1.8, # 0.738, # Growth rate of bacteria, 1/hour
        'e':  1.92e-6, # removal rate of nutrients at maximal bacterial growth, ug / cell / hour DEPRECIATED
        'monodOn': 0, # Monod equation on (1) or off (0) DEPRECIATED
        'b2': 98, # burst size of phage2, number of new viruses released per infected cell
        'tau2': 0.5, # latent period of phage2, hours
        'logdelta2-1': np.log(3.5e-8), # adsorption rate of phage 2 to bacteria 1
        'logdelta2-2': np.log(3.5e-8), # adsorption rate of phage 2 to bacteria 2
        'logdelta1-2': np.log(3.12e-8), # adsorption rate of phage 1 to bacteria 2
        'mu_max2': 1.8 # Maximum growth rate of bacteria 2
        }
    model_params = jnp.array(list(model_params_dict.values()), dtype=jnp.float64)

    # Find equilibrium
    u = 1.25 # Setpoint flow rate
    susceptible_0 = 800000
    phage_0 = 1.8e8 # Initial guess, from solving a delay-free model
    y_hat_0 = jnp.array([susceptible_0, 0., 0., 0., 0., 0., phage_0])
    # Uses the tuned initial conditions
    y_hat_0, converged = newton_dynamical2(Plant.ode_model_one_species,
                                           0.,
                                           y_hat_0,
                                           u,
                                           jnp.zeros_like(y_hat_0),
                                           max_iter=10000,
                                           args=model_params,
                                           lr=0.1)
    y_hat_0 = jnp.array([y_hat_0])
    if not converged:
        print('Newton did not converge; using default values')
        y_hat_0 = jnp.array([[7.50502485e+05, 3.20144226e+04, 2.84572646e+04,
                              2.52953463e+04, 2.24847522e+04, 1.99864464e+04, 1.53812442e+07]])
    elif VERBOSE:
        print('y_eq:', y_hat_0)
        print('S0:', y_hat_0[0, 0])
        print('I_total:', np.sum(y_hat_0[0, 1:6]))
        print('P:', y_hat_0[0, 6])

    if CHECK_OBSERVABILITY:
        # Check LINEAR controllability and observability
        # This is NOT sufficient for NONLINEAR systems
        # Structural identifiability is checked with STRIKE-GOLDD
        print('Standard system')
        A = jax.jacfwd(Plant.ode_model_one_species, 1)(0.,
                                                     y_hat_0[0, :],
                                                     u, jnp.array([0.]),
                                                     model_params)
        print(f'A: {A}')
        B = jax.jacfwd(Plant.ode_model_one_species, 2)(0.,
                                                     y_hat_0[0, :],
                                                     u,
                                                     jnp.array([0.]),
                                                     model_params)
        print(f'B: {B}')
        C = jax.jacfwd(Plant.sensor_one_species, 1)(0.,
                                                   y_hat_0[0, :],
                                                   u,
                                                   jnp.array([0.]),
                                                   model_params)
        print(f'C: {C}')
        # Form the controllability matrix
        controllability_matrix = jnp.hstack([B, jnp.linalg.matrix_power(A, 1) @ B,
                                            jnp.linalg.matrix_power(A, 2) @ B,
                                            jnp.linalg.matrix_power(A, 3) @ B,
                                            jnp.linalg.matrix_power(A, 4) @ B,
                                            jnp.linalg.matrix_power(A, 5) @ B,
                                            jnp.linalg.matrix_power(A, 6) @ B])
        rank_of_controllability_matrix = jnp.linalg.matrix_rank(controllability_matrix)
        print("Rank of Controllability Matrix:", rank_of_controllability_matrix)
        # Form the observability matrix
        observability_matrix = jnp.vstack([C,
                                           C @ A,
                                           C @ jnp.linalg.matrix_power(A, 2),
                                           C @ jnp.linalg.matrix_power(A, 3),
                                           C @ jnp.linalg.matrix_power(A, 4),
                                           C @ jnp.linalg.matrix_power(A, 5),
                                           C @ jnp.linalg.matrix_power(A, 6)]).squeeze()
        rank_of_observability_matrix = jnp.linalg.matrix_rank(observability_matrix)
        print("Rank of Observability Matrix:", rank_of_observability_matrix)

        # Check controllability and observability of extended system
        print('Extended system')
        y_hat_extended = jnp.hstack([y_hat_0[0, :], model_params[0:7]])
        A = jax.jacfwd(Plant.ode_model_one_species_extended, 1)(0.,
                                                              y_hat_extended,
                                                              u,
                                                              jnp.array([0.]),
                                                              model_params)
        print(f'A: {A}')
        B = jax.jacfwd(Plant.ode_model_one_species_extended, 2)(0.,
                                                              y_hat_extended,
                                                              u,
                                                              jnp.array([0.]),
                                                              model_params)
        print(f'B: {B}')
        C = jax.jacfwd(Plant.sensor_one_species, 1)(0.,
                                                   y_hat_extended,
                                                   u,
                                                   jnp.array([0.]),
                                                   model_params)
        print(f'C: {C}')
        # Form the controllability matrix
        controllability_matrix = jnp.hstack([B,
                                            jnp.linalg.matrix_power(A, 1) @ B,
                                            jnp.linalg.matrix_power(A, 2) @ B,
                                            jnp.linalg.matrix_power(A, 3) @ B,
                                            jnp.linalg.matrix_power(A, 4) @ B,
                                            jnp.linalg.matrix_power(A, 5) @ B,
                                            jnp.linalg.matrix_power(A, 6) @ B])
        rank_of_controllability_matrix = jnp.linalg.matrix_rank(controllability_matrix)
        print("Rank of Controllability Matrix:", rank_of_controllability_matrix)
        # Form the observability matrix
        observability_matrix = jnp.vstack([C,
                                           C @ A,
                                           C @ jnp.linalg.matrix_power(A, 2),
                                           C @ jnp.linalg.matrix_power(A, 3),
                                           C @ jnp.linalg.matrix_power(A, 4),
                                           C @ jnp.linalg.matrix_power(A, 5),
                                           C @ jnp.linalg.matrix_power(A, 6)]).squeeze()
        rank_of_observability_matrix = jnp.linalg.matrix_rank(observability_matrix)
        print("Rank of Observability Matrix:", rank_of_observability_matrix)

    if RUN_SIMULATION:
        # Tuned initial conditions
        sff, pff = 1, 1 # constants to account for system needs to enter steady state.
        susceptible_0 = sff * y_hat_0[0,0]
        phage_0 = pff * y_hat_0[0,6]
        B_eq = float(susceptible_0 + y_hat_0[0, 1] + y_hat_0[0, 2] + y_hat_0[0, 3] + \
                    y_hat_0[0, 4] + y_hat_0[0, 5])
        # x uses tuned initial conditions
        y0 = jnp.array([susceptible_0, 0., 0., 0., 0., 0.,
                        phage_0, 0., 0., 0., 0., 0., 0., 0.])
        # x_hat uses the same tuned initial conditions
        y_hat_0 = jnp.array([susceptible_0, 0., 0., 0., 0., 0., phage_0])

        # Simulation parameters
        n_integration = int(ceil(STEP_SIZE / MAXIMUM_DT_INTEGRATION))
        dt_integration = STEP_SIZE / n_integration # hours, actual time step in integration
        n_steps = int(ceil((tf - t0)/STEP_SIZE)) # number of simulation steps

        # Reestimation window
        observation_history = jnp.zeros(REESTIMATE_WINDOW_SIZE)
        input_history = u * jnp.ones(REESTIMATE_WINDOW_SIZE)
        x_hat_history = jnp.tile(y_hat_0, (REESTIMATE_WINDOW_SIZE,1))

        # Error detection
        nis_history = jnp.zeros(NIS_WINDOW_SIZE)
        nis_half_significance = 0.0001 # on each tail, set to zero to deactivate MHE
        nis_avg_lb = 0. # stats.chi2.ppf(nis_half_significance, NIS_WINDOW_SIZE)
        nis_avg_ub = stats.chi2.ppf(1 - nis_half_significance, NIS_WINDOW_SIZE)
        # print(f'NIS bounds: {nis_avg_lb}, {nis_avg_ub}') # debugging
        # Matches HLT if (1.145, 11.07) for WINDOW_SIZE = 5 and half_alpha = 0.05
        # (0.4117, 16.75) for WINDOW_SIZE = 5 and half_alpha = 0.005

        # Observer parameters
        Q_0 = jnp.eye(y_hat_0.shape[0]) * 1e-6  # Process noise covariance, Small positive values on the diagonal
        Q_0 = Q_0.at[0, 0].set(1e6)
        # Q_0 = Q_0.at[6, 6].set(1e8)
        R_0 = jnp.array([[3.6e7]]) # Measurement noise covariance. R>0 to avoid singular matrix
        # p_0 = 1. # A positive scalar
        # P0 = p_0 * jnp.eye(y_hat_0.shape[0])
        P0 = jnp.diag(0.1 * jnp.abs(y_hat_0) + 1.0) # NOTE change from model8.py
        observer_params = (Q_0, R_0, P0) # Observer parameters: [0] = Q, [1] = R
        # Perfect parameter estimates NOTE different array size - must use Plant.sllos_dukf
        theta_hat_dict = {'B': 98, # burst size, number of new viruses released per infected cell
            'log_delta': jnp.log(2.9e-8), # virus absorption rate, 1/hour
            'tau': 0.5, # latent period of virus, hours
            'muMax': 1.8, # 0.738, # Growth rate of bacteria, 1/hour
            }
        model_params_hat = jnp.array(list(theta_hat_dict.values()), dtype=jnp.float64)
        # IF USING FILTERED NOISE TRANSITION:
        model_params_hat = jnp.concat([model_params_hat, jnp.zeros_like(model_params_hat)])  
        Q_param = jnp.eye(len(model_params_hat)) * 1e-10
        Q_param = Q_param.at[5, 5].set(1.0e-2)
        Q_param = Q_param.at[7, 7].set(1.0e-6)
        model_params_P_hat = 1.0e-10 * jnp.eye(len(model_params_hat))
        model_params_P_hat = model_params_P_hat.at[1, 1].set(1.0e-4)
        model_params_P_hat = model_params_P_hat.at[3, 3].set(1.0e-4)

        # Plant parameters
        # y_disturbance = jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # No evolution
        # y_disturbance = jnp.array([-5.0e3, 0., 0., 0., 0., 0., 0., 5.0e3, 0., 0., 0., 0., 0., 0.]) # Bacterial strains emerging
        y_disturbance = jnp.array([0., 0., 0., 0., 0., 0., -1.0e4, 0., 0., 0., 0., 0., 0., 1.0e4]) # Phage strain emerging
        disturbance_params = DisturbanceParams(jnp.array([10.]),
                                            jnp.array([y_disturbance]))
        # No emergence/disturbance
        # disturbance_params = DisturbanceParams(jnp.array([0.]),
        #                                     jnp.array([jnp.zeros_like(y0)]))
        Q_plant = jnp.block([[Q_0, jnp.zeros_like(Q_0)],
                     [jnp.zeros_like(Q_0), Q_0]])
        noise_params = (1.e3, Q_plant) # sensor, process noise standard deviation

        # Controller parameters
        # If using PID controller:
        # controller_params = (5.0e-8, 5.0e-8, 5.0e-8, u) # PID parameters (kp, ki, kd, offset)
        # controller_params = (0., 0., 0., u) # OPEN LOOP
        # controller_state = (0.0, 0.0, B_eq) # Initial integral, previous error, setpoint
        # IF USING SMITH:
        low_level_state = (0.0, 0.0, B_eq)
        smith_predictions = B_eq * jnp.ones(PREDICTION_HORIZON) # Make so burn-in doesn't affect
        filter_state = (0.9, 0.) # alpha, u_prev
        controller_state = (low_level_state, smith_predictions, filter_state)
        controller_params = (-3.0e-7, -1.0e-9, 0., u) # PID parameters for SMITH (kp, ki, kd, offset) USED FOR ALL SIMULATIONS SO FAR
        controller_params = (-5.0e-7, -1.0e-9, 0., u) # PID parameters for SMITH (kp, ki, kd, offset)
        # controller_params = (-2.0e-7, 0., 0., u) # no integral action (Testing if better)
        # controller_params = (0., 0., 0., u) # OPEN LOOP

        # Package parameters and states
        simulation_params = SimulationParams(STEP_SIZE=STEP_SIZE,
                                             MAXIMUM_DT_INTEGRATION=MAXIMUM_DT_INTEGRATION,
                                             dt_integration=dt_integration,
                                             n_integration=n_integration,
                                             n_steps=n_steps,
                                             t0=t0,
                                             tf=tf,
                                             reestimate_window_size=REESTIMATE_WINDOW_SIZE,
                                             prediction_horizon=PREDICTION_HORIZON,
                                             optimisation_params=optimisation_params,
                                             dual_inner_loops=10, # Must be less than reestimate_window_size
                                             param_transition_magic_number=0.0001) 
        opt_constraints = OptimisationConstraints(minval=jnp.array([1.0, np.log(1e-12), 0.1, 0.1, 90., 0.1, 1e-7, 0.0]),
                                                maxval=jnp.array([300.0, np.log(1e-6), 2.0, 20.0, 110., 2.0, 1e-5, 1.0]),
                                                replacement_params=jnp.array([98, np.log(3.12e-8), 0.5, 4, 100, 1.8, 1.92e-6, 1]))
        control_system_params = ControlSystemParams(controller_params=controller_params,
                                                    observer_params=observer_params,
                                                    Q_k =Q_0,
                                                    R_k = R_0,
                                                    nis_avg_lb=nis_avg_lb,
                                                    nis_avg_ub=nis_avg_ub,
                                                    opt_constraints=opt_constraints,
                                                    ukf_params=(1., 0., -1.), # alpha, beta, kappa OR (1e-3, 2.0, 0.0),
                                                    Q_param=Q_param,
                                                    R_param=R_0)
        plant_params = PlantParams(model_params,
                                   disturbance_params,
                                   noise_params)
        control_system_state = ControlSystemState(x_hat=y_hat_0,
                                                P=P0,
                                                nis_history=nis_history,
                                                model_params_hat=model_params_hat,
                                                model_params_P_hat=model_params_P_hat,
                                                z_history=observation_history,
                                                u_history=input_history,
                                                x_hat_history=x_hat_history,
                                                controller_state=controller_state,
                                                mhe_activation=0,
                                                dual_counter=1)
        plant_state = PlantState(y0)

        state_history = Simulation.run_simulation(plant_state,
                                           control_system_state,
                                           simulation_params,
                                           control_system_params,
                                           plant_params,
                                           random_key,
                                           verbose=True)
        x_history, x_hat_history, z_history, nis_history, u_history, param_history = state_history

        # Move to numpy
        x_plot = np.array(x_history)
        x_hat_plot = np.array(x_hat_history)
        z_plot = np.array(z_history)
        nis_plot = np.array(nis_history)
        u_plot = np.array(u_history)
        param_plot = np.array(param_history)
        t_plot = np.linspace(t0, tf, n_steps)

        # Downsample for plotting efficiency
        n_points_plot = float('inf') # do not downsample
        if n_steps > n_points_plot:
            t_plot = t_plot[::int(n_steps / n_points_plot)]
            x_plot = x_plot[::int(n_steps/ n_points_plot), :]
            x_hat_plot = x_hat_plot[::int(n_steps / n_points_plot), :]
            z_plot = z_plot[::int(n_steps / n_points_plot)]
            nis_plot = nis_plot[::int(n_steps / n_points_plot)]
            u_plot = u_plot[::int(n_steps / n_points_plot)]
            param_plot = param_plot[::int(n_steps / n_points_plot), :]

    if SAVE_RESULTS:
        # Save the results
        os.makedirs('supplementary/outputs', exist_ok=True)
        np.savez('supplementary/outputs/ode_model9_main.npz',
                t_plot=t_plot,
                x_plot=x_plot,
                x_hat_plot=x_hat_plot,
                z_plot=z_plot,
                nis_plot=nis_plot,
                u_plot=u_plot,
                param_plot=param_plot)

    if LOAD_RESULTS:
        # Load the results
        data = np.load('supplementary/outputs/ode_model9_main.npz')
        t_plot = data['t_plot']
        x_plot = data['x_plot']
        x_hat_plot = data['x_hat_plot']
        z_plot = data['z_plot']
        nis_plot = data['nis_plot']
        u_plot = data['u_plot']
        param_plot = data['param_plot']

    if PLOT_RESULTS:
        # PLOTTING SETUP
        plt.style.use('seaborn-v0_8-colorblind') # set the default plot style
        # plt.rcParams.update({"font.family": "Latin Modern"})
        plt.rcParams.update({'font.size': 8})
        # enable LaTeX rendering for matplotlib text - very slow, but nice
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        })
        plt.rcParams.update({'lines.linewidth': 2})
        plt.rcParams.update({'grid.alpha': 0.5, 'grid.linestyle': '-', 'grid.linewidth': 0.5})
        # plt.rcParams.update({'axes.grid': True})
        # name colors
        sns_blue, sns_green, sns_orange, sns_pink, sns_yellow, sns_lightblue = \
            ('#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9')
        color_bacteria = sns_blue
        color_phage = '#ff724c'
        color_od600 = '#e50000'
        color_nutrient = '#FFD700'
        color_line = '#000000'
        color_inf0, color_inf1, color_inf2, color_inf3, color_inf4 = \
        ('#b2e2e2', '#a1d99b', '#74c476', '#31a354', '#006d2c')

        # Plot the solution
        cm = 1/2.54  # centimeters in inches
        fig, axs = plt.subplots(3, 1, figsize=(18*cm, 14.4*cm), gridspec_kw={'height_ratios': [2, 2, 1]})

        axs[0].plot(t_plot, x_plot[:, 0], c=color_bacteria, label='$S$')
        axs[0].plot(t_plot, x_plot[:, 7], c="#c79fef", label='$S2$') # If plotting emergent strain
        infected_total = x_plot[:, 1] + x_plot[:, 2] + x_plot[:, 3] + x_plot[:, 4] + x_plot[:, 5]
        axs[0].plot(t_plot, infected_total, c=color_inf4, label='$I$')
        axs[0].set_ylabel('Concentration [1/mL]')
        ax0_twin = axs[0].twinx()
        ax0_twin.plot(t_plot, x_plot[:, 6], color=color_phage, label='Phage')
        ax0_twin.plot(t_plot, x_plot[:, 13], color="#ffb3a7", label='Phage2') # If plotting emergent phage
        ax0_twin.set_ylabel('Phage Concentration [1/mL]', color=color_phage)
        ax0_twin.tick_params(axis='y', labelcolor=color_phage)
        axs[0].legend(loc='upper left')
        axs[0].set_title('Plant')

        axs[1].plot(t_plot, x_hat_plot[:, 0], c=color_bacteria, label='$S$')
        infected_total_hat = x_hat_plot[:, 1] + x_hat_plot[:, 2] + x_hat_plot[:, 3] + x_hat_plot[:, 4] + x_hat_plot[:, 5]
        axs[1].plot(t_plot, infected_total_hat, c=color_inf4, label='$I$')
        axs[1].set_ylabel('Concentration [1/mL]')
        ax1_twin = axs[1].twinx()
        ax1_twin.plot(t_plot, x_hat_plot[:, 6], color=color_phage, label='Phage')
        ax1_twin.set_ylabel('Phage Concentration [1/mL]', color=color_phage)
        ax1_twin.tick_params(axis='y', labelcolor=color_phage)
        # axs[1].legend(loc='right')
        axs[1].set_title('Observer')

        axs[2].plot(t_plot, u_plot, color='tab:red', label='Control Signal')
        axs[2].set_ylabel('Control Signal', color='tab:red')
        axs[2].tick_params(axis='y', labelcolor='tab:red')
        axs[2].set_xlabel('Time / hours')

        plt.tight_layout()
        os.makedirs('supplementary/outputs', exist_ok=True)
        plt.savefig('supplementary/outputs/ode_model9_main.png', dpi=500)
        if SHOW_GRAPHS:
            plt.show()

    if PLOT_INNOVATIONS:
        # Innovations plot
        bacteria_all = x_plot[:, 0] + x_plot[:, 1] + x_plot[:, 2] + x_plot[:, 3] + \
            x_plot[:, 4] + x_plot[:, 5] + x_plot[:, 7] + x_plot[:, 8] + x_plot[:, 9] + \
            x_plot[:, 10] + x_plot[:, 11] + x_plot[:, 12]
        bacteria_all_hat = x_hat_plot[:, 0] + x_hat_plot[:, 1] + x_hat_plot[:, 2] + x_hat_plot[:, 3] + \
            x_hat_plot[:, 4] + x_hat_plot[:, 5]
        fig, axs = plt.subplots(2, 1, figsize=(8.4*cm, 7*cm))
        axs[0].plot(t_plot, bacteria_all, color='black', linewidth=0.8, label='Plant')
        axs[0].plot(t_plot, bacteria_all_hat, color=sns_blue, linestyle='--', linewidth=0.8, label='Observer')
        axs[0].plot(t_plot, z_plot, ':', color=color_od600, label='Measurement')
        axs[0].set_xticklabels([])
        axs[0].set_ylabel('Total bacteria [1/mL]')
        axs[0].legend()

        phage_all = x_plot[:, 6] + x_plot[:, 13]
        axs[1].plot(t_plot, phage_all, color='black', linewidth=0.8, label='Plant')
        axs[1].plot(t_plot, x_hat_plot[:, 6], color=sns_blue, linestyle='--', linewidth=0.8, label='Observer')
        axs[1].set_ylabel('Phage [1/mL]')
        axs[1].set_xlabel('Time / hours')

        fig.tight_layout()
        plt.savefig('supplementary/outputs/ode_model9_innovations.png', dpi=500)
        if SHOW_GRAPHS:
            plt.show()

    if REDUCED_GRAPH:
        # Plot the solution
        cm = 1/2.54  # centimeters in inches
        fig, axs = plt.subplots(2, 1, figsize=(18*cm, 10*cm), gridspec_kw={'height_ratios': [2, 1]})

        axs[0].plot(t_plot, x_plot[:, 0], c=color_bacteria, label='$S$')
        # axs[0].plot(t_plot, x_plot[:, 7], c="#c79fef", label='$S2$')
        infected_total = x_plot[:, 1] + x_plot[:, 2] + x_plot[:, 3] + x_plot[:, 4] + x_plot[:, 5]
        axs[0].plot(t_plot, infected_total, c=color_inf4, label='$I$')
        axs[0].set_ylabel('Concentration [1/mL]')
        ax0_twin = axs[0].twinx()
        ax0_twin.plot(t_plot, x_plot[:, 6], color=color_phage, label='P')
        ax0_twin.set_ylabel('Phage Concentration [1/mL]', color=color_phage)
        ax0_twin.tick_params(axis='y', labelcolor=color_phage)
        axs[0].set_title('Plant')
        leg = axs[0].legend(loc='upper left') 
        leg.remove() # disgusting workaround; matplotlib is hierarchical and there may be no elegant
        ax0_twin.add_artist(leg) # solution: https://github.com/matplotlib/matplotlib/issues/3706
        
        axs[1].plot(t_plot, u_plot, color='tab:red', label='Control Signal')
        axs[1].set_ylabel('Control Signal', color='tab:red')
        axs[1].tick_params(axis='y', labelcolor='tab:red')
        axs[1].set_xlabel('Time / hours')

        plt.tight_layout()
        plt.savefig('supplementary/outputs/ode_model9_reduced.png', dpi=500)
        if SHOW_GRAPHS:
            plt.show()

    if PLOT_PARAMETERS:
        # Plot parameters as proportion of their initial value
        param_plot_np = np.asarray(param_plot)
        if param_plot_np.ndim == 1:
            param_plot_np = param_plot_np[:, None]

        initial = param_plot_np[0, :].astype(float)
        # avoid division by zero
        initial_safe = np.where(initial == 0.0, 1.0, initial)
        prop = param_plot_np / initial_safe[None, :]

        n_params = 4 # Hardcoded to only plot parameters, not derivatives
        labels = ['B', r'$\delta$', r'$\tau$', r'$\mu_{max}$']

        fig, ax = plt.subplots(figsize=(8.4*cm, 6*cm))
        for i in range(n_params):
            if i == 1:
                # Move to delta space
                delta_prop = np.exp(param_plot_np[:, 1]) / np.exp(initial_safe[1])
                ax.plot(t_plot, delta_prop, label=labels[i])
            else:
                ax.plot(t_plot, prop[:, i], label=labels[i])
        ax.axhline(1.0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Time / hours')
        ax.set_ylabel(r'$\hat{\theta}/\theta_{0}$')
        # ax.set_title('Parameter trajectories (normalized)')
        ax.legend(loc='best', fontsize='small', ncol=min(2, n_params))
        plt.tight_layout()
        plt.savefig('supplementary/outputs/ode_model9_params_normalized.png', dpi=500)
        if SHOW_GRAPHS:
            plt.show()

    if PLOT_PARAMETERS_AND_DERIVATIVES:
        # Plot parameters as proportion of their initial value
        param_plot_np = np.asarray(param_plot)
        if param_plot_np.ndim == 1:
            param_plot_np = param_plot_np[:, None]

        initial = param_plot_np[0, :].astype(float)
        # avoid division by zero
        initial_safe = np.where(initial == 0.0, 1.0, initial)
        prop = param_plot_np / initial_safe[None, :]

        n_params = prop.shape[1]       
        labels = ['B', r'$\delta$', r'$\tau$', r'$\mu_{max}$', r'$\frac{d}{dt}B$', r'$\frac{d}{dt}log\delta$', r'$\frac{d}{dt}\tau$', r'$\frac{d}{dt}\mu_{max}$']

        fig, ax = plt.subplots(figsize=(8.4*cm, 6*cm))
        for i in range(n_params):
            if i == 1:
                # Move to delta space
                delta_prop = np.exp(param_plot_np[:, 1]) / np.exp(initial_safe[1])
                ax.plot(t_plot, delta_prop, label=labels[i])
            else:
                ax.plot(t_plot, prop[:, i], label=labels[i])
        ax.axhline(1.0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Time / hours')
        ax.set_ylabel(r'$\hat{\theta}/\theta_{0}$')
        # ax.set_title('Parameter trajectories (normalized)')
        ax.legend(loc='best', fontsize='small', ncol=min(2, n_params))
        plt.tight_layout()
        plt.savefig('supplementary/outputs/ode_model9_params_derivatives_normalized.png', dpi=500)
        if SHOW_GRAPHS:
            plt.show()
    


if __name__ == "__main__":
    main()
