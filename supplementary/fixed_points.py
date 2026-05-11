"""Explore stability of nonlinear system"""
import os
import numpy as np
import jax
import matplotlib as mpl
import jax.numpy as jnp
import matplotlib.pyplot as plt

def newton(func, x_0, u_input, tol=1e-6, max_iter=1000, args=(), lr: float=1.0):
    """
    Newton's method for solving systems of nonlinear equations
    Function of time with additional arguments u_input and args
    Args:
    func: function to solve, with arguments (y, u, args)
    x_0: initial estimate; must be float array, not integer array
    tol: tolerance
    max_iter: maximum number of iterations
    args: additional arguments to pass to func
    Returns:
    y: solution
    converged: boolean indicating whether the method converged
    error: final error (norm of difference between last two iterations)
    n: number of iterations
    """
    x_state = x_0
    n = 0
    error = tol+1
    converged = False
    f_prime = jax.jacfwd(lambda x_state, u_input, args: func(x_state, u_input, args), argnums=0)

    def loop_body(state):
        x_state, u_input, error, max_iter, n, converged, args = state
        x_new = x_state - lr * jnp.linalg.solve(f_prime(x_state, u_input, args), func(x_state, u_input, args))
        error = jnp.linalg.norm(x_new - x_state)
        converged = jax.lax.cond(error <= tol, lambda _: True, lambda _: converged, None)
        return x_new, u_input, error, max_iter, n + 1, converged, args

    def cond_fun(state):
        x_state, u_input, error, max_iter, n, converged, args = state
        return jnp.logical_and(error > tol, n < max_iter)

    x_state, _, error, _, n, converged, _ = jax.lax.while_loop(cond_fun,
                                                               loop_body,
                                                               (x_state, u_input, error, max_iter, n, converged, args))

    return x_state, converged, error, n

def newton_general_form(func, x_0, tol=1e-6, max_iter=1000, args=(), lr: float=1.0):
    """
    Newton's method for solving systems of nonlinear equations
    Function of time with additional arguments u_input and args
    Args:
    func: function to solve, with arguments (y, u, args)
    x_0: initial estimate; must be float array, not integer array
    tol: tolerance
    max_iter: maximum number of iterations
    args: additional arguments to pass to func
    Returns:
    y: solution
    converged: boolean indicating whether the method converged
    error: final error (norm of difference between last two iterations)
    n: number of iterations
    """
    x_state = x_0
    n = 0
    error = tol+1
    converged = False
    f_prime = jax.jacfwd(func)

    def loop_body(state):
        x_state, error, max_iter, n, converged, args = state
        x_new = x_state - lr * jnp.linalg.solve(f_prime(x_state), func(x_state))
        error = jnp.linalg.norm(x_new - x_state)
        converged = jax.lax.cond(error <= tol, lambda _: True, lambda _: converged, None)
        return x_new, error, max_iter, n + 1, converged, args

    def cond_fun(state):
        x_state, error, max_iter, n, converged, args = state
        return jnp.logical_and(error > tol, n < max_iter)

    x_state, error, _, n, converged, _ = jax.lax.while_loop(cond_fun,
                                                            loop_body,
                                                            (x_state, error, max_iter, n, converged, args))

    return x_state, converged, error, n
    

def sllos_bif(x_state, u_input, theta_params) -> jnp.array:
    """
    Simplified Levin-Lenski Dynamics for Bifurcation Analysis
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
    control_signal = u_input
    # Ensure no negative values in y
    x_state = jnp.maximum(x_state, 0.)
    delta_clipped = jnp.clip(jnp.exp(theta_params[1]), 1e-12, 1e-6)
    # delta_clipped = jnp.exp(model_params[1]) # OPTION TO REMOVE DELTA CLIPPING

    S1, I11, I12, I13, I14, I15, P1  = x_state

    dS1 = theta_params[3] * S1 - delta_clipped * S1 * P1 \
                                - control_signal * S1
    dI11 = delta_clipped * S1 * P1   - M / theta_params[2] * I11 \
        - control_signal * I11
    dI12 = M / theta_params[2] * (I11 - I12) - control_signal * I12
    dI13 = M / theta_params[2] * (I12 - I13) - control_signal * I13
    dI14 = M / theta_params[2] * (I13 - I14) - control_signal * I14
    dI15 = M / theta_params[2] * (I14 - I15) - control_signal * I15
    dP1 = theta_params[0] * M / theta_params[2] * I15 \
        - delta_clipped * S1 * P1 - control_signal * P1

    dy = jnp.array([dS1, dI11, dI12, dI13, dI14, dI15, dP1])
    return dy

def sllos_cl_bif(x_state, u_probe, args) -> jnp.array:
    """Closed loop version of sllos_bif with simple proportional control
    to track x_target"""
    x_target, theta_params, p_control = args
    total_bacteria_target = jnp.sum(x_target[:-1])
    total_bacteria = jnp.sum(x_state[:-1])
    u_input = jnp.clip(u_probe + p_control * (total_bacteria - total_bacteria_target), 0., None)
    return sllos_bif(x_state, u_input, theta_params)

def d_sllos_bif_dx(x_state, u_input, theta_params):
    """
    Jacobian of sllos_bif with respect to state variables x_state
    """
    return jax.jacfwd(sllos_bif, argnums=0)(x_state, u_input, theta_params)

def d_sllos_cl_bif_dx(x_state, u_probe, args):
    """
    Jacobian of sllos_cl_bif with respect to state variables x_state
    """
    return jax.jacfwd(sllos_cl_bif, argnums=0)(x_state, u_probe, args)

def find_steady_state():
    theta_params = jnp.array([100, np.log(3.12e-8), 0.5, 1.8])  # Example parameters
    u_input = 1.25 # Setpoint flow rate
    x_0_guess = jnp.array([800000, 0., 0., 0., 0., 0., 1.8e7]) # Initial state guess
    x_steady_state, converged, error, n = newton(sllos_bif,
                                x_0_guess,
                                u_input,
                                tol=1e-6,
                                max_iter=10000,
                                args=theta_params,
                                lr=0.1)
    if converged:
        print("Converged to steady state:", x_steady_state, "in", n, "iterations with error", error)
    else:
        print("Did not converge in ", n, "iterations. Final estimate:", x_steady_state, "with error", error)

    # recompute Jacobian at the found steady state
    dfdx = d_sllos_bif_dx(x_steady_state, u_input, theta_params)
    eigvals, eigvecs = jnp.linalg.eig(dfdx)
    eigvals_np = np.asarray(eigvals)
    eigvecs_np = np.asarray(eigvecs)
    max_real = np.max(np.real(eigvals_np))
    print("dfdx:\n", np.asarray(dfdx))
    print("Eigenvalues:\n", eigvals_np)
    print("Max Re(eigenvalue):", max_real)
    print("Eigenvectors (columns):\n", eigvecs_np)
    
    # Plot eigenvalues in the complex plane (real vs imaginary parts)
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(eigvals_np), np.imag(eigvals_np), c='tab:purple', edgecolors='k')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    for i, ev in enumerate(eigvals_np):
        plt.annotate(str(i), (np.real(ev), np.imag(ev)), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel('Re(λ)')
    plt.ylabel('Im(λ)')
    plt.title('Eigenvalues in the Complex Plane')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig("supplementary/outputs/eigenvalues_complex_plane.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Create two horizontal subplots: real parts on the left, imaginary on the right
    fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    im_real = ax_real.imshow(np.real(eigvecs_np), aspect='auto', cmap='bwr', interpolation='nearest')
    ax_real.set_title('Real components of eigenvectors')
    ax_real.set_xlabel('Eigenvector index')
    ax_real.set_ylabel('State index')
    fig.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04, label='Real part')

    im_imag = ax_imag.imshow(np.imag(eigvecs_np), aspect='auto', cmap='bwr', interpolation='nearest')
    ax_imag.set_title('Imaginary components of eigenvectors')
    ax_imag.set_xlabel('Eigenvector index')
    # sharey=True ensures state index ticks are shown only on left subplot
    fig.colorbar(im_imag, ax=ax_imag, fraction=0.046, pad=0.04, label='Imag part')

    plt.xticks(np.arange(eigvecs_np.shape[1]))
    plt.yticks(np.arange(eigvecs_np.shape[0]))
    plt.tight_layout()
    plt.savefig("supplementary/outputs/eigenvectors_real.png", dpi=300, bbox_inches="tight")
    plt.show()

def check_stability_cl():    
    theta_params = jnp.array([100, np.log(3.12e-8), 0.5, 1.8])  # Example parameters
    u_input = 1.25 # Setpoint flow rate
    x_0_guess = jnp.array([800000, 0., 0., 0., 0., 0., 1.8e7]) # Initial state guess
    p_control = 2.0e-7 # note opposite sign to model9.py

    x_steady_state, converged, error, n = newton(sllos_bif,
                                x_0_guess,
                                u_input,
                                tol=1e-6,
                                max_iter=10000,
                                args=theta_params,
                                lr=0.1)

    if converged:
        print("Converged to steady state:", x_steady_state, "in", n, "iterations with error", error)
    else:
        print("Did not converge in ", n, "iterations. Final estimate:", x_steady_state, "with error", error)

    # recompute Jacobian at the found steady state
    dfdx = d_sllos_cl_bif_dx(x_steady_state, u_input, (x_steady_state, theta_params, p_control))
    eigvals, eigvecs = jnp.linalg.eig(dfdx)
    eigvals_np = np.asarray(eigvals)
    eigvecs_np = np.asarray(eigvecs)
    max_real = np.max(np.real(eigvals_np))
    print("dfdx:\n", np.asarray(dfdx))
    print("Eigenvalues:\n", eigvals_np)
    print("Max Re(eigenvalue):", max_real)
    print("Eigenvectors (columns):\n", eigvecs_np)
    
    # Plot eigenvalues in the complex plane (real vs imaginary parts)
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(eigvals_np), np.imag(eigvals_np), c='tab:purple', edgecolors='k')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    for i, ev in enumerate(eigvals_np):
        plt.annotate(str(i), (np.real(ev), np.imag(ev)), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel('Re(λ)')
    plt.ylabel('Im(λ)')
    plt.title('Eigenvalues in the Complex Plane')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig("supplementary/outputs/eigenvalues_complex_plane.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Create two horizontal subplots: real parts on the left, imaginary on the right
    fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    im_real = ax_real.imshow(np.real(eigvecs_np), aspect='auto', cmap='bwr', interpolation='nearest')
    ax_real.set_title('Real components of eigenvectors')
    ax_real.set_xlabel('Eigenvector index')
    ax_real.set_ylabel('State index')
    fig.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04, label='Real part')

    im_imag = ax_imag.imshow(np.imag(eigvecs_np), aspect='auto', cmap='bwr', interpolation='nearest')
    ax_imag.set_title('Imaginary components of eigenvectors')
    ax_imag.set_xlabel('Eigenvector index')
    # sharey=True ensures state index ticks are shown only on left subplot
    fig.colorbar(im_imag, ax=ax_imag, fraction=0.046, pad=0.04, label='Imag part')

    plt.xticks(np.arange(eigvecs_np.shape[1]))
    plt.yticks(np.arange(eigvecs_np.shape[0]))
    plt.tight_layout()
    plt.savefig("supplementary/outputs/eigenvectors_real.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_equilibrium_curve_ol():
    theta_params = jnp.array([100, np.log(3.12e-8), 0.5, 1.8])  # Example parameters
    x_0_guess = jnp.array([800000, 0., 0., 0., 0., 0., 1.8e7]) # Initial state guess
    us_values = jnp.linspace(0.8, 1.8, 100)

    def scan_step(carry, u):
        x_0_guess = carry
        y_hat, converged, error, n = newton(sllos_bif,
                                    x_0_guess,
                                    u,
                                    tol=1e-6,
                                    max_iter=100000,
                                    args=theta_params,
                                    lr=0.1)
        # steady_state = jnp.where(converged, y_hat, jnp.full_like(x_0_guess, 0.)) # Set to zero if not converged
        steady_state = y_hat  # Keep last estimate even if not converged
        
        dfdx = d_sllos_bif_dx(steady_state, u, theta_params)
        eigvals = jnp.linalg.eigvals(dfdx)
        max_eigenvalue = jnp.max(jnp.real(eigvals))
        return y_hat, (steady_state, max_eigenvalue)
    
    _, results = jax.lax.scan(scan_step, x_0_guess, us_values)
    steady_states, max_eigenvalues = results

    # Convert to numpy for plotting
    us_np = np.asarray(us_values)
    steady_state_np = np.asarray(steady_states)
    max_eigenvalues_np = np.asarray(max_eigenvalues)

    # Create two vertical subplots and override their plot methods to show stable as circles and unstable as crosses
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    # Susceptible bacteria on first subplot
    ln1 = ax1.plot(us_values, steady_state_np[:, 0], color='tab:blue', label='Susceptible Bacteria')
    ax1.set_ylabel('Susceptible Bacteria', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid()

    # Phage on second subplot
    ln2 = ax2.plot(us_values, steady_state_np[:, -1], color='tab:orange', label='Phage')
    ax2.set_ylabel('Phage', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.grid()

    ax3.plot(us_np, max_eigenvalues_np, color='tab:red', linestyle='-',
                label='Max Re(λ)')
    ax3.set_ylabel('Max Re(λ)')
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax3.grid()
    
    # Common x-label
    ax3.set_xlabel('Turnover Rate (u)')
    fig.tight_layout()
    # Save figure to file (PNG, 500 dpi)
    fig.savefig("supplementary/outputs/equilibrium_curve.png", dpi=500, bbox_inches="tight")
    plt.show()

def plot_equilibrium_curve_cl():
    theta_params = jnp.array([100, np.log(3.12e-8), 0.5, 1.8])  # Example parameters
    x_0_guess = jnp.array([800000, 0., 0., 0., 0., 0., 1.8e7]) # Initial state guess
    us_values = jnp.linspace(0.8, 1.8, 10)
    p_control = 2.0e-7 # note opposite sign to model9.py

    def scan_step(carry, u):
        x_0_guess = carry
        y_ol_hat, converged, error, n = newton(sllos_bif,
                                    x_0_guess,
                                    u,
                                    tol=1e-6,
                                    max_iter=100000,
                                    args=theta_params,
                                    lr=0.1)

        # steady_state = jnp.where(converged, y_hat, jnp.full_like(x_0_guess, 0.)) # Set to zero if not converged

        dfdx = d_sllos_cl_bif_dx(y_ol_hat, u, (y_ol_hat, theta_params, p_control))
        eigvals = jnp.linalg.eigvals(dfdx)
        max_eigenvalue = jnp.max(jnp.real(eigvals))
        return y_ol_hat, (y_ol_hat, max_eigenvalue)
    
    _, results = jax.lax.scan(scan_step, x_0_guess, us_values)
    steady_states, max_eigenvalues = results

    # Convert to numpy for plotting
    us_np = np.asarray(us_values)
    steady_state_np = np.asarray(steady_states)
    max_eigenvalues_np = np.asarray(max_eigenvalues)

    # Create two vertical subplots and override their plot methods to show stable as circles and unstable as crosses
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    # Susceptible bacteria on first subplot
    ln1 = ax1.plot(us_values, steady_state_np[:, 0], color='tab:blue', label='Susceptible Bacteria')
    ax1.set_ylabel('Susceptible Bacteria', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid()

    # Phage on second subplot
    ln2 = ax2.plot(us_values, steady_state_np[:, -1], color='tab:orange', label='Phage')
    ax2.set_ylabel('Phage', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.grid()

    ax3.plot(us_np, max_eigenvalues_np, color='tab:red', linestyle='-',
                label='Max Re(λ)')
    ax3.set_ylabel('Max Re(λ)')
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax3.grid()
    
    # Common x-label
    ax3.set_xlabel('Turnover Rate (u)')
    fig.tight_layout()
    # Save figure to file (PNG, 500 dpi)
    fig.savefig("supplementary/outputs/equilibrium_curve_cl.png", dpi=500, bbox_inches="tight")
    plt.show()

def plot_vector_field():
    """Plot vector field for sllos_bif"""
    u_input = 1.25
    theta_params = jnp.array([100, np.log(3.12e-8), 0.5, 1.8])  # Example parameters
    # Define the range of S and P values
    S_max = 1e8
    S_min = 0
    P_max = 1e8
    P_min = 0
    points = 50
    S_range = jnp.linspace(S_min, S_max, points)  # Susceptible bacteria
    P_range = jnp.linspace(P_min, P_max, points)  # Phage

    # Create a meshgrid for S and P
    S, P = jnp.meshgrid(S_range, P_range)

    # Initialize the vector field
    dS = jnp.zeros_like(S)
    dP = jnp.zeros_like(P)

    # Compute the vector field
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            x_state = jnp.array([S[i, j], 0., 0., 0., 0., 0., P[i, j]])
            dx = sllos_bif(x_state, u_input, theta_params)
            dS = dS.at[i, j].set(dx[0])
            dP = dP.at[i, j].set(dx[-1])

    # Normalize the vectors for better visualization
    magnitude = jnp.sqrt(dS**2 + dP**2 +1e-10)
    max_dS = jnp.max(jnp.abs(dS))
    max_dP = jnp.max(jnp.abs(dP))
    scale_S = (S_max - S_min)/points / max_dS
    scale_P = (P_max - P_min)/points / max_dP
    dS_scaled = dS * scale_S
    dP_scaled = dP * scale_P
    dS_normalized = dS / magnitude
    dP_normalized = dP / magnitude

    # Plot the vector field
    plt.figure(figsize=(8, 6))
    plt.quiver(S, P, dS_scaled, dP_scaled, angles='xy', scale_units='xy', scale=1, color='tab:blue')
    plt.xlabel('Susceptible Bacteria (S)')
    plt.ylabel('Phage (P)')
    plt.title('Vector Field for sllos_bif')
    plt.tight_layout()
    plt.savefig("supplementary/outputs/vector_field_sllos_bif.png", dpi=300, bbox_inches="tight")
    plt.show()

def find_u_given_s():
    
    def func_to_solve(u_input):
        S = 2e7  # Desired susceptible bacteria level
        B = 98
        delta = 3.12e-8
        tau = 0.5
        return B * jnp.exp(- tau * u_input) - 1/(S*delta) * u_input - 1

    u_0_guess = jnp.array([1.0])  # Initial guess for u
    u_solution, converged, error, n = newton_general_form(func_to_solve,
                                            u_0_guess,
                                            tol=1e-6,
                                            max_iter=1000,
                                            args=(),
                                            lr=0.1)
    if converged:
        print("Converged to u:", u_solution, "in", n, "iterations with error", error)
    else:
        print("Did not converge in ", n, "iterations. Final estimate:", u_solution, "with error", error)

    S = 1e7  # Desired susceptible bacteria level
    B = 98
    mu_max = 1.8
    delta = 3.12e-8
    tau = 0.5
    p_steady = (mu_max - u_solution[0]) / (delta)
    i_steady = delta * S * p_steady * (1 - jnp.exp(- tau * u_solution[0])) / (u_solution[0])
    print("Steady state values at found u:")
    print("Phage P:", p_steady)
    print("Infected I:", i_steady)

    s_limit = mu_max / (delta * (B * jnp.exp(- tau * mu_max) - 1))
    print("Theoretical limit on S for existence:", s_limit)

def find_u_bar():
    def func_to_solve(u_input):
        B = 98
        tau = 0.5
        return (u_input * tau + 1) * B * jnp.exp(- tau * u_input) - 1
    
    u_0_guess = jnp.array([1.0])  # Initial guess for u
    u_solution, converged, error, n = newton_general_form(func_to_solve,
                                            u_0_guess,
                                            tol=1e-6,
                                            max_iter=1000,
                                            args=(),
                                            lr=0.1)
    
    if converged:
        print("Converged to u_bar:", u_solution, "in", n, "iterations with error", error)
    else:
        print("Did not converge in ", n, "iterations. Final estimate:", u_solution, "with error", error)

def main():
    os.makedirs('supplementary/outputs', exist_ok=True)

    find_steady_state()
    check_stability_cl()
    plot_equilibrium_curve_ol()
    # plot_equilibrium_curve_cl()
    # plot_vector_field()
    # find_u_given_s()
    # find_u_bar()

if __name__ == "__main__":
    main()
