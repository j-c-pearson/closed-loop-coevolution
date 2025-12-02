"""Basic numerical methods for closed-loop coevolution simulations"""
import jax
import jax.numpy as jnp

# Runge-Kutta methods for solving ODEs
# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
# Also consider Runge-Kutta-Fehlberg method and Dormand-Prince method

def reRK4Step(func, t, y, h, args):
    k1 = h*func(t, y, *args)
    k2 = h*func(t + 0.5*h, y + 0.5*k1, *args)
    k3 = h*func(t + 0.5*h, y + 0.5*k2, *args)
    k4 = h*func(t + h, y + k3, *args)
    y = y + (k1 + 2*k2 + 2*k3 + k4)/6
    return jnp.maximum(y, 0.)  # Ensure non-negative values

def rk4Step(func, t, y, h, args):
    k1 = h*func(t, y, *args)
    k2 = h*func(t + 0.5*h, y + 0.5*k1, *args)
    k3 = h*func(t + 0.5*h, y + 0.5*k2, *args)
    k4 = h*func(t + h, y + k3, *args)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

def rk4(func, t_span: list[float], y0, N=100, args=()):
    """
    Runge-Kutta 4th order method for solving ODEs, RK4
    Constant step size

    Both SciPy solve_ivp and MATLAB ode45 use the Dormand-Prince 4th order Runge Kutta method,
    which is slightly different, being a 6-term, 5th order method with adaptive step size control

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
        return y.at[i].set(rk4Step(func, t[i - 1], y[i - 1, :], h, args))

    y = jax.lax.fori_loop(1, N + 1, body_fun, y)
    return t, y

def reRK4(func, t_span: list[float], y0, N=100, args=()):
    """
    Runge-Kutta 4th order method for solving ODEs, RK4
    With rectification
    Constant step size

    Both SciPy solve_ivp and MATLAB ode45 use the Dormand-Prince 4th order Runge Kutta method,
    which is slightly different, being a 6-term, 5th order method with adaptive step size control

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

    # original code
    # y = jax.lax.fori_loop(1, N+1, lambda i, y: y.at[i].set(rk4Step(func, t[i-1], y[i-1, :], h, args)), y)
    # alternative
    def body_fun(i, y):
        return y.at[i].set(reRK4Step(func, t[i - 1], y[i - 1, :], h, args))

    y = jax.lax.fori_loop(1, N + 1, body_fun, y)
    return t, y

def newton(func, y0, tol=1e-6, max_iter=1000, args=(), lr: float=1.0):
    """
    Newton's method for solving systems of nonlinear equations
    Args:
    func: function to solve, with arguments (y, *args)
    y0: initial estimate; must be float array, not integer array
    tol: tolerance
    max_iter: maximum number of iterations
    args: additional arguments to pass to func
    lr: learning rate
    Returns:
    y: solution
    converged: boolean indicating whether the method converged
    """
    f_prime = jax.jacfwd(func)
    y = y0
    n = 0
    error = tol+1
    converged = False

    def loop_body(state):
        y, error, max_iter, n, converged = state
        y_new = y - lr * jnp.linalg.solve(f_prime(y, *args), func(y, *args))
        error = jnp.linalg.norm(y_new - y)
        converged = jax.lax.cond(error <= tol, lambda _: True, lambda _: converged, None)
        return y_new, error, max_iter, n + 1, converged

    def cond_fun(state):
        y, error, max_iter, n, converged = state
        return jnp.logical_and(error > tol, n < max_iter)

    y, error, max_iter, n, converged = jax.lax.while_loop(cond_fun, loop_body, (y, error, max_iter, n, converged))

    return y, converged

def newton_dynamical2(func, t, y0, u, w, tol=1e-6, max_iter=1000, args=(), lr: float=1.0):
    """
    Newton's method for solving systems of nonlinear equations
    Function of time with additional arguments u and w
    Args:
    func: function to solve, with arguments (t, y, u, w, args)
    t: time; must be float
    y0: initial estimate; must be float array, not integer array
    tol: tolerance
    max_iter: maximum number of iterations
    args: additional arguments to pass to func
    Returns:
    y: solution
    converged: boolean indicating whether the method converged
    """
    y = y0
    n = 0
    error = tol+1
    converged = False
    f_prime = jax.jacfwd(lambda t, y, u, w, args: func(t, y, u, w, args), argnums=1)


    def loop_body(state):
        y, u, w, error, max_iter, n, converged, args = state
        y_new = y - lr * jnp.linalg.solve(f_prime(t, y, u, w, args), func(t, y, u, w, args))
        error = jnp.linalg.norm(y_new - y)
        converged = jax.lax.cond(error <= tol, lambda _: True, lambda _: converged, None)
        return y_new, u, w, error, max_iter, n + 1, converged, args

    def cond_fun(state):
        y, u, w, error, max_iter, n, converged, args = state
        return jnp.logical_and(error > tol, n < max_iter)

    y, error, _, _, max_iter, n, converged, args = jax.lax.while_loop(cond_fun, loop_body, (y, u, w, error, max_iter, n, converged, args))

    return y, converged