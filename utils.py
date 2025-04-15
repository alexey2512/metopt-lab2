import numpy as np
import scipy.optimize as so

def __basis_v(dim, l):
    b = np.zeros(dim)
    b[l] = 1
    return b


# global counter of derivative invocations
der_count = 0
def __approx_der(func, x, dx, precision=1e-8):
    global der_count
    der_count += 1
    return (func(x + precision * dx) - func(x - precision * dx)) / precision / 2.0


# global counter of gradient invocations
grad_count = 0
def __grad(func_n, x):
    global grad_count
    grad_count += 1
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        dxi = __basis_v(n, i)
        result[i] = __approx_der(func_n, x, dxi)
    return result


# global counter of hessian invocations
hess_count = 0
def __hess(func_n, x):
    global hess_count
    hess_count += 1
    n = len(x)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dxi = __basis_v(n, i)
            dxj = __basis_v(n, j)
            result[i][j] = __approx_der(lambda t: __approx_der(func_n, t, dxi), x, dxj)
        for j in range(i):
            result[i][j] = result[j][i]
    return result


def wolfe(phi, c1=1e-4, c2=0.9, amax=None, max_iter=40):
    phi0 = phi(0)
    derphi0 = __approx_der(phi, 0, 1)
    a = min(1.0, abs(0.5 / derphi0)) if derphi0 != 0 else 1.0
    c2_adjusted = 0.5 if abs(derphi0) > 1e3 else c2
    for _ in range(max_iter):
        if amax is not None and a > amax:
            a = amax
        phi_a = phi(a)
        derphi_a = __approx_der(phi, a, 1)
        wolfe1 = phi_a <= phi0 + c1 * a * derphi0
        wolfe2 = derphi_a >= c2_adjusted * derphi0
        if wolfe1 and wolfe2:
            return a
        if not wolfe1:
            a *= 0.5
        else:
            a *= 1.5
        if a < 1e-10:
            return a
    return a


def __orient_in_bounds(x, p, bounds):
    if bounds is None:
        return p, None
    lbs = [i[0] for i in bounds]
    ubs = [i[1] for i in bounds]
    new_p = p.copy()
    amax = float('inf')
    for i in range(len(x)):
        far = 0
        if p[i] > 0:
            far = (ubs[i] - x[i]) / p[i]
        if p[i] < 0:
            far = (lbs[i] - x[i]) / p[i]
        if far == 0:
            new_p[i] = 0
        else:
            amax = min(far, amax)
    return new_p, amax

def __descent(func_n, start, strategy, bounds, tol, max_iter, get_direction):
    xk = start
    poses = [start]
    for k in range(max_iter):
        p = get_direction(xk)
        p, amax = __orient_in_bounds(xk, p, bounds)
        if np.linalg.norm(p) < tol:
            break
        func_1 = lambda a: func_n(xk + a * p)
        alpha = strategy(k, func_1, amax)
        xk = xk + alpha * p
        poses.append(xk)
        if np.linalg.norm(poses[-1] - poses[-2]) < tol:
            break
    return poses

def gradient_descent(
        func_n,
        start,
        strategy=lambda k, fu, ma: 1 if ma is None else min(1, ma),
        bounds=None,
        tol=1e-8,
        max_iter=150,
):
    def gradient_direction(x):
        return -__grad(func_n, x)
    return __descent(func_n, start, strategy, bounds, tol, max_iter, gradient_direction)

def newton_descent(
        func_n,
        start,
        strategy=lambda k, fu, ma: 1 if ma is None else min(1, ma),
        bounds=None,
        tol=1e-8,
        max_iter=30
):
    def newton_direction(x):
        grd = __grad(func_n, x)
        hss = __hess(func_n, x)
        hss += tol * np.eye(len(x))
        try:
            direction = -np.linalg.solve(hss, grd)
            if np.linalg.norm(direction) < tol:
                return -grd
            return direction
        except np.linalg.LinAlgError:
            return -grd
    return __descent(func_n, start, strategy, bounds, tol, max_iter, newton_direction)


def bfgs_descent(
        func_n,
        start,
        strategy=lambda k, fu, ma: 1 if ma is None else min(1, ma),
        bounds=None,
        tol=1e-8,
        max_iter=30,
):
    n = len(start)
    poses = [start]
    xk = start
    dxk = __grad(func_n, xk)
    hk = np.eye(n)
    for k in range(max_iter):
        p = -hk @ dxk
        p, amax = __orient_in_bounds(xk, p, bounds)
        if np.linalg.norm(p) < tol:
            break
        func_1 = lambda a: func_n(xk + a * p)
        xk1 = xk + strategy(k, func_1, amax) * p
        dxk1 = __grad(func_n, xk1)

        s = xk1 - xk
        y = dxk1 - dxk
        ro = 1.0 / ((y @ s) + tol)
        im = np.eye(n)
        sy = np.outer(s, y)
        hk = (im - ro * sy) @ hk @ (im - ro * sy.T) + ro * np.outer(s, s)

        xk = xk1
        dxk = dxk1
        poses.append(xk)

        if np.linalg.norm(poses[-1] - poses[-2]) < tol:
            break

    return poses



# ==========
# strategies:

def fixed_step(step):
    return lambda k, fu, ma: step if ma is None else min(ma, step)

def polynomial_decay(alpha, beta):
    return lambda k, fu, ma: 1 / np.sqrt(k + 1) / (beta * k + 1) ** alpha

def wolfe_strategy(_, fu, ma):
    return wolfe(fu, amax=ma)

def so_wolfe_strategy(_, fu, ma):
    phi = fu
    result = so.line_search(
        phi, lambda a: __approx_der(phi, a, 1),
        np.array(0.0), np.array(1.0), amax=ma
    )
    alpha = result[0] if result is not None else None
    return alpha if alpha is not None else 0


# =======
# lib methods:

def __so_minimize(func_n, start, method, jac=None, hes=None, tol=1e-10):
    poses = []
    so.minimize(
        fun=func_n,
        x0=start,
        method=method,
        jac=jac,
        hess=hes,
        tol=tol,
        callback=lambda xk: poses.append(xk.copy())
    )
    return poses

# uses wolfe as step strategy
def so_newton_sg(func_n, start):
    return __so_minimize(
        func_n, start, 'Newton-CG',
        jac=lambda x: __grad(func_n, x),
        hes=lambda x: __hess(func_n, x)
    )

# uses wolfe as step strategy
def so_bfgs(func_n, start):
    return __so_minimize(
        func_n, start, 'BFGS',
        jac=lambda x: __grad(func_n, x)
    )

def so_nelder_mead(func_n, start):
    return __so_minimize(func_n, start, 'Nelder-Mead')

