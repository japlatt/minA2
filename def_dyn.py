from sympy import *
import numpy as np
from numba import njit

################# MODIFY #################

def dynamics(x, t, p, stim = None):
    D = len(x)
    dxdt = zeros(1, D)
    for i in range(D-1):
        dxdt[i] = (x[i+1] - x[i-2])*x[i-1] - x[i]
    dxdt[-1] = (x[0] - x[-3])*x[-2]-x[-1]
    dxdt = dxdt + p[0]*ones(1, D)
    return Matrix([dxdt])

##########################################


def get_dynamics(specs):
    num_vars = specs['num_dims']
    num_pars = specs['num_par']

    x = np.array(symbols('x:{:d}'.format(num_vars)))
    p = np.array(symbols('p:{:d}'.format(num_pars)))
    stim = symbols('stim')
    t = symbols('t')

    f = dynamics(x, t, p, stim)
    fjacx = dynamics(x, t, p, stim).jacobian(x)
    fjacp = dynamics(x, t, p, stim).jacobian(p)

    lam_f = lambdify((x, t, p, stim), np.squeeze(f))
    lam_fjacx = lambdify((x, t, p, stim), fjacx)
    lam_fjacp = lambdify((x, t, p, stim), fjacp)

    return njit(lam_f), njit(lam_fjacx), njit(lam_fjacp)