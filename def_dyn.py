from sympy import *
import numpy as np

def dynamics(x, t, p, stim = None):
    D = len(x)
    dxdt = zeros(1, D)
    for i in range(D-1):
        dxdt[i] = (x[i+1] - x[i-2])*x[i-1] - x[i]
    dxdt[-1] = (x[0] - x[-3])*x[-2]-x[-1]
    dxdt = dxdt + p[0]*ones(1, D)
    return Matrix([dxdt])