from sympy import *

def dynamics(X, t, p, stim = None):
    a, b, tau = p
    v, w = X
    dvdt = (v-v**3/3 - w + stim)/tau
    dwdt = v - b*w + a
    return Matrix([dvdt, dwdt])