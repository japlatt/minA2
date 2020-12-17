import numpy as np
import pyoptsparse
import sympy as sym
from sympy.utilities.lambdify import lambdify
import yaml

from def_dyn import dynamics

def write_dynamics(dynamics, specs):
    num_vars = specs['num_dims']
    num_par = specs['num_par']

    x = sym.symbols('x:{:d}'.format(num_var))
    p = sym.symbols('p:{:d}'.format(num_par))
    stim = sym.symbols('stim')
    t = sym.symbols('t')

    f = dynamics(x, t, p, stim)
    fjacx = dynamics(x, t, p, stim).jacobian(x)
    fjacp = dynamics(x, t, p, stim).jacobian(p)

    lam_f = lambdify((x, t, p, stim), f, 'numpy')
    lam_fjacx = lambdify((x, t, p, stim), fjacx, 'numpy')
    lam_fjacp = lambdify((x, t, p, stim), fjacp, 'numpy')

    return lam_f, lam_fjacx, lam_fjacp



def read_spec(path_to_specs):
    with open(path_to_specs) as file:
        specs = yaml.load(file, Loader=yaml.FullLoader)
    return specs
    


if __name__ == '__main__':
    specs = read_specs('specs.txt')
    f, fjacx, fjacp = write_dynamics(dynamics, specs)

    
