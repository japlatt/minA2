import numpy as np
import yaml


def RK4(f, r, t, dt, params = None):
    k1 = dt*f(r,t, params)
    k2 = dt*f(r+k1/2, t+dt/2, params)
    k3 = dt*f(r+k2/2, t+dt/2, params)
    k4 = dt*f(r+k3, t+dt, params)
    return r + (k1+2*k2+2*k3+k4)/6

def read_specs(path_to_specs):
    with open(path_to_specs) as file:
        specs = yaml.load(file, Loader=yaml.FullLoader)
    return specs

def read_bounds(path_to_bounds):
    return np.loadtxt(path_to_bounds, delimiter=',')

