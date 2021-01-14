#!/usr/bin/env python
'''
Jason Platt (jplatt@ucsd.edu)
Department of Physics
University of California, San Diego
2021
'''


import numpy as np
import matplotlib.pyplot as plt
import os

from run_da import path_to_specs, read_specs
from def_dyn import get_dynamics

def RK4(f, r, t, dt, params = None):
    k1 = dt*f(r,t, params)
    k2 = dt*f(r+k1/2, t+dt/2, params)
    k3 = dt*f(r+k2/2, t+dt/2, params)
    k4 = dt*f(r+k3, t+dt, params)
    return r + (k1+2*k2+2*k3+k4)/6


if __name__ == '__main__':
    ########### MODIFY HERE ###########
    dt = 0.025 # time step of the model
    num_data = 20000 # number of time steps to generate
    x0 = np.random.rand(4) # initial condition
    p = (1, 120, 20, 0.3, 50, -77, -54.4,\
        -40, -60, -55, 15, -15, 30, 0.1,\
        0.4, 1, 7, 1, 5,) # parameters, must be in tuple
    noise_std = np.sqrt(1e-5) # add noise to the data
    # dt = 0.025 # time step of the model
    # num_data = 1000 # number of time steps to generate
    # x0 = 20*np.random.rand(5)-10 # initial condition
    # p = (8.17,) # parameters, must be in tuple
    # noise_std = np.sqrt(0.25) # add noise to the data
    ####################################
    
    specs = read_specs(path_to_specs)
    dyn, _, _ = get_dynamics(specs)
    if not os.path.exists(specs['data_folder']): os.makedirs(specs['data_folder'])

    f = lambda r, t, params: np.array(dyn(r, t, *params))
    if specs.get('stim_file') is not None:
        stim = np.load(specs['data_folder']+specs['stim_file'])[:, 1]
    else:
        stim = np.empty(num_data)

    time_arr = np.linspace(0, num_data*dt, num_data)
    sol = np.zeros((num_data+1, specs['num_dims']))
    sol[0] = x0
    for i, t in enumerate(time_arr, 1):
        sol[i] = RK4(f, sol[i-1], t, dt, params = (p, stim[i-1]))

    plt.plot(time_arr, sol[:-1, 0])
    plt.show()

    obs_dim = specs['obs_dim'] if specs['obs_dim'] != -1 else np.arange(specs['num_dims'])
    np.save(specs['data_folder']+specs['data_file'],
               np.vstack((time_arr, sol[:-1].T[obs_dim]+np.random.normal(0, noise_std, (sol[:-1].T[obs_dim].shape)))).T)
    np.save(specs['data_folder']+'all_'+specs['data_file'], np.vstack((time_arr, sol[:-1].T)).T)
    
    






