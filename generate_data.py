import numpy as np
import matplotlib.pyplot as plt

from run_da import read_specs, get_dynamics
from def_dyn import dynamics

def RK4(f, r, t, dt, params = None):
    k1 = dt*f(r,t, params)
    k2 = dt*f(r+k1/2, t+dt/2, params)
    k3 = dt*f(r+k2/2, t+dt/2, params)
    k4 = dt*f(r+k3, t+dt, params)
    return r + (k1+2*k2+2*k3+k4)/6



if __name__ == '__main__':
    ###### MODIFY HERE ###############
    path_to_specs = 'specs.yaml'
    dt = 0.025
    num_data = 1000
    x0 = 20*np.random.rand(5)-10
    p = (8.17,)
    noise_std = np.sqrt(0.25)
    ##################################
    
    specs = read_specs(path_to_specs)
    dyn, _, _ = get_dynamics(dynamics, specs)

    f = lambda r, t, params: np.array(dyn(r, t, *params))
    if specs.get('stim_file') is not None:
        stim = np.load(specs['data_folder']+specs['stim_file'])[:, 1]
    else:
        stim = None

    time_arr = np.arange(0, num_data*dt, dt)
    sol = np.zeros((num_data+1, specs['num_dims']))
    sol[0] = x0
    for i, t in enumerate(time_arr, 1):
        sol[i] = RK4(f, sol[i-1], t, dt, params = (p, stim))

    plt.plot(time_arr, sol[:-1, 0])
    plt.show()

    obs_dim = specs['obs_dim'] if specs['obs_dim'] != -1 else np.arange(specs['num_dims'])
    np.save(specs['data_folder']+specs['data_file'],
               np.vstack((time_arr, sol[:-1].T[obs_dim]+np.random.normal(0, noise_std, (sol[:-1].T[obs_dim].shape)))).T)
    if len(obs_dim) != specs['num_dims']:
            np.save(specs['data_folder']+'all_'+specs['data_file'], np.vstack((time_arr, sol[:-1].T)).T)
    
    






