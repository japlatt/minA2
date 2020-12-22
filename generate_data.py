import numpy as np

from run_da import read_specs, get_dynamics
from def_dyn import dynamics

def RK4(f, r, t, dt, params = None):
    k1 = dt*f(r,t, params)
    k2 = dt*f(r+k1/2, t+dt/2, params)
    k3 = dt*f(r+k2/2, t+dt/2, params)
    k4 = dt*f(r+k3, t+dt, params)
    return r + (k1+2*k2+2*k3+k4)/6



if __name__ == '__main__':
    specs = read_specs('specs.yaml')
    dyn, _, _ = get_dynamics(dynamics, specs)

    f = lambda r, t, params: np.array(dyn(r, t, *params))
    
    ###### MODIFY HERE ###############
    dt = 0.02
    num_data = 20000
    x0 = [-1.2, .6]
    p = (0.7, 0.8, 0.08)
    time_arr = np.arange(0, num_data*dt, dt)
    stim = np.sin(time_arr)
    #stim = np.load(specs['data_folder']+specs['stim_file'])
    ##################################

    sol = np.zeros(num_data+1, specs['num_dims'])
    sol[0] = x0
    for i, t in enumerate(time_arr, 1):
        sol[i] = RK4(f, sol[i-1], t, dt, params = (p, stim[i-1]))

    np.savetxt(specs['data_folder']+specs['data_file'],
               np.vstack((time_arr, sol[:-1].T)).T,
               fmt = '%1.5f')
    
    






