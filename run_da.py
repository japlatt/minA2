#!/usr/bin/env python
'''
Jason Platt (jplatt@ucsd.edu)
Department of Physics
University of California, San Diego
2021
'''

'''
To Do:
error checking
add comments

test
-NaKL Neuron
'''

'''
Command to run DA, X means replace with number.  With pool, one process is used
to schedule jobs so number of jobs run in parallel is (# hosts * process_per_host)-1.

Single Machine  : python -m charmrun.start ++processPerHost X run_da.py +isomalloc_sync
CPU Cluster     : python -m charmrun.start ++numHosts X ++processPerHost X run_da.py ++nodelist nodelist.txt +isomalloc_sync

nodelist is to run an application on multiple hosts (e.g. a network of workstations)
by passing it a file containing the list of nodes (nodelist file).
Hosts can be specified by IP address or host name.
A couple examples:

group mynodes
    host 192.168.0.10
    host myhost

group boom-cluster:
    host compute-0-7
    host compute-0-8
'''

import numpy as np
from functools import partial

from def_dyn import get_dynamics
from model import Action
from utils import read_specs, read_bounds


######## Modify Here ##############
path_to_folder = 'Runs/NaKL/'
path_to_specs = path_to_folder+'specs.yaml'
path_to_params = path_to_folder+'params.txt'
max_iter_per_step = 50000
tol_per_step = 1e-6
num_init_cond = 1
###################################

charm = True if num_init_cond > 1 else 0

def min_action(random_seed, params):
    pid = random_seed.spawn_key[0]
    action = Action(params, random_seed)
    return action.min_A(pid)
    

def run(Args):
    specs = read_specs(path_to_specs)
    f, fjacx, fjacp = get_dynamics(specs)

    if specs.get('generate_twin', False) is True:
        generate_twin(specs, f)

    optimizer = specs.get('optimizer','IPOPT')
    print_level = 0 if charm else 5

    if optimizer == 'IPOPT':
        opt_options = { 
                        'print_level' : print_level,
                        'max_iter' : max_iter_per_step, # Set the max number of iterations
                        # 'derivative_test' : 'first-order', # set derivative test
                        'tol' : tol_per_step, # set termination criteria
                        'linear_solver' : 'mumps',
                        'output_file' : specs['data_folder']+'IPOPT.out'
                        }
    if optimizer == 'SNOPT':
        from snopt import SNOPT_options
        opt_options = SNOPT_options()
        opt_options.setOption('Print filename', specs['data_folder']+'SNOPT.out')
        opt_options.setOption('Iteration limit', max_iter_per_step)
        opt_options.setOption('Optimality tolerance', tol_per_step)
        if charm:
            opt_options.setOption('Major print level', 0)
            opt_options.setOption('Minor print level', 0)
            opt_options.setOption('Verbose',False)


    obs_dim = specs['obs_dim'] if specs['obs_dim'] != -1 else np.arange(specs['num_dims'])

    params = {'name'            : specs['name'],
              'data_folder'     : specs['data_folder'],
              'data_file'       : specs['data_file'],
              'stim_file'       : specs.get('stim_file'),
              'nstart'          : specs.get('start_data', 0),
              'ndata'           : specs.get('num_data', -1),
              'f'               : f,
              'fjacx'           : fjacx,
              'fjacp'           : fjacp,
              'alpha'           : specs['alpha'],
              'max_beta'        : specs['max_beta'],
              'Rm'              : specs['RM'],
              'Rf0'             : specs['RF0'],
              'Lidx'            : obs_dim,
              'dt_model'        : specs['time_step'],
              'optimizer'       : optimizer,
              'opt_options'     : opt_options,
              'num_pars'        : specs['num_par'],
              'num_dims'        : specs['num_dims'],
              'bounds'          : read_bounds(path_to_params),
              }

    rng = np.random.default_rng()
    ss = rng.bit_generator._seed_seq
    init_seeds = ss.spawn(num_init_cond)
    if charm:
        sol = np.array(charm.pool.map(partial(min_action, params = params), init_seeds), dtype=object)
        np.savez(specs['data_folder']+specs['name']+'_results.npz',
                path = np.array(sol[:, 0]),
                params = np.array(sol[:, 1]),
                action = np.array(sol[:, 2]),
                time = np.array(sol[0, 3]),
                **specs
                )
    else:
        sol = min_action(init_seeds[0], params)
        np.savez(specs['data_folder']+specs['name']+'_results.npz',
                path = sol[0],
                params = sol[1],
                action = sol[2],
                time = sol[3],
                **specs
                )

    exit()

if __name__ == '__main__':
    if charm:
        from charm4py import charm
        charm.start(run)
    else:
        run(None)

    
