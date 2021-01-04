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
readme

test
-20D L96
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
    host 192.168.0.133
    host myhost
    host myhost2

group boom-cluster:
    host compute-0-7
    host compute-0-8
    host compute-0-9
    host compute-0-10
    host compute-0-11
    host compute-0-12
    host compute-0-13
'''

import numpy as np
import yaml
from charm4py import charm
from functools import partial

from def_dyn import get_dynamics
from model import Action


######## Modify Here ##############
path_to_folder = 'Runs/Lor96_20D/'
path_to_specs = path_to_folder+'specs.yaml'
path_to_params = path_to_folder+'params.txt'
max_iter_per_step = 1000
tol_per_step = 1e-8
num_init_cond = 50
###################################

def read_specs(path_to_specs):
    with open(path_to_specs) as file:
        specs = yaml.load(file, Loader=yaml.FullLoader)
    return specs

def read_bounds(path_to_bounds):
    return np.loadtxt(path_to_bounds, delimiter=',')


def min_action(random_seed, params):
    pid = random_seed.spawn_key[0]
    action = Action(params, random_seed)
    return action.min_A(pid)
    

def run(args):
    specs = read_specs(path_to_specs)
    f, fjacx, fjacp = get_dynamics(specs)

    if specs.get('generate_twin', False) is True:
        generate_twin(specs, f)

    optimizer = specs.get('optimizer','IPOPT')

    opt_options = { 
                    'print_level' : 0,
                    'max_iter' : max_iter_per_step, # Set the max number of iterations
                    # derivative_test : second-order, # set derivative test
                    'tol' : tol_per_step, # set termination criteria
                    # 'dual_inf_tol' : 0.001,
                    # 'compl_inf_tol' : 1.0e-12,
                    # 'constr_viol_tol' : 1.0e-8,
                    # 'acceptable_tol' : 1.0e-10,
                    # 'nlp_scaling_method' : none, #turn off the NLP scaling
                    # 'mehrotra_algorithm' : yes,
                    'linear_solver' : 'mumps',
                    # 'mu_strategy' : 'adaptive',
                    # 'adaptive_mu_globalization' : 'never-monotone-mode',
                    # 'linear_system_scaling' : none,
                    # 'bound_relax_factor' : 0
                    'output_file' : specs['data_folder']+'IPOPT.out',
                    }


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
    sol = np.array(charm.pool.map(partial(min_action, params = params), init_seeds), dtype=object)

    np.savez(specs['data_folder']+specs['name']+'_results.npz',
             path = sol[:, 0],
             params = sol[:, 1],
             action = sol[:, 2],
             time = sol[0, 3],
             **specs
             )

    exit()

if __name__ == '__main__':
    charm.start(run)

    
