'''
To Do:
charm for multiple init cond
smarter saving for multiple init cond
make compatible with multiple experiments with samre functions
error checking
comments

test
-5D L96 with 2 obs variables
-20D L96
-NaKL Neuron
'''

import numpy as np
import sympy as sym
import yaml
from numba import njit

from def_dyn import dynamics
from model import Action

def get_dynamics(dynamics, specs):
    num_vars = specs['num_dims']
    num_pars = specs['num_par']

    x = np.array(sym.symbols('x:{:d}'.format(num_vars)))
    p = np.array(sym.symbols('p:{:d}'.format(num_pars)))
    stim = sym.symbols('stim')
    t = sym.symbols('t')

    f = dynamics(x, t, p, stim)
    fjacx = dynamics(x, t, p, stim).jacobian(x)
    fjacp = dynamics(x, t, p, stim).jacobian(p)

    lam_f = sym.lambdify((x, t, p, stim), np.squeeze(f))
    lam_fjacx = sym.lambdify((x, t, p, stim), fjacx)
    lam_fjacp = sym.lambdify((x, t, p, stim), fjacp)

    return njit(lam_f), njit(lam_fjacx), njit(lam_fjacp)



def read_specs(path_to_specs):
    with open(path_to_specs) as file:
        specs = yaml.load(file, Loader=yaml.FullLoader)
    return specs

def read_bounds(path_to_bounds):
    return np.loadtxt(path_to_bounds, delimiter=',')
    

if __name__ == '__main__':
    ######## Modify Here ##############
    path_to_specs = 'specs.yaml'
    path_to_params = 'params.txt'
    ###################################


    specs = read_specs(path_to_specs)
    f, fjacx, fjacp = get_dynamics(dynamics, specs)

    if specs.get('generate_twin', False) is True:
        generate_twin(specs, f)

    optimizer = specs.get('optimizer','IPOPT')

    opt_options = { 
                    'print_level' : 0,
                    'max_iter' : 1000, # Set the max number of iterations
                    # derivative_test : second-order, # set derivative test
                    'tol' : 1.0e-10, # set termination criteria
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

    action = Action(params)
    action.min_A(0)
    
