import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdify
import yaml

from def_dyn import dynamics

def get_dynamics(dynamics, specs):
    num_vars = specs['num_dims']
    num_pars = specs['num_par']

    x = sym.symbols('x:{:d}'.format(num_vars))
    p = sym.symbols('p:{:d}'.format(num_pars))
    stim = sym.symbols('stim')
    t = sym.symbols('t')

    f = dynamics(x, t, p, stim)
    fjacx = dynamics(x, t, p, stim).jacobian(x)
    fjacp = dynamics(x, t, p, stim).jacobian(p)

    lam_f = lambdify((x, t, p, stim), np.squeeze(f))
    lam_fjacx = lambdify((x, t, p, stim), fjacx)
    lam_fjacp = lambdify((x, t, p, stim), fjacp)

    return lam_f, lam_fjacx, lam_fjacp



def read_specs(path_to_specs):
    with open(path_to_specs) as file:
        specs = yaml.load(file, Loader=yaml.FullLoader)
    return specs
    


if __name__ == '__main__':
    specs = read_specs('specs.txt')
    f, fjacx, fjacp = get_dynamics(dynamics, specs)

    optimizer = 'IPOPT'

    opt_options = { 
                    # max_iter : 10000, # Set the max number of iterations
                    # derivative_test : second-order, # set derivative test
                    # tol : 1.0e-12, # set termination criteria
                    # dual_inf_tol : 0.001,
                    # compl_inf_tol : 1.0e-12,
                    # constr_viol_tol : 1.0e-8,
                    # acceptable_tol : 1.0e-10,
                    # acceptable_iter 
                    # nlp_scaling_method : none, #turn off the NLP scaling
                    # mehrotra_algorithm : yes,
                    'linear_solver' : 'mumps',
                    # mu_strategy : adaptive,
                    # adaptive_mu_globalization : never-monotone-mode,
                    # linear_system_scaling : none,
                    # bound_relax_factor = 0
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
              'Rm'              : specs['RM'],
              'Rf0'             : specs['RF0'],
              'Lidx'            : obs_dim,
              'dt_model'        : specs['time_step'],
              'optimizer'       : ,
              'opt_options'     : ,
              'num_pars'        : ,
              'num_dims'        : ,
              'bounds'          : ,
              }

    
