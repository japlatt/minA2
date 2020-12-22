import numpy as np
from pyoptsparse import Optimization, OPT

class Action:

    def __init__(self, params):
        self.name = params.get('name')

        N_data = params['ndata'] if params['ndata'] > 0 else None
        self.set_data_fromfile(params['data_folder']+params['data_file'],
                               params.get('stim_file'),
                               nstart = params.get('nstart', 0),
                               N = N_data)
        self.action_init(params)


    def action_init(self, params):
        '''
        Args:
        '''
        self.f              = params.get('f')
        self.fjacx          = params.get('fjacx')
        self.fjacp          = params.get('fjacp')
        alpha               = params.get('alpha')
        self.Rm             = params.get('Rm')
        self.Rf0            = params.get('Rf0')
        self.Lidx           = params.get('Lidx')
        dt_model            = params.get('dt_model')
        self.optimizer      = params.get('optimizer', 'IPOPT')
        self.opt_options    = params.get('opt_options')
        self.NP             = params.get('num_par')
        self.D              = params.get('num_dims')
        self.var_bounds     = params.get('bounds')[:self.D]
        self.par_bounds     = params.get('bounds')[self.D:]


        assert(type(dt_model) is int)
        self.dt_model = self.dt_data/dt_model
        self.N_model = (self.N_data - 1) * dt_model + 1
        self.t_model = np.linspace(self.t_data[0], self.t_data[-1], self.N_model, dtype = np.float32)
        if self.stim is not None:
            self.stim = np.interp(self.t_model, self.t_data, self.stim).astype(np.float32)




        self.model_skip = dt_model
        self.Rf = RF0 * alpha**beta_array

        self.minpaths = np.zeros((len(beta_array)+1, self.N_model*self.D+self.NP), dtype=np.float32)

        self.X0 = np.zeros((self.N_model, self.D), dtype = np.float32)
        for i, b in enumerate(self.var_bounds):
            self.X0[:, i] = np.float32(np.random.uniform(low =b[0], high = b[1], size = self.N_model))
        self.X0[::self.model_skip, Lidx] = self.Y

        self.P0 = np.zeros(self.NP, dtype = np.float32)
        for i, b in enumerate(self.par_bounds):
            self.P0[i] = np.float32(np.random.uniform(low = b[0], high = b[1]))

        self.minpaths[0] = np.concatenate(self.X0.flatten(), self.P0)


    def set_data_fromfile(self, data_file, stim_file=None, nstart=0, N=None):
        """
        Load data & stimulus time series from file.
        If data is a text file, must be in multi-column format with L+1 columns:
            t  y_1  y_2  ...  y_L
        If a .npy archive, should contain an Nx(D+1) array with times in the
        zeroth element of each entry.
        Column/array formats should also be in the form t  s_1  s_2 ...
        """
        
        data = np.load(data_file) if data_file.endswith('npy') else np.loadtxt(data_file)
        data = data.astype(np.float32)

        self.N_data = N if N is not None else data.shape[0]
        self.t_data = data[:, 0]
        self.dt_data = self.t_data[1] - self.t_data[0]
        self.Y = data[:, 1:]

        if stim_file is not None:
            self.stim = np.load(stim_file) if stim_file.endswith('npy')[:, 1:]\
                                           else np.loadtxt(stim_file)[:, 1:]
            self.stim = self.stim.astype(np.float32)

        else: self.stim = None



    def min_A(self):
        f_mpath = open(self.name+'_min_paths.txt', 'bw+')
        f_maction = open(self.name+'_min_action.txt', 'bw+')
        for i, rf in enumerate(self.Rf):
            self.rf = rf
            self.minpaths[i+1], Amin = self._min_A_step(i)
            np.savetxt(f_mpath, self.minpaths[i+1], fmt = '%1.5f')
            np.savetxt(f_maction, Amin, fmt = '%1.5f')
            f_mpath.flush()
            f_maction.flush()


    ##### PRIVATE FUNCTIONS #############

    def disc_trapezoid(self, x, p):
        """
        Time discretization for the action using the trapezoid rule.
        """
        stim_n = self.stim[:-1]
        stim_np1 = self.stim[1:]

        fn = np.array(self.f(x[:-1].T, self.t_model[:-1], p, stim_n)).T
        fnp1 = np.array(self.f(x[1:].T, self.t_model[1:], p, stim_np1)).T

        return x[:-1] + self.dt_model * (fn + fnp1) / 2.0

    def _action(self, xdict):
        x = xdict['x']
        p = xdict['p']
        funcs = {}

        X = np.reshape(x, (self.N_model, self.D))

        diff_m = X[::self.model_skip, self.Lidx] - self.Y
        merr = self.Rm * np.linalg.norm(diff_m)**2
        merr/=(len(self.Lidx) * self.N_data)

        diff_f = X[1:] - self.disc_trapezoid(X, p)
        ferr = self.rf*np.linalg.norm(diff_f)**2
        ferr/=(self.D * (self.N_model - 1))

        funcs['obj'] = merr+ferr

        return funcs, False


    def _grad_action(self, xdict, funcs):
        x = xdict['x']
        p = xdict['p']
        X = np.reshape(x, (self.N_model, self.D))

        funcSens = {}

        dmdx = np.zeros((self.N_model, self.D))
        dmdx[::self.model_skip, self.Lidx] = self.Rm*(X[::self.model_skip, self.Lidx] - self.Y)
        dmdx = dmdx.flatten()
        dmdx/=(len(self.Lidx)/2 * self.N_data)

        diff_f = X[1:] - self.disc_trapezoid(X, p)

        J = np.zeros((self.D, self.D))
        J = self.fjacx(X[0], self.t_model[0], p, self.stim[0])
        dfdx = np.zeros((self.N_model, self.D))
        dfdx[0] = self.rf*np.sum(-diff_f[0].reshape(-1, 1)*
                                 (np.eye(self.D) + 0.5*self.dt_model*J),
                                 axis = 0)
        
        for i in range(1, self.N_model-1):
            Jp1 = self.fjacx(X[i], self.t_model[i], p, self.stim[i])
            dfdx[i] = self.rf*np.sum(diff_f[i-1].reshape(-1, 1)*
                                     (np.eye(self.D) - 0.5*self.dt_model*J),
                                     axis = 0) + \
                      self.rf*np.sum(-diff_f[i].reshape(-1, 1)*
                                     (np.eye(self.D) + 0.5*self.dt_model*Jp1),
                                     axis = 0)
            J = Jp1

        dfdx[-1] = self.rf*np.sum(diff_f[-1].reshape(-1, 1)*
                                 (np.eye(self.D) - 0.5*self.dt_model*Jp1),
                                 axis = 0)

        dfdx = (dfdx/(self.D/2 * (self.N_model - 1))).flatten()

        dfdp = np.zeros((self.N_model, self.NP))
        G = self.fjacp(X[0], self.t_model[0], p, self.stim[0])
        for i in range(self.N_model-1):
            Gp1 = self.fjacp(X[i+1], self.t_model[i+1], p, self.stim[i+1])
            dfdp[i] = self.rf*np.sum(diff_f[i].reshape(-1, 1)*(G+Gp1), axis = 0)
            G = Gp1
        dfdp = -self.dt_model*np.sum(dfdp, axis = 0)/(self.D * (self.N_model - 1))
        
        funcSens['obj'] = {'x' : dmdx+dfdx,
                           'p' : dfdp}

        return funcSens, False


    def _optimize(self, XP0):
        optProb = Optimization("action", self._action)
        optProb.addVarGroup("x",
                             self.N_model*self.D,
                             "c",
                             lower = np.repeat(self.var_bounds[:, 0], self.N_model),
                             upper = np.repeat(self.var_bounds[:, 1], self.N_model),
                             value = XP0[:self.N_model*self.D])
        optProb.addVarGroup("p",
                            self.NP,
                            "c",
                            lower = self.par_bounds[:, 0],
                            upper = self.par_bounds[:, 1],
                            value = XP0[self.N_model*self.D:])

        optProb.addObj("obj")
        opt = OPT(self.optimizer, options = self.opt_options)
        sol = opt(optProb, sens = self._grad_action)
        return np.concatenate(sol.xStar['x'], sol.xStar['p']), sol.fStar




    def _min_A_step(self, beta_i):
        self.beta_i = beta_i
        XP0 = self.minpaths[beta_i]
        XPmin, Amin = self._optimize(XP0)
        return XPmin, Amin








             























    
    

