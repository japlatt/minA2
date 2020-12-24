import numpy as np
from pyoptsparse import Optimization, OPT
from numba import njit

class Action:

    def __init__(self, params):
        self.name = params.get('name')
        self.data_folder = params['data_folder']

        N_data = params['ndata'] if params['ndata'] > 0 else None
        path_to_stim = self.data_folder+params.get('stim_file') if params.get('stim_file') is not None else None
        self.set_data_fromfile(self.data_folder+params['data_file'],
                               path_to_stim,
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
        beta_array          = np.arange(1, params['max_beta']+1, 1)
        self.Rm             = float(params.get('Rm'))
        self.Rf0            = params.get('Rf0')
        self.Lidx           = params.get('Lidx')
        dt_model            = params.get('dt_model')
        self.optimizer      = params.get('optimizer', 'IPOPT')
        self.opt_options    = params.get('opt_options')
        self.NP             = params.get('num_pars')
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
        self.Rf = self.Rf0 * alpha**beta_array

        self.minpaths = np.zeros((len(beta_array)+1, self.N_model*self.D+self.NP), dtype=np.float32)

        X0 = np.zeros((self.N_model, self.D), dtype = np.float32)
        for i, b in enumerate(self.var_bounds):
            X0[:, i] = np.float32(np.random.uniform(low =b[0], high = b[1], size = self.N_model))
        X0[::self.model_skip, self.Lidx] = self.Y

        P0 = np.zeros(self.NP, dtype = np.float32)
        for i, b in enumerate(self.par_bounds):
            P0[i] = np.float32(np.random.uniform(low = b[0], high = b[1]))

        self.minpaths[0] = np.concatenate((X0.flatten(), P0))


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
        self.t_data = data[nstart:nstart+self.N_data, 0]
        self.dt_data = self.t_data[1] - self.t_data[0]
        self.Y = data[nstart:nstart+self.N_data, 1:]

        if stim_file is not None:
            self.stim = np.load(stim_file)[:self.N_data, 1:] if stim_file.endswith('npy')\
                                           else np.loadtxt(stim_file)[:self.N_data, 1:]
            self.stim = np.squeeze(self.stim.T).astype(np.float32)

        else: self.stim = None



    def min_A(self, id):
        f_mpath = open(self.data_folder+self.name+'_min_paths.txt', 'bw+')
        f_mpars = open(self.data_folder+self.name+'_min_pars.txt', 'bw+')
        f_maction = open(self.data_folder+self.name+'_min_action.txt', 'bw+')
        for i, rf in enumerate(self.Rf):
            self.rf = rf
            self.minpaths[i+1], Amin = self._min_A_step(i)

            path = self.minpaths[i+1][:self.D*self.N_model].reshape(1, -1)
            pars = self.minpaths[i+1][self.D*self.N_model:].reshape(1, -1)

            np.savetxt(f_mpath, path, fmt = '%1.8e')
            f_mpath.write(b"\n")
            np.savetxt(f_maction, [Amin], fmt = '%1.8e')
            f_maction.write(b"\n")
            np.savetxt(f_mpars, pars, fmt = '%1.8e')
            f_mpars.write(b"\n")

            f_mpath.flush()
            f_maction.flush()
            f_mpars.flush()


    ##### PRIVATE FUNCTIONS #############

    def _action(self, xdict):
        x = xdict['x']
        p = xdict['p']
        funcs = {}

        X = np.reshape(x, (self.N_model, self.D))

        diff_m = X[::self.model_skip, self.Lidx] - self.Y
        merr = self.Rm * np.linalg.norm(diff_m)**2
        merr/=(len(self.Lidx) * self.N_data)

        diff_f = X[1:] - self.disc_trapezoid(self.f, X, p, self.stim, self.t_model)
        ferr = self.rf*np.linalg.norm(diff_f)**2
        ferr/=(self.D * (self.N_model - 1))

        funcs['obj'] = merr+ferr

        return funcs, False

    def _grad_action(self, xdict, funcs):
        x = xdict['x']
        p = xdict['p']
        X = np.reshape(x, (self.N_model, self.D))

        funcSens = {}

        diff_m = X[::self.model_skip, self.Lidx] - self.Y

        dmdx = np.zeros((self.N_model, self.D))
        dmdx[::self.model_skip, self.Lidx] = self.Rm*diff_m
        dmdx = dmdx.flatten()
        dmdx/=(len(self.Lidx)/2 * self.N_data)

        diff_f = X[1:] - self.disc_trapezoid(self.f, X, p, self.stim, self.t_model)
        
        dfdx = self._get_dfdx(X, p, self.fjacx, self.N_model, self.D,
                              self.t_model, self.rf, diff_f, self.stim)

        dfdp = self._get_dfdp(X, p, self.fjacp, self.N_model, self.D,
                              self.t_model, self.rf, diff_f, self.stim)
        
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
        return np.concatenate((sol.xStar['x'], sol.xStar['p'])), sol.fStar

    def _min_A_step(self, beta_i):
        self.beta_i = beta_i
        XP0 = self.minpaths[beta_i]
        XPmin, Amin = self._optimize(XP0)
        return XPmin, Amin

    @staticmethod
    @njit
    def disc_trapezoid(f, x, p, stim, t_model):
        """
        Time discretization for the action using the trapezoid rule.
        """
        N_model, D = x.shape
        dt_model = t_model[1] - t_model[0]
        fn = np.zeros((N_model-1, D))
        fnp1 = np.zeros((N_model-1, D))
        if stim == None: stim = [None]*N_model

        for i in range(N_model):
            eval_f = np.array(f(x[i], t_model[i], p, stim[i]))
            if i != N_model-1:
                fn[i] = eval_f
            if i != 0:
                fnp1[i-1] = eval_f

        return x[:-1] + dt_model * (fn + fnp1) / 2

    @staticmethod
    @njit
    def _get_dfdx(X, p, fjacx, N_model, D, t_model, rf, diff_f, stim):
        if stim == None: stim = [None]*N_model
        dt_model = t_model[1] - t_model[0]
        dfdx = np.zeros((N_model, D))
        J = np.zeros((D, D))
        J = fjacx(X[0], t_model[0], p, stim[0])
        dfdx[0] = rf*np.sum(-diff_f[0].reshape(-1, 1)*(np.eye(D) + 0.5*dt_model*J),
                             axis = 0)
        df1 = np.sum(diff_f[0].reshape(-1, 1)*(np.eye(D) - 0.5*dt_model*J),
                     axis = 0)
        for i in range(1, N_model-1):
            Jp1 = fjacx(X[i], t_model[i], p, stim[i])
            df2 = np.sum(-diff_f[i].reshape(-1, 1)*(np.eye(D) + 0.5*dt_model*Jp1),
                         axis = 0)
            dfdx[i] = rf*(df1 + df2)
            J = Jp1
            df1 = df2

        dfdx[-1] = rf*np.sum(diff_f[-1].reshape(-1, 1)*(np.eye(D) - 0.5*dt_model*Jp1),
                             axis = 0)
        return (dfdx/(D/2 * (N_model - 1))).flatten()

    @staticmethod
    @njit
    def _get_dfdp(X, p, fjacp, N_model, D, t_model, rf, diff_f, stim):
        if stim == None: stim = [None]*N_model
        dfdp = np.zeros((N_model, len(p)))
        dt_model = t_model[1] - t_model[0]

        G = fjacp(X[0], t_model[0], p, stim[0])
        for i in range(N_model-1):
            Gp1 = fjacp(X[i+1], t_model[i+1], p, stim[i+1])
            dfdp[i] = rf*np.sum(diff_f[i].reshape(-1, 1)*(G+Gp1), axis = 0)
            G = Gp1
        dfdp = -dt_model*np.sum(dfdp, axis = 0)/(D * (N_model - 1))
        return dfdp