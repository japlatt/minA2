import numpy as np

class action:

    def __init__(self, params):
        self.action_init(params)


    def action_init(self, params):
        '''
        Args:
            X0              :: time dependent state variables
            P0              :: time independent parameters to be estimated
            alpha
            beta_array
            RM
            RF0
            Lidx            :: indicies of measured vars         
            bounds
            optimization
        '''
        self.f              = params.get('f')
        self.fjac           = params.get('fjac')
        self.fhess          = params.get('fhess')
        self.P0             = params.get('P0')
        alpha               = params.get('alpha')
        self.Rm             = params.get('Rm')
        self.Rf0            = params.get('Rf0')
        self.Lidx           = params.get('Lidx')
        self.bound          = params.get('bounds')
        self.dt_model       = params.get('dt_model')
        self.optimization   = params.get('optimization', 'IPOPT')

        if dt_model is None:
            self.dt_model = self.dt_data
            self.N_model = self.N_data
            self.t_model = np.copy(self.t_data)
        else:
            assert(self.dt_model < self.dt_data and np.isclose(self.dt_data % self.dt_model,0))
            self.dt_model = dt_model
            self.N_model = (self.N_data - 1) * int(self.dt_data / self.dt_model) + 1
            self.t_model = np.linspace(self.t_data[0], self.t_data[-1], self.N_model, dtype = np.float32)
            self.stim = np.interp(self.t_model, self.t_data, self.stim).astype(np.float32)

        self.model_skip = int(self.dt_data / self.dt_model)
        self.NP = len(self.P0)

        self.D = len(X0)
        self.Rf = RF0 * alpha**beta_array

        self.minpaths = np.zeros((len(beta_array), self.N_model*self.D+self.NP), dtype=np.float32)

        self.X0 = np.zeros((self.N_model, self.D), dtype = np.float32)
        for i, b in enumerate(bounds):
            self.X0[:, i] = np.float32(np.random.uniform(low =9 b[0], high = b[1], size = self.N_model))
            self.X0[::self.model_skip, Lidx] = self.Y

        self.minpaths[0] = np.concatenate(self.X0.flatten(), P0)


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


    def min_A_step(self):




             























    
    

