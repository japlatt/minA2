import numpy as np

class action:

    def __init__(self, f, fjacx, fjacp):
        self.f = f
        self.fjacx = fjacx
        self.fjacp = fjacp


    def set_data_fromfile(self, data_file, stim_file=None, nstart=0, N=None):
        """
        Load data & stimulus time series from file.
        If data is a text file, must be in multi-column format with D+1 columns:
            t  y_1  y_2  ...  y_L
        If a .npy archive, should contain an Nx(D+1) array with times in the
        zeroth element of each entry.
        Column/array formats should also be in the form t  s_1  s_2 ...
        """
        if data_file.endswith('npy'):
            data = np.load(data_file)
        else:
            data = np.loadtxt(data_file)

        self.t_data = data[:, 0]
        self.dt_data = self.t_data[1] - self.t_data[0]
        self.Y = data[:, 1:]

        if stim_file is not None:
            if stim_file.endswith('npy'):
                s = np.load(stim_file)
            else:
                s = np.loadtxt(stim_file)
            self.stim = s[:, 1:]
        else:
            self.stim = None

        self.dt_data = dt_data



    def anneal_init(self, X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, dt_model=None,
                init_to_data=True, action='A_gaussian', disc='trapezoid',
                method='L-BFGS-B', bounds=None, opt_args=None, adolcID=0)



    
    

