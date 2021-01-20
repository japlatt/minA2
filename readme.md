# Installation Instructions

### Python Dependencies
- Python 3.7+ <sup>[1]</sup>:
- [charm4py](https://charm4py.readthedocs.io/en/latest/install.html)
- [numba](https://numba.readthedocs.io/en/stable/user/installing.html)
- [cyipopt](https://github.com/mechmotum/cyipopt)
- [seaborn](https://seaborn.pydata.org/installing.html) for some plotting purposes

The simplest way to install these packages is to use anaconda using the following commands: 
1. conda install numba
2. conda install -c conda-forge cyipopt
3. conda install seaborn
4. conda install -c conda-forge greenlet
5. conda install -c anaconda cython
6. pip3 install charm4py
7. conda install -c anaconda pyyaml

For those who wish to use the commercial solver SNOPT, which is much more powerful than ipopt
1. https://ccom.ucsd.edu/~optimizers
2. https://github.com/snopt/snopt-python

### Troubleshoot Installation
Note: charm4py requires c++11 (gcc > 4.8.1, 2011 release) support in order to compile as well as ld compiled with --sysroot enabled.  For those on the boom cluster this can present an issue because the compiler is (at the time of writing) over 10 years out of date.  Of course you can install [gcc](https://superuser.com/questions/986949/upgrading-gcc-for-a-specific-user-account) and [dependencies](https://gcc.gnu.org/wiki/InstallingGCC) and [binutils](https://www.gnu.org/software/binutils/) from source yourself.  If you have root access then obviously using the package manager is by the simplest options.  Without root access my workaround involved using conda to do the installation:
1. conda install gcc_linux-64
2. conda install gxx_linux-64
3. conda install gfortran_linux-64
4. ln -s /path/to/anaconda/bin/x86_64-conda_cos6-linux-gnu-gcc /path/to/anaconda/bin/gcc<sup>[2]</sup>
5. ln -s /path/to/anaconda/bin/x86_64-conda_cos6-linux-gnu-g++ /path/to/anaconda/bin/g++
6. In ~/.bash_profile add "export PATH=/path/to/anaconda/x86_64-conda_cos6-linux-gnu/sysroot/usr/bin:/path/to/anaconda/bin:/path/to/anaconda/compiler_compat:$PATH"
7. Now run pip3 install charm4py<sup>[3]</sup>

Note: SNOPT requires a version of glibc higher than 2.14

For glibc install if you don't have root priveledges follow these [instructions](https://unix.stackexchange.com/questions/176489/how-to-update-glibc-to-2-14-in-centos-6-5)
-install a glibc >= 2.14
-only compiles if compilers are of similar generation
-add path to libc.so.6 to LD_LIBRARY_PATH

### Workflow
Follow these steps to get minA2 to work with your program (tldr modify: specs, params, def_dyn, run_da):
1. go to def_dyn and define the dynamics of the model.  These must be defined using sympy functions and return a Matrix.
2. Create a specs.yaml and params.txt file.  These files can reside anywhere you just have to specify the path to them in run_da.py.  The specs.yaml file should have the fields: name, num_data, start_data, time_step, num_dims, num_par, obs_dim, RM, RF0, alpha, max_beta, data_folder, data_file.  The params file should be a list of bounds in format "lower, upper" for each of the state variables (first) and the parameters (second).  See the example for more details.
3. If running a twin experiment set the options in generate_data.py and then execute it to create data.  Be sure to have a stim file if needed.
4. If not running a twin experiment then set data with the filename and location specified in the specs file
5. Finally call run_da.py using the charm4py initialization.  This will output the file name_results.npz in the data_folder specified.
6. The script analyze_results.py will take the path_to_specs specified in run_da.py and analyze the data found in the data folder.  You should call it with the signature python analyze_results.py -np "num_pred" to have the script predict forward in time and compare.

### Specs.yaml details
name             : name for saving files (str)
num_data         : number of data steps (int)
start_data       : start index for data (int)
time_step        : time steps of model per time step of data (int >= 1)
num_dims         : number of state variables in model (int)
num_par          : number of parameters in the model to estimate (int)
obs_dim          : observed dimensions ex: [0], [0, 3, 5] or -1 means all observed (list or -1)
RM               : measurement error value, inverse of variance of measurement noise (list or int or float)
RF0              : model error init cond (list or int or float)
alpha            : Rf = RF0 * alpha**beta (float or int)
max_beta         : beta = 1,2,3...,max_beta (int)
data_folder      : path to data folder, can be relative from run_da.py (str)
data_file        : name (not path) of data file in data folder, generate data creates file with this name (str ending in npy or txt)
stim_file        : name (not path) of stim file in data folder, if none comment out (str ending in npy or txt)

Note: scientific notation must have decimal point and sign, ex: 1.e-3, 1.2e+4



<a>1</a>: Highly recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage packages.  Will have to have standard packages: numpy, scipy, sympy, matplotlib, pyyaml

<a>2</a>: Replace "/path/to/anaconda/" with the path to your anaconda/miniconda distribution.  For me that was "/home/jplatt/miniconda3/""

<a>3</a>: I had a weird error when compiling charm4py that seems to be an error in the conda version of gcc.  If you get this same error abour redefining align_alloc then go to vim /path/to/anaconda/x86_64-conda_cos6-linux-gnu/sysroot/usr/include/stdlib.h and comment out the whole function "static inline void* aligned_alloc (size_t al, size_t sz)".  Lines 513-523.