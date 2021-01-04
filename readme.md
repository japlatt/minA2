# Installation Instructions

### Python Dependencies
- Python 3.7+ [^1]
- [charm4py](https://charm4py.readthedocs.io/en/latest/install.html)
- [numba](https://numba.readthedocs.io/en/stable/user/installing.html)
- [cyipopt](https://github.com/mechmotum/cyipopt)
- [seaborn](https://seaborn.pydata.org/installing.html) for some plotting purposes

The simplest way to install these packages is using the following commands: 
1. conda install numba
2. conda install -c conda-forge cyipopt
3. conda install seaborn
4. conda install -c conda-forge greenlet
5. conda install -c anaconda cython
5. pip3 install charm4py

If you have an already installed an optimized version of ipopt on your machine you can compile and install cyipopt and then add the following lines to your ~/.bashrc file:
1. export IPOPT_DIR=path/to/ipopt/dir
2. export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$IPOPT_DIR/build/lib

[^1]: Highly recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage packages.  Will have to install scientific computing packages: numpy, scipy, sympy, matplotlib