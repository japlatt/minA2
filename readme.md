# Installation Instructions

I've only tried to install IPOPT on a linux operating system.  On windows you can use WSL.

### Python Dependencies
- Python 3.7+ [^1]
- [charm4py](https://charm4py.readthedocs.io/en/latest/install.html)
- [numba](https://numba.readthedocs.io/en/stable/user/installing.html)
- [pyoptsparse](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/install.html#)

Note: pyoptsparse needs to be installed with IPOPT.  This can be done most easily in the 
following way.
1. Install gcc and gfortran (this will be os dependent).  On Linux sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev
2. https://github.com/OpenMDAO/build_pyoptsparse






[^1]: Highly recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage packages.  Will have to install scientific computing packages: numpy, scipy, sympy, matplotlib.