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

Note: charm4py requires c++11 (gcc > 4.8.1) support in order to compile as well as ld compiled with --sysroot enabled.  For those on the boom cluster this can present an issue because the compiler may not be up to date.  Of course you can install gcc and binutils from source yourself.  My workaround involved the following:
1. conda install gcc_linux-64
2. conda install gxx_linux-64
3. conda install gfortran_linux-64
4. ln -s /path/to/anaconda/bin/x86_64-conda_cos6-linux-gnu-gcc /path/to/anaconda/bin/gcc
5. ln -s /path/to/anaconda/bin/x86_64-conda_cos6-linux-gnu-g++ /path/to/anaconda/bin/g++
6. In ~/.bash_profile add "export PATH=/path/to/anaconda/x86_64-conda_cos6-linux-gnu/sysroot/usr/bin:/path/to/anaconda/bin:/path/to/anaconda/compiler_compat:$PATH"
7. Now run pip3 install charm4py[^2]

[^1]: Highly recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage packages.  Will have to have standard packages: numpy, scipy, sympy, matplotlib, yaml
[^2]: I had a weird error when compiling charm4py that seems to be an error in the conda version of gcc.  If you get this same error abour redefining align_alloc then go to vim /path/to/anaconda/x86_64-conda_cos6-linux-gnu/sysroot/usr/include/stdlib.h and comment out the whole function "static inline void* aligned_alloc (size_t al, size_t sz)".  Lines 513-523.