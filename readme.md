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

Note: charm4py requires c++11 (gcc > 4.8.1, 2011 release) support in order to compile as well as ld compiled with --sysroot enabled.  For those on the boom cluster this can present an issue because the compiler is (at the time of writing) over 10 years out of date.  Of course you can install [gcc](https://superuser.com/questions/986949/upgrading-gcc-for-a-specific-user-account) and [dependencies](https://gcc.gnu.org/wiki/InstallingGCC) and [binutils](https://www.gnu.org/software/binutils/) from source yourself.  If you have root access then obviously using the package manager is by the simplest options.  Without root access my workaround involved using conda to do the installation:
1. conda install gcc_linux-64
2. conda install gxx_linux-64
3. conda install gfortran_linux-64
4. ln -s /path/to/anaconda/bin/x86_64-conda_cos6-linux-gnu-gcc /path/to/anaconda/bin/gcc<sup>[2]</sup>
5. ln -s /path/to/anaconda/bin/x86_64-conda_cos6-linux-gnu-g++ /path/to/anaconda/bin/g++
6. In ~/.bash_profile add "export PATH=/path/to/anaconda/x86_64-conda_cos6-linux-gnu/sysroot/usr/bin:/path/to/anaconda/bin:/path/to/anaconda/compiler_compat:$PATH"
7. Now run pip3 install charm4py<sup>[3]</sup>

<a>1</a>: Highly recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage packages.  Will have to have standard packages: numpy, scipy, sympy, matplotlib, yaml

<a>2</a>: Replace "/path/to/anaconda/" with the path to your anaconda/miniconda distribution.  For me that was "/home/jplatt/miniconda3/""

<a>3</a>: I had a weird error when compiling charm4py that seems to be an error in the conda version of gcc.  If you get this same error abour redefining align_alloc then go to vim /path/to/anaconda/x86_64-conda_cos6-linux-gnu/sysroot/usr/include/stdlib.h and comment out the whole function "static inline void* aligned_alloc (size_t al, size_t sz)".  Lines 513-523.