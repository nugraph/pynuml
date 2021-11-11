## Set up the python environment

### On Cori at NERSC
  + module load python
  + conda init `<--- run this only when first time use of conda`
  + To install under user's home folder:
    * conda env remove --name h5pyenv
    * conda create --name h5pyenv python=3.8
    * conda activate h5pyenv
  + To install under Cori global common project folder:
    * conda remove --name --name /global/common/software/myproject/h5pyenv
    * conda create --prefix /global/common/software/myproject/h5pyenv python=3.8
    * conda activate /global/common/software/myproject/h5pyenv
    * (replace 'myproject' in the above 3 commands with your project ID)
  + module swap PrgEnv-intel PrgEnv-gnu
  + setenv MPICC "cc -shared" `<--- for bash, export MPICC="cc -shared"`
  + pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
  + setenv CC cc  `<--- for bash, export CC=cc`
  + setenv HDF5_MPI ON  `<--- for bash, export HDF5_MPI=ON`
  + module load cray-hdf5-parallel
  + pip install --force --no-cache-dir --no-deps --no-binary=h5py h5py
  + pip install --force --no-cache-dir torch
  + pip install --force --no-cache-dir torch-scatter torch-sparse torch-geometric
  + pip install --force --no-cache-dir particle
  + pip list  `<--- to show the installed packages`
  + See more information in [Python User Guide](https://docs.nersc.gov/development/languages/python/nersc-python) and [Parallelism in Python](https://docs.nersc.gov/development/languages/python/parallel-python) at NERSC.

### On a local Linux machine
  + Install MPICH
    ```
    wget http://www.mpich.org/static/downloads/3.4.2/mpich-3.4.2.tar.gz
    gzip -dc mpich-3.4.2.tar.gz | tar -xf -
    cd mpich-3.4.2
    ./configure --prefix=$HOME/MPICH/3.4.2 \
                --with-device=ch3 \
                --enable-romio \
                --with-file-system=ufs \
                CC=gcc FC=gfortran
    make -j4 install
    ```
  + Install HDF5 with parallel feature enabled
    * download HDF5 source codes from https://www.hdfgroup.org/downloads/hdf5/source-code/
    ```
    gzip -dc hdf5-1.12.1.tar.gz | tar -xf -
    cd hdf5-1.12.1
    ./configure --prefix=$HOME/HDF5/1.12.1 \
                --disable-fortran --disable-cxx \
                CC=$HOME/MPICH/3.4.2/bin/mpicc \
                --enable-parallel=yes
    make -j4 install
    ```
  + virtualenv --system-site-packages -p python3 ~/venv
  + source ~/venv/bin/activate.csh
  + wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.1.1.tar.gz
  + gzip -dc mpi4py-3.1.1.tar.gz |tar -xf -
  + cd mpi4py-3.1.1
  + python3 setup.py build
  + python3 setup.py install
  + setenv HDF5_MPI ON
  + setenv HDF5_DIR $HOME/HDF5/1.12.1
  + setenv CC $HOME/MPICH/3.4.2/bin/mpicc
  + pip install --no-binary=h5py h5py
  + pip install pandas
  + pip install boost_histogram
  + pip install torch
  + pip install torch-scatter torch-sparse torch-geometric
  + pip install particle

