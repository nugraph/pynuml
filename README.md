## Instructions for running branch `phdf5`:

### Set up python environment
* On Cori at NERSC
  + rm -rf .conda/envs/h5pyenv
  + conda create --name h5pyenv --clone lazy-mpi4py
  + conda activate h5pyenv
  + module swap PrgEnv-intel PrgEnv-gnu
  + setenv HDF5_MPI ON
  + setenv CC cc
  + module load cray-hdf5-parallel
  + pip install --no-binary=h5py h5py
  + pip3 install torch
  + pip3 install numpy
  + pip install torch-scatter
  + pip install torch-sparse
  + pip install torch-geometric

* On a local Linux machine
  + virtualenv --system-site-packages -p python3 ~/venv
  + source ~/venv/bin/activate.csh
  + wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.1.1.tar.gz
  + gzip -dc mpi4py-3.1.1.tar.gz |tar -xf -
  + cd mpi4py-3.1.1
  + python3 setup.py build
  + python3 setup.py install
  + pip3 install pandas
  + pip3 install h5py
  + pip3 install boost_histogram
  + pip3 install torch
  + pip3 install numpy
  + pip3 install torch-scatter
  + pip3 install torch-sparse
  + pip3 install torch-geometric

### Run commands
* On Cori at NERSC
  + Here is the batch script file that runs 128 MPI processes on 2 KNL nodes
    ```
    % cat sbatch.sh 
    #!/bin/bash -l
    #SBATCH -t 00:20:00
    #SBATCH --nodes=2
    #SBATCH -o qout.128.%j
    #SBATCH -e qout.128.%j
    #SBATCH --constraint=knl,quad,cache
    #SBATCH -L SCRATCH
    #SBATCH --qos=debug

    NP=128
    cd $PWD
    export OMP_NUM_THREADS=1
    export KMP_AFFINITY=disabled
    conda activate h5pyenv

    srun -n $NP python3 example/process.py -p \
         -i $SCRATCH/uboone/numu_slice_89_seq_cnt_seq.h5 \
         -o $SCRATCH/uboone_out
    ````
* On a local Linux machine
  + To run on 8 MPI processes:
  ```
  source ~/venv/bin/activate.csh
  mpiexec -l -n 8 python example/process.py -p \
          -i /scratch/x0123_seq_cnt_seq.h5 \
          -o /scratch/output
  ```

* Example output:
  ```
  Processing input file: /global/cscratch1/sd/wkliao/uboone/numu_slice_89_seq_cnt_seq.h5
  Output folder: /global/cscratch1/sd/wkliao/uboone_out
  Size of event_table/event_id is  574174
  ------------------------------------------------------------------
  Use event_id.seq_cnt as graph IDs
  read seq    time MAX=    0.42  MIN=    0.00
  bin search  time MAX=    2.33  MIN=    0.00
  scatter     time MAX=    0.00  MIN=    0.00
  scatterV    time MAX=    0.09  MIN=    0.00
  read remain time MAX=    3.30  MIN=    2.57
  ------------------------------------------------------------------
  Number of MPI processes =  128
  Total number of graphs =  537165
  Local number of graphs MAX= 4503     MIN= 3879
  Local graph size       MAX=  147.13  MIN=  124.57 (MiB)
  ------------------------------------------------------------------
  read from file  time MAX=    8.25  MIN=    7.97
  build dataframe time MAX=   60.50  MIN=   36.03
  graph creation  time MAX=  652.59  MIN=  552.08
  write to files  time MAX=   75.78  MIN=   64.41
  total           time MAX=  791.44  MIN=  686.86
  (MAX and MIN timings are among 128 processes)
  ------------------------------------------------------------------
  edep grouping   time MAX=   25.04  MIN=   21.22
  edep merge      time MAX=   55.37  MIN=   48.16
  label           time MAX=  342.62  MIN=  269.74
  hit_table merge time MAX=   31.40  MIN=   26.50
  plane build     time MAX=   61.19  MIN=   51.77
  torch           time MAX=   26.93  MIN=   47.78
  knn             time MAX=   26.93  MIN=   47.78
  ```
