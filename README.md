## Instructions for running branch `phdf5`:

### Set up python environment
* On Cori at NERSC
  + rm -rf .conda/envs/h5pyenv
  + module load python
  + conda init    <--- run this only when first time use of conda
  + conda create --name h5pyenv --clone lazy-mpi4py
  + conda activate h5pyenv
  + module swap PrgEnv-intel PrgEnv-gnu
  + setenv HDF5_MPI ON
  + setenv CC cc
  + module load cray-hdf5-parallel
  + pip install --no-binary=h5py h5py
  + pip install torch
  + pip install numpy
  + pip install torch-scatter
  + pip install torch-sparse
  + pip install torch-geometric
  + See more information in [Python User Guide](https://docs.nersc.gov/development/languages/python/nersc-python) and [Parallelism in Python](https://docs.nersc.gov/development/languages/python/parallel-python) at NERSC.

* On a local Linux machine
  + Install MPICH
  + Install HDF5 with parallel feature enabled
  + virtualenv --system-site-packages -p python3 ~/venv
  + source ~/venv/bin/activate.csh
  + wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.1.1.tar.gz
  + gzip -dc mpi4py-3.1.1.tar.gz |tar -xf -
  + cd mpi4py-3.1.1
  + python3 setup.py build
  + python3 setup.py install
  + setenv HDF5_MPI ON
  + setenv HDF5_DIR /hdf5/install/path
  + setenv CC mpicc
  + pip install --no-binary=h5py h5py
  + pip install pandas
  + pip install boost_histogram
  + pip install torch
  + pip install numpy
  + pip install torch-scatter
  + pip install torch-sparse
  + pip install torch-geometric

### Run commands
* On Cori at NERSC
  + Concatenate multiple HDF5 files into a single file.
    * Clone and build `ph5cacat`
      ```
      git clone https://github.com/NU-CUCIS/ph5concat
      cd ph5concat
      module load cray-hdf5-parallel
      ./configure --with-hdf5=$HDF5_DIR
      make
      ```
    * Run concatenate program `ph5_concat` in parallel on Cori compute nodes.
      Below gives an example batch script file that allocates 64 MPI processes
      on 4 Haswell nodes to concatenate 89 uboone files. Input file `numu_slice.txt`
      contains a list of file path names of the 89 uboone files.
      ```
      % cat sbatch.sh
      #!/bin/bash -l
      #SBATCH -A m3253
      #SBATCH -t 00:20:00
      #SBATCH --nodes=4
      #SBATCH --tasks-per-node=16
      #SBATCH --constraint=haswell
      #SBATCH -o qout.64.%j
      #SBATCH -e qout.64.%j
      #SBATCH -L SCRATCH
      #SBATCH --qos=debug

      NP=64
      cd $PWD

      srun -n $NP ./ph5_concat -i numu_slice.txt -o $SCRATCH/FS_1M_64/numu_slice_89_seq_cnt_seq.h5
      ````
    * Then, go the directory `utils` and run the utility program `utils/add_key`
      to add the partition key datasets to the concatenated file.
      ````
      ./add_key -k /event_table/event_id -c $SCRATCH/FS_1M_64/numu_slice_89_seq_cnt_seq.h5
      ````
  + A copy of the concatenated uboone file with partition key added is
    available on Cori. The file size is 14 GB.
    ```
    /global/cscratch1/sd/wkliao/uboone/numu_slice_89_seq_cnt_seq.h5
    ```
  + Below is the batch script file for generating the graph files. It allocates
    128 MPI processes on 2 KNL nodes. The output files are in HDF5 format and
    there are 128 files, one per MPI process.
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

    srun -n $NP python3 example/process.py -p -5 \
         -i $SCRATCH/uboone/numu_slice_89_seq_cnt_seq.h5 \
         -o $SCRATCH/uboone_out/numu_slice
    ````
* On a local Linux machine
  + To run on 8 MPI processes:
  ```
  source ~/venv/bin/activate.csh
  mpiexec -l -n 8 python example/process.py -p -5 \
          -i /scratch/x0123_seq_cnt_seq.h5 \
          -o /scratch/output/x0123
  ```

* Example output from a 128-process run on Cori:
  ```
  Processing input file: /global/cscratch1/sd/wkliao/uboone/numu_slice_89_seq_cnt_seq.h5
  Output file: /global/cscratch1/sd/wkliao/uboone_out/numu_slice.0000.h5
  Size of event_table/event_id is  574174
  ------------------------------------------------------------------
  Use event_id.seq_cnt as graph IDs
  read seq    time MAX=    0.39  MIN=    0.00
  bin search  time MAX=    2.34  MIN=    0.00
  scatter     time MAX=    0.00  MIN=    0.00
  scatterV    time MAX=    0.11  MIN=    0.00
  read remain time MAX=    2.99  MIN=    2.48
  ------------------------------------------------------------------
  Number of MPI processes =  128
  Total number of graphs =  537165
  Local number of graphs MAX= 4503     MIN= 3879
  Local graph size       MAX=  147.13  MIN=  124.57 (MiB)
  ------------------------------------------------------------------
  read from file  time MAX=    6.03  MIN=    5.82
  build dataframe time MAX=   56.54  MIN=   35.19
  graph creation  time MAX=  666.21  MIN=  458.51
  write to files  time MAX=   65.39  MIN=   49.37
  total           time MAX=  794.20  MIN=  572.70
  (MAX and MIN timings are among 128 processes)
  ------------------------------------------------------------------
  edep grouping   time MAX=   26.06  MIN=   19.69
  edep merge      time MAX=   59.00  MIN=   44.30
  label           time MAX=  334.41  MIN=  205.81
  hit_table merge time MAX=   33.98  MIN=   23.98
  plane build     time MAX=   66.00  MIN=   48.38
  torch           time MAX=   29.12  MIN=   21.39
  knn             time MAX=   62.69  MIN=   45.48
  ```
* Merge multiple output HDF5 files into one
  + When command-line option '-5' is used, the number of output HDF5 files is
    equal to the number of MPI processes used to run 'example/process.py'.
  + The utility program 'h5merge.py' can be used to merge the output files into
    a single HDF5, by appending one file after another.
  + 'h5merge.py' takes command-line option '-i input_file' and '-o output.h5'
    where 'input_file' is a text file containing the names of files to be
    merged, one file name per line.
  + Note this utility program requires the HDF5 utility program 'h5copy'. Make
    sure 'h5copy' is available under PATH environment variable. For instance,
    by running command 'module load hdf5'
  + Below is an example run command:
    ```
    % cat in_file_names.txt
    x1_graphs.0000.h5
    x1_graphs.0001.h5
    x1_graphs.0002.h5
    x1_graphs.0003.h5

    % python h5merge.py -i in_file_names.txt -o output.h5
    input file name = in_file_names.txt
    output file name = output.h5
    copy file: x1_graphs.0000.h5
    copy file: x1_graphs.0001.h5
    copy file: x1_graphs.0002.h5
    copy file: x1_graphs.0003.h5
    ```

### Performance timing breakdowns
* Cori KNL nodes at NERSC, 64 MPI processes per node, time in seconds.

   | No. MPI processes | 64      | 128    | 256    | 512    | 1024   |
   | ----------------- |--------:|-------:|-------:|-------:|-------:|
   | read from file    |    8.62 |   6.01 |   4.85 |   4.59 |  14.66 |
   | build dataframes  |  100.45 |  51.02 |  28.02 |  15.13 |   7.47 |
   | graph creation    | 1146.41 | 589.48 | 352.56 | 170.40 |  87.62 |
   | write to files    |  105.47 |  53.72 |  39.15 |  14.47 |   7.31 |
   | total             | 1359.55 | 699.85 | 422.17 | 204.60 | 116.78 |

