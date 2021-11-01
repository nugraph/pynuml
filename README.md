## Instructions for running branch `phdf5`:

### Set up python environment
* On Cori at NERSC
  + module load python
  + conda init `   <--- run this only when first time use of conda`
  + conda env remove --name h5pyenv `   <--- delete conda environment h5pyenv if exists`
  + conda create --name h5pyenv python=3.8 `    <--- install under user's home folder`
    * conda activate h5pyenv
  + conda create --prefix /global/common/software/myproject/h5pyenv python=3.8 `    <--- install under Cori global common folder`
    * conda activate /global/common/software/myproject/h5pyenv
    * replace 'myproject' in the above 2 commands with your project ID
  + module swap PrgEnv-intel PrgEnv-gnu
  + conda install -c defaults --override-channels numpy
  + setenv MPICC "cc -shared" `   <--- for bash, export MPICC="cc -shared"`
  + pip install --no-binary=mpi4py mpi4py
  + setenv CC cc  `   <--- for bash, export CC=cc`
  + setenv HDF5_MPI ON  `   <--- for bash, export HDF5_MPI=ON`
  + module load cray-hdf5-parallel
  + pip install --no-deps --no-binary=h5py h5py
  + pip install torch
  + pip install torch-scatter torch-sparse torch-geometric
  + pip install particle
  + pip list  `   <--- to show the installed packages`
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

* Example output from a 128-process run on two Cori KNL nodes:
  ```
  Processing input file: /global/cscratch1/sd/wkliao/uboone/numu_slice_89_seq_cnt_seq.h5
  Output file: /global/cscratch1/sd/wkliao/uboone_out/numu_slice.0000.h5
  ---- Timing break down of the file read phase (in seconds) -------
  Use event_id.seq_cnt to calculate subarray boundaries
  calc boundaries time MAX=    2.72  MIN=    2.23
  read datasets   time MAX=    2.91  MIN=    2.32
  (MAX and MIN timings are among 128 processes)
  ---- Timing break down of graph creation phase (in seconds) ------
  edep grouping   time MAX=   26.29  MIN=   20.22
  edep merge      time MAX=   59.04  MIN=   45.87
  labelling       time MAX=  312.70  MIN=  212.02
  hit_table merge time MAX=   32.94  MIN=   25.03
  plane build     time MAX=   63.98  MIN=   49.40
  torch_geometric time MAX=   28.84  MIN=   22.51
  edge knn        time MAX=   53.81  MIN=   41.42
  (MAX and MIN timings are among 128 processes)
  ------------------------------------------------------------------
  Number of MPI processes          =     128
  Total no. event IDs              =  574174
  Total no. non-empty events       =  245891
  Size of all events               =   16952.1 MiB
  Local no. events assigned     MAX=    2012   MIN=    1832   AVG=    1921.0
  Local indiv event size in KiB MAX=    1077.9 MIN=       3.0 AVG=      70.6
  Local sum   event size in MiB MAX=     147.1 MIN=     124.6 AVG=     132.4
  Total no.  of graphs             =  537165
  Size of all graphs               =   27008.8 MiB
  Local no. graphs created      MAX=    4503   MIN=    3879   AVG=    4196.6
  Local indiv graph size in KiB MAX=     629.8 MIN=       6.3 AVG=      51.5
  Local sum   graph size in MiB MAX=     231.4 MIN=     193.1 AVG=     211.0
  (MAX and MIN timings are among 128 processes)
  ---- Top-level timing breakdown (in seconds) ---------------------
  read from file  time MAX=    5.61  MIN=    5.41
  build dataframe time MAX=   56.63  MIN=   35.95
  graph creation  time MAX=  634.71  MIN=  458.93
  write to files  time MAX=   94.64  MIN=   73.58
  total           time MAX=  787.78  MIN=  594.78
  (MAX and MIN timings are among 128 processes)
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
* Input file: /global/cscratch1/sd/wkliao/uboone/numu_slice_89_seq_cnt_seq.h5
* Input file's Lustre striping setting: striping count 32 and striping size 1 MiB
* Input file size: 13.74 GiB
* Output file folder's Lustre striping setting: striping count 1 and striping size 1 MiB
* Output file size: 12.45 GiB (HDF5 files with compression enabled)

   | No. MPI processes | 64      | 128    | 256    | 512    | 1024   |
   | ----------------- |--------:|-------:|-------:|-------:|-------:|
   | read from file    |    8.09 |   5.61 |   5.07 |   4.59 |  14.66 |
   | build dataframes  |  112.07 |  56.63 |  28.92 |  15.13 |   7.47 |
   | graph creation    | 1220.07 | 634.71 | 333.52 | 170.40 |  87.62 |
   | write to files    |  179.51 |  94.64 |  54.39 |  14.47 |   7.31 |
   | total             | 1519.54 | 787.78 | 412.49 | 204.60 | 116.78 |

* Input data statistics:
  + Number of event IDs: 574174
  + Number of non-empty events: 245891
  + Total size of all event data: 16.55 GiB
  + Max event data size: 1077.9 KiB
  + Min event data size:    3.0 KiB
* Output data statistics:
  + Number of graphs: 537165
  + Total size of all graphs: 26.38 GiB
  + Max graph size: 629.8 KiB
  + Min graph size:   6.3 KiB

