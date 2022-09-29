import os
import sys
import h5py
from mpi4py import MPI
from ..util import requires_torch

class PTOut:
    def __init__(self, outdir):
        requires_torch()
        self.outdir = outdir
        isExist = os.path.exists(outdir)
        if not isExist:
            rank = MPI.COMM_WORLD.Get_rank()
            if rank == 0:
                print("Error: output directory does not exist", outdir)
            sys.stdout.flush()
            MPI.COMM_WORLD.Abort(1)

    def save(self, obj, name):
        torch.save(obj, os.path.join(self.outdir, name)+".pt")

    def exists(self, name):
        return os.path.exists(os.path.join(self.outdir, name)+".pt")

class H5Out:
    def __init__(self, fname, overwrite=False):
        requires_torch()
        # This implements one-file-per-process I/O strategy.
        # append MPI process rank to the output file name
        rank = MPI.COMM_WORLD.Get_rank()
        file_ext = ".{:04d}.h5"
        self.fname = fname + file_ext.format(rank)
        if os.path.exists(self.fname):
            if overwrite:
                os.remove(self.fname)
            else:
                print(f"Error: file already exists: {self.fname}")
                sys.stdout.flush()
                MPI.COMM_WORLD.Abort(1)
        # open/create the HDF5 file
        self.f = h5py.File(self.fname, "w")
        # print(f"{rank}: creating {self.fname}")
        # sys.stdout.flush()

    def save(self, obj, name):
        for key, val in obj:
            # set chunk sizes to val shape, so there is only one chunk per dataset
            # if isinstance(val, torch.Tensor) and val.nelement() == 0 :
            #   print("zero val ",name,"/",key," shape=",val.shape)
            if isinstance(val, torch.Tensor) and val.nelement() > 0 :
                # Note compressed datasets can only be read/written in MPI collective I/O mode in HDF5
                self.f.create_dataset(f"/{name}/{key}", data=val, chunks=val.shape, compression="gzip")
                # The line below is to not enable chunking/compression
                # self.f.create_dataset(f"/{name}/{key}", data=val)
            else:
                # if data is not a tensor or is empty, then disable chunking/compression
                self.f.create_dataset(f"/{name}/{key}", data=val)

    def __del__(self):
        if self.f != None: self.f.close()

