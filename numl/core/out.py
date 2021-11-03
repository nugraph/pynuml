import os, torch, h5py, sys
from mpi4py import MPI

class PTOut:
  def __init__(self, outdir):
    self.outdir = outdir
    isExist = os.path.exists(outdir)
    if not isExist:
      rank = MPI.COMM_WORLD.Get_rank()
      if rank == 0:
        print("Error: output directory does not exist",outdir)
      sys.stdout.flush()
      MPI.COMM_WORLD.Abort(1)

  def save(self, obj, name):
    torch.save(obj, os.join(self.outdir, name)+".pt")

  def exists(self, name):
    return osp.exists(osp.join(self.outdir, name)+".pt")

class H5Out:
  def __init__(self, fname):
    # This implements one-file-per-process I/O strategy.
    # append MPI process rank to the output file name
    rank = MPI.COMM_WORLD.Get_rank()
    file_ext = ".{:04d}.h5"
    self.fname = fname + file_ext.format(rank)
    # open/create the HDF5 file
    self.f = h5py.File(self.fname, "w")

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
    self.f.close()

