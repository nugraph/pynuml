import os.path as osp, torch, h5py
from mpi4py import MPI

class PTOut:
  def __init__(self, outdir):
    self.outdir = outdir

  def save(self, obj, name):
    torch.save(obj, osp.join(self.outdir, name)+".pt")

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
      # self.f.create_dataset(f"/{name}/{key}", data=val, compression="gzip")
      self.f.create_dataset(f"/{name}/{key}", data=val)

  def __del__(self):
    self.f.close()

