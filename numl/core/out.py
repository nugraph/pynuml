import os.path as osp, torch, h5py

class PTOut:
  def __init__(self, outdir):
    self.outdir = outdir

  def save(self, obj, name):
    torch.save(obj, osp.join(self.outdir, name)+".pt")

class H5Out:
  def __init__(self, fname):
    self.f = h5py.File(fname, "w")

  def save(self, obj, name):
    for key, val in obj.items():
      self.f.create_dataset(f"/{name}/{key}", data=val, compression="gzip")
#    self.f.create_dataset(name, data=obj, compression="gzip")

  # do we need this?
  # def __del__(self):
  #   self.f.close()

