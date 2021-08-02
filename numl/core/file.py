import h5py, numpy as np, pandas as pd

class NuMLFile:
  def __init__(self, file):
    self._filename = file
    self._colmap = {
      "event_table": {
        "nu_dir": [ "nu_dir_x", "nu_dir_y", "nu_dir_z" ],
      },
      "particle_table": {
        "start_position": [ "start_position_x", "start_position_y", "start_position_z" ],
        "end_position": [ "end_position_x", "end_position_y", "end_position_z" ],
      },
      "hit_table": {},
      "spacepoint_table": {
        "hit_id": [ "hit_id_u", "hit_id_v", "hit_id_y" ],
        "position": [ "position_x", "position_y", "position_z" ],
      },
      "edep_table": {}
    }
    self._groups = []

  def __len__(self):
    with h5py.File(self._filename, "r") as f:
      return f["event_table/event_id"].shape[0]

  def __str__(self):
    with h5py.File(self._filename, "r") as f:
      ret = ""
      for k1 in self._file.keys():
        ret += f"{k1}:\n"
        for k2 in self._file[k1].keys(): ret += f"  {k2}"
        ret += "\n"
      return ret

  def add_group(self, group, keys=[]):
    self._groups.append([ group, keys ])

  def keys(self):
    with h5py.File(self._filename, "r") as f:
      return f.keys()

  def _cols(self, group, key):
    if key == "event_id": return [ "run", "subrun", "event" ]
    if key in self._colmap[group].keys(): return self._colmap[group][key]
    else: return [key]

  def get_dataframe(self, group, keys=[]):
    with h5py.File(self._filename, "r") as f:
      if not keys: keys = list(f.keys())
      dfs = [ pd.DataFrame(np.array(f[group][key]), columns=self._cols(group, key)) for key in keys ]
      return pd.concat(dfs, axis="columns").set_index(["run","subrun","event"])

  def __getitem__(self, idx):
    """load a single event from file"""
    with h5py.File(self._filename, "r") as f:
      index = f["event_table/event_id"][idx]
      ret = { "index": index }

      for group, datasets in self._groups:
        m = (f[group]["event_id"][()] == index).all(axis=1)
        if not datasets: datasets = list(f[group].keys())
        def slice(g, d, m):
          n = f[g][d].shape[1]
          m = m[:,None].repeat(n, axis=1)
          return pd.DataFrame(f[g][d][m].reshape([-1,n]), columns=self._cols(g,d))
        dfs = [ slice(group, dataset, m) for dataset in datasets ]
        ret[group] = pd.concat(dfs, axis="columns")
      return ret

