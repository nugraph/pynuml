import h5py, numpy as np, pandas as pd

class NuMLFile:
  def __init__(self, file):
    self._file = h5py.File(file, "r")
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

  def __len__(self):
    return self._file["event_table/event_id"].shape[0]

  def __str__(self):
    ret = ""
    for k1 in self._file.keys():
      ret += f"{k1}:\n"
      for k2 in self._file[k1].keys(): ret += f"  {k2}"
      ret += "\n"
    return ret

  def id(self, idx):
    if not 0 <= idx < len(self):
      raise Exception(f"Index {idx} out of range in file!")
    return self._file["event_table/event_id"][idx]

  def keys(self):
    return self._file.keys()

  def _cols(self, group, key):
    if key == "event_id": return [ "run", "subrun", "event" ]
    if key in self._colmap[group].keys(): return self._colmap[group][key]
    else: return [key]

  def get_dataframe(self, group, keys=[]):
    if not keys: keys = list(self._file[group].keys())
    dfs = [ pd.DataFrame(np.array(self._file[group][key]), columns=self._cols(group, key)) for key in keys ]
    return pd.concat(dfs, axis="columns").set_index(["run","subrun","event"])

  def events(self, groups):
    for i in range(len(self)):
      index = self.id(i)
      ret = {}
      # loop over groups in file
      for group, datasets in groups:
        # mask out based on run, subrun and event
        m = (self._file[group]["event_id"][()] == index).all(axis=1)
        if not datasets: datasets = list(self._file[group].keys())
        def slice(g, d, m):
          n = self._file[g][d].shape[1]
          m = m[:,None].repeat(n, axis=1)
          return pd.DataFrame(self._file[g][d][m].reshape([-1,n]), columns=self._cols(g,d))
        dfs = [ slice(group, dataset, m) for dataset in datasets ]
        ret[group] = pd.concat(dfs, axis="columns")
      yield ret

