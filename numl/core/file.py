import h5py, numpy as np, pandas as pd
from mpi4py import MPI

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

    # open the input HDF5 file in parallel
    self._fd = h5py.File(self._filename, "r", driver='mpio', comm=MPI.COMM_WORLD)

    # all processes read dataset "event_table/event_id" stored in an numpy array
    self._index = self._fd["event_table/event_id"][:]

    # self._groups is a python list, each member is a 2-element list consisting
    # of group name, and a python list of dataset names
    self._groups = []

    # a python dictionary storing dataset event_id.seq in each group, keys are
    # group names, values are the event_id.seq subarrays assigned to this
    # process
    self._evt_seq = {}

    # a python nested dictionary storing datasets in each group, read from
    # input file keys are group names, values are python dictionaries, each has
    # names of dataset in that group as keys, and values storing dataset
    # subarrays
    self._data = {}

    # this process's starting array index of event_table/event_id.seq
    self._my_start = -1

    # this process's end array index of event_table/event_id.seq
    self._my_end = -1

  def __len__(self):
    # inquire the number of unique event IDs in the input file
    return len(self._index)

  def __str__(self):
    with h5py.File(self._filename, "r") as f:
      ret = ""
      for k1 in self._file.keys():
        ret += f"{k1}:\n"
        for k2 in self._file[k1].keys(): ret += f"  {k2}"
        ret += "\n"
      return ret

  def add_group(self, group, keys=[]):
    if not keys:
      # retrieve all the dataset names of the group
      keys = list(self._fd[group].keys())
      # dataset event_id is not needed
      keys.remove("event_id")
    self._groups.append([ group, keys ])

  def keys(self):
    # with h5py.File(self._filename, "r") as f:
    #   return f.keys()
    return self._fd.keys()

  def _cols(self, group, key):
    if key == "event_id": return [ "run", "subrun", "event" ]
    if key in self._colmap[group].keys(): return self._colmap[group][key]
    else: return [key]

  def get_dataframe(self, group, keys=[]):
    with h5py.File(self._filename, "r") as f:
      if not keys: keys = list(f.keys())
      dfs = [ pd.DataFrame(np.array(f[group][key]), columns=self._cols(group, key)) for key in keys ]
      return pd.concat(dfs, axis="columns").set_index(["run","subrun","event"])

  def index(self, idx):
    """get the index for a given row"""
    #with h5py.File(self._filename, "r") as f:
      #return f["event_table/event_id"][idx]
    return self._index[idx]

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

  def __del__(self):
    self._fd.close()

  def binary_search_min(self, key, base, nmemb):
    low = 0
    high = nmemb
    while low != high:
        mid = (low + high) // 2
        if base[mid] < key:
            low = mid + 1
        else:
            high = mid
    return low

  def binary_search_max(self, key, base, nmemb):
    low = 0
    high = nmemb
    while low != high:
        mid = (low + high) // 2
        if base[mid] <= key:
            low = mid + 1
        else:
            high = mid
    return (low - 1)


  def read_data(self, starts, ends):
    # Parallel read dataset subarrays assigned to this process ranging from
    # array index of my_start to my_end
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    self._my_start = starts[rank]
    self._my_end = ends[rank]

    for group, datasets in self._groups:
      all_evt_seq = []
      bounds = None
      if rank == 0:
        # root reads the entire event_id.seq dataset
        seq_name = group + "/event_id.seq"
        all_evt_seq = self._fd[seq_name][:, 0]
        # print("all_evt_seq type=", all_evt_seq.dtype)
        dim = len(all_evt_seq)
        # print("group=",group," dataset=",datasets," len of event_id.seq=",dim)

        # calculate the sizes, start, end indices for all other processes
        displ = []
        count = []
        bounds = np.empty([nprocs, 2], dtype=np.int)
        for i in range(nprocs):
          bounds[i, 0] = self.binary_search_min(starts[i], all_evt_seq, dim)
          bounds[i, 1] = self.binary_search_max(ends[i],   all_evt_seq, dim)
          displ.append(bounds[i, 0])
          count.append(bounds[i, 1] - bounds[i, 0] + 1)
        # print("count=",count)
        # print("displ=",displ)
      else:
        count = np.zeros(nprocs, dtype=np.int)
        displ = None

      # root distributes start and end indices to all processes
      lower_upper = np.empty(2, dtype=np.int)
      comm.Scatter(bounds, lower_upper, root=0)

      # this process is assigned array indices from lower to upper
      lower = lower_upper[0]
      upper = lower_upper[1]
      # print("rank ",rank," group=",group," lower=",lower," upper=",upper," count=",upper-lower+1)

      # size of local _evt_seq array
      self._evt_seq[group] = np.zeros(upper - lower + 1, dtype=np.int64)

      # root distributes all_evt_seq to all processes
      comm.Scatterv([all_evt_seq, count, displ, MPI.LONG_LONG], self._evt_seq[group], root=0)

      # Iterate through all the datasets and read the subarray from index lower
      # to upper and store it into a dictionary with the names of group and
      # dataset as the key.
      self._data[group] = {}
      for dataset in datasets:
        with self._fd[group][dataset].collective:  # read each dataset collectively
          self._data[group][dataset] = self._fd[group][dataset][lower : upper + 1]

  def build_evt(self, start, end):
    # This process is responsible for event IDs from start to end.
    # All data of the same event ID will be used to create a graph.
    # This function collects all data based on event_id.seq into a python
    # list containing Panda DataFrames, one for a unique event ID.
    ret_list = []

    # The values of event IDs are stored in self._evt_seq[group][]
    # Iterate through assigned event IDs
    for idx in range(int(start), int(end) + 1):
      # for each event ID, create a dictionary
      #   first item: key is "index" and value is the event ID
      #   remaining items: key is group name and value is a Panda DataFrame
      #   containing the dataset subarray in this group with the event ID, idx
      ret = { "index": idx }

      # Iterate through all groups
      for group in self._data.keys():
        # Note self._evt_seq stores event ID values and is already sorted in an
        # increasing order
        dim = len(self._evt_seq[group])

        # Find the local start and end row indices for this event ID, idx
        idx_start = self.binary_search_min(idx, self._evt_seq[group], dim)
        idx_end   = self.binary_search_max(idx, self._evt_seq[group], dim)
        # print("For idx=",idx, " idx_start=",idx_start, " idx_end=",idx_end)

        # dfs is a python list containing Panda DataFrame objects
        dfs = []
        for dataset in self._data[group].keys():
          if idx_start > idx_end:
            # idx is missing from the dataset event_id.seq
            # In this case, create an empty numpy array
            data = np.array([])
          else:
            # array elements from idx_start to idx_end of this dataset have the
            # event ID == idx
            data = self._data[group][dataset][idx_start : idx_end + 1]

          # create a Panda DataFrame to store the numpy array
          data_dataframe = pd.DataFrame(data, columns=self._cols(group, dataset))

          dfs.append(data_dataframe)

        # concatenate into the dictionary "ret" with group names as keys
        ret[group] = pd.concat(dfs, axis="columns")

      # Add all dictionaries "ret" into a list.
      # Each of them corresponds to the data of one single event ID
      ret_list.append(ret)
    return ret_list

