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
    # print("event_table/event_id type==",type(self._index)," shape=",self._index.shape)

    # self._groups is a python list, each member is a 2-element list consisting
    # of group name, and a python list of dataset names
    self._groups = []

    # a python dictionary storing a sequence-count dataset in each group, keys
    # are group names, values are the sequence-count dataset subarrays assigned
    # to this process
    self._seq_cnt = {}
    self._evt_seq = {}

    self._use_seq_cnt = True

    # a python nested dictionary storing datasets of each group read from the
    # input file. keys of self._data are group names, values are python
    # dictionaries, each has names of dataset in that group as keys, and values
    # storing dataset subarrays
    self._data = {}

    # starting array index of event_table/event_id assigned to this process
    self._my_start = -1

    # end array index of event_table/event_id assigned to this process
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
      if "event_id" in keys: keys.remove("event_id")
      if "event_id.seq" in keys: keys.remove("event_id.seq")
      if "event_id.seq_cnt" in keys: keys.remove("event_id.seq_cnt")
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

  def calc_bound_seq(self, starts, ends, group):
    # return the lower and upper array indices of subarray assigned to this
    # process, using the partition sequence dataset

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    displ = np.empty([nprocs], dtype=np.int)
    count = np.empty([nprocs], dtype=np.int)
    bounds = np.empty([nprocs, 2], dtype=np.int)

    all_evt_seq = []
    if rank == 0:
      # root reads the entire dataset event_id.seq
      all_evt_seq = self._fd[group+"/event_id.seq"][:]
      dim = len(all_evt_seq)

      # calculate displ, count to be used in scatterV for all processes
      for i in range(nprocs):
        bounds[i, 0] = self.binary_search_min(starts[i], all_evt_seq, dim)
        bounds[i, 1] = self.binary_search_max(ends[i],   all_evt_seq, dim)
        displ[i] = bounds[i, 0]
        count[i] = bounds[i, 1] - bounds[i, 0] + 1

    lower_upper = np.empty([2], dtype=np.int)

    # root distributes start and end indices to all processes
    comm.Scatter(bounds, lower_upper, root=0)

    # this process is assigned array indices from lower to upper
    # print("group=",group," lower=",lower," upper=",upper," count=",upper-lower)

    lower = lower_upper[0]
    upper = lower_upper[1] + 1

    # root scatters the subarray of evt_seq to all processes
    self._evt_seq[group] = np.zeros(upper - lower, dtype=np.int64)
    comm.Scatterv([all_evt_seq, count, displ, MPI.LONG_LONG], self._evt_seq[group], root=0)

    return lower, upper

  def calc_bound_seq_cnt(self, starts, ends, group):
    # return the lower and upper array indices of subarray assigned to this
    # process, using the partition sequence-count dataset

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    displ = np.empty([nprocs], dtype=np.int)
    count = np.empty([nprocs], dtype=np.int)
    seq_cnt = np.empty([nprocs, 2], dtype=np.int)

    all_seq_cnt = []
    if rank == 0:
      # root reads the entire dataset event_id.seq_cnt
      all_seq_cnt = self._fd[group+"/event_id.seq_cnt"][:]
      dim = len(all_seq_cnt)

      # calculate displ, count for all processes to be used in scatterV
      recv_rank = 0  # receiver rank
      displ[recv_rank] = 0
      seq_cnt[recv_rank, 0] = 0
      seq_end = ends[recv_rank]
      seq_id = 0
      for i in range(dim):
        if all_seq_cnt[i, 0] > seq_end :
          seq_cnt[recv_rank, 1] = i - displ[recv_rank]
          recv_rank += 1  # move on to the next receiver rank
          seq_end = ends[recv_rank]
          displ[recv_rank] = i
          seq_cnt[recv_rank, 0] = seq_id
        seq_id += all_seq_cnt[i, 1]

      # last receiver rank
      seq_cnt[nprocs-1, 1] = dim - displ[nprocs-1]

      # print("starts=",starts," ends=",ends," displ=",displ," count=",count," seq_cnt=",seq_cnt )
      displ[:] *= 2
      count[:] = seq_cnt[:, 1] * 2

    # root distributes seq_cnt to all processes
    my_seq_cnt  = np.empty([2], dtype=np.int)
    comm.Scatter(seq_cnt, my_seq_cnt, root=0)

    # this process is assigned array indices from lower to upper
    # print("group=",group," lower=",lower," upper=",upper," count=",upper-lower)

    # self._seq_cnt[group][:, 0] is the event ID
    # self._seq_cnt[group][:, 1] is the number of elements
    self._seq_cnt[group] = np.empty([my_seq_cnt[1], 2], dtype=np.int64)

    # root scatters the subarray of evt_seq to all processes
    comm.Scatterv([all_seq_cnt, count, displ, MPI.LONG_LONG], self._seq_cnt[group], root=0)

    lower = my_seq_cnt[0]
    upper = my_seq_cnt[0] + np.sum(self._seq_cnt[group][:, 1])

    return lower, upper

  def read_data(self, starts, ends, use_seq=False, profile=False):
    # Parallel read dataset subarrays assigned to this process ranging from
    # array index of my_start to my_end
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()

    self._my_start    = starts[rank]
    self._my_end      = ends[rank]
    self._use_seq_cnt = use_seq

    bnd_time = 0
    rds_time = 0

    for group, datasets in self._groups:
      if profile: time_s = MPI.Wtime()

      if use_seq:
        # use evt_id.seq to calculate subarray boundaries
        lower, upper = self.calc_bound_seq(starts, ends, group)
      else:
        # use evt_id.seq_cnt to calculate subarray boundaries
        lower, upper = self.calc_bound_seq_cnt(starts, ends, group)

      if profile:
        time_e = MPI.Wtime()
        bnd_time += time_e - time_s
        time_s = time_e

      # Iterate through all the datasets and read the subarray from index lower
      # to upper and store it into a dictionary with the names of group and
      # dataset as the key.
      self._data[group] = {}
      for dataset in datasets:
        with self._fd[group][dataset].collective:  # read each dataset collectively
          self._data[group][dataset] = self._fd[group][dataset][lower : upper]

      if profile:
        time_e = MPI.Wtime()
        rds_time += time_e - time_s
        time_s = time_e

    if profile:
      total_t = np.array([bnd_time, rds_time])
      max_total_t = np.zeros(2)
      comm.Reduce(total_t, max_total_t, op=MPI.MAX, root = 0)
      min_total_t = np.zeros(2)
      comm.Reduce(total_t, min_total_t, op=MPI.MIN, root = 0)
      if rank == 0:
        print("---- Timing break down of the file read phase (in seconds) -------")
        if self._use_seq_cnt:
          print("Use event_id.seq_cnt to calculate subarray boundaries")
        else:
          print("Use event_id.seq to calculate subarray boundaries")
        print("calc boundaries time MAX=%8.2f  MIN=%8.2f" % (max_total_t[0], min_total_t[0]))
        print("read datasets   time MAX=%8.2f  MIN=%8.2f" % (max_total_t[1], min_total_t[1]))
        print("(MAX and MIN timings are among %d processes)" % nprocs)

  def build_evt(self, start, end):
    # This process is responsible for event IDs from start to end.
    # All data of the same event ID will be used to create a graph.
    # This function collects all data based on event_id.seq or event_id.seq_cnt
    # into a python list containing Panda DataFrames, one for a unique event
    # ID.
    ret_list = []

    num_miss = 0

    # track the latest used index per group
    idx_grp = dict.fromkeys(self._data.keys(), 0)

    # accumulate starting array index per group
    idx_start = dict.fromkeys(self._data.keys(), 0)

    # Iterate through assigned event IDs
    for idx in range(int(start), int(end) + 1):
      # check if idx is missing in all groups
      is_missing = True
      if self._use_seq_cnt:
        for group in self._data.keys():
          if idx_grp[group] >= len(self._seq_cnt[group][:, 0]):
            continue
          if idx == self._seq_cnt[group][idx_grp[group], 0]:
            is_missing = False
            break
      else:
        for group in self._data.keys():
          dim = len(self._evt_seq[group])
          lower = self.binary_search_min(idx, self._evt_seq[group], dim)
          upper = self.binary_search_max(idx, self._evt_seq[group], dim) + 1
          if lower < upper:
            is_missing = False
            break

      # this idx is missing in all groups
      if is_missing:
        num_miss += 1
        continue

      # for each event seq ID, create a dictionary, ret
      #   first item: key is "index" and value is the event seq ID
      #   remaining items: key is group name and value is a Panda DataFrame
      #   containing the dataset subarray in this group with the event ID, idx
      ret = { "index": idx }

      # Iterate through all groups
      for group in self._data.keys():

        if self._use_seq_cnt:
          # self._seq_cnt[group][:, 0] is the event ID
          # self._seq_cnt[group][:, 1] is the number of elements

          if idx_grp[group] >= len(self._seq_cnt[group][:, 0]) or idx < self._seq_cnt[group][idx_grp[group], 0]:
            # idx is missing from this group but may not in other groups
            # create an empty Panda DataFrame
            dfs = []
            for dataset in self._data[group].keys():
              data = np.array([])
              data_dataframe = pd.DataFrame(data, columns=self._cols(group, dataset))
              dfs.append(data_dataframe)
            ret[group] = pd.concat(dfs, axis="columns")
            # print("xxx",group," missing idx=",idx," _seq_cnt[0]=",self._seq_cnt[group][idx_grp[group], 0])
            continue

          lower = idx_start[group]
          upper = self._seq_cnt[group][idx_grp[group], 1] + lower

          idx_start[group] += self._seq_cnt[group][idx_grp[group], 1]
          idx_grp[group] += 1

        else:
          # Note self._evt_seq stores event ID values and is already sorted in
          # an increasing order
          dim = len(self._evt_seq[group])

          # Find the local start and end row indices for this event ID, idx
          lower = self.binary_search_min(idx, self._evt_seq[group], dim)
          upper = self.binary_search_max(idx, self._evt_seq[group], dim) + 1
          # print("For idx=",idx, " lower=",lower, " upper=",upper)

        # dfs is a python list containing Panda DataFrame objects
        dfs = []
        for dataset in self._data[group].keys():
          if lower >= upper:
            # idx is missing from the dataset event_id.seq
            # In this case, create an empty numpy array
            data = np.array([])
          else:
            # array elements from lower to upper of this dataset have the
            # event ID == idx
            data = self._data[group][dataset][lower : upper]

          # create a Panda DataFrame to store the numpy array
          data_dataframe = pd.DataFrame(data, columns=self._cols(group, dataset))

          dfs.append(data_dataframe)

        # concate into the dictionary "ret" with group names as keys
        ret[group] = pd.concat(dfs, axis="columns")

      # Add all dictionaries "ret" into a list.
      # Each of them corresponds to the data of one single event ID
      ret_list.append(ret)

    # print("start=",start," end=",end," num=",end-start+1," num miss IDs=",num_miss)
    return ret_list

