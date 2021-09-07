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


  def read_data(self, starts, ends, profile=False):
    # Parallel read dataset subarrays assigned to this process ranging from
    # array index of my_start to my_end
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    self._my_start = starts[rank]
    self._my_end = ends[rank]

    seq_cnt = np.empty([nprocs, 2], dtype=np.int)
    displ = np.empty([nprocs], dtype=np.int)
    count = np.empty([nprocs], dtype=np.int)

    seq_time = 0
    bin_time = 0
    sca_time = 0
    scv_time = 0
    rds_time = 0
    if profile: time_s = MPI.Wtime()

    for group, datasets in self._groups:

      all_seq_cnt = []
      if rank == 0:
        # root reads the entire dataset event_id.seq_cnt
        all_seq_cnt = self._fd[group+"/event_id.seq_cnt"][:]
        # print("all_seq_cnt type=", all_seq_cnt.dtype)
        dim = len(all_seq_cnt)
        # print("group=",group," dataset=",datasets," len of seq_cnt=",dim)

        if profile:
          time_e = MPI.Wtime()
          seq_time += time_e - time_s
          time_s = time_e

        # calculate displ, count to be used in scatterV for all processes
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

        if profile:
          time_e = MPI.Wtime()
          bin_time += time_e - time_s
          time_s = time_e

      # root distributes seq_cnt to all processes
      my_seq_cnt = np.empty([2], dtype=np.int)
      comm.Scatter(seq_cnt, my_seq_cnt, root=0)
      # print("------------------scatter my_seq_cnt=",my_seq_cnt)

      if profile:
        time_e = MPI.Wtime()
        if rank == 0: sca_time += time_e - time_s
        time_s = time_e

      # size of local _seq_cnt array
      # self._seq_cnt[group][:, 0] is the event ID
      # self._seq_cnt[group][:, 1] is the number of elements
      self._seq_cnt[group] = np.empty([my_seq_cnt[1], 2], dtype=np.int64)
      # print("------------------ group=",group," _seq_cnt size=",my_seq_cnt[1])

      # root distributes all_seq_cnt to all processes
      comm.Scatterv([all_seq_cnt, count, displ, MPI.LONG_LONG], self._seq_cnt[group], root=0)
      # print("group=",group," _seq_cnt[0]=",self._seq_cnt[group][0,:]," [e]=",self._seq_cnt[group][my_seq_cnt[1]-1,:]," my_seq_cnt[1]=",my_seq_cnt[1])

      if profile:
        time_e = MPI.Wtime()
        scv_time += time_e - time_s
        time_s = time_e

      # this process is assigned array indices from lower to upper
      lower = my_seq_cnt[0]
      upper = my_seq_cnt[0] + np.sum(self._seq_cnt[group][:, 1])
      # print("group=",group," lower=",lower," upper=",upper," count=",upper-lower)

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
      total_t = np.array([seq_time, bin_time, sca_time, scv_time, rds_time])
      max_total_t = np.zeros(5)
      comm.Reduce(total_t, max_total_t, op=MPI.MAX, root = 0)
      min_total_t = np.zeros(5)
      comm.Reduce(total_t, min_total_t, op=MPI.MIN, root = 0)
      if rank == 0:
        print("read seq    time MAX=%8.2f  MIN=%8.2f" % (max_total_t[0], min_total_t[0]))
        print("bin search  time MAX=%8.2f  MIN=%8.2f" % (max_total_t[1], min_total_t[1]))
        print("scatter     time MAX=%8.2f  MIN=%8.2f" % (max_total_t[2], min_total_t[2]))
        print("scatterV    time MAX=%8.2f  MIN=%8.2f" % (max_total_t[3], min_total_t[3]))
        print("read remain time MAX=%8.2f  MIN=%8.2f" % (max_total_t[4], min_total_t[4]))

  def build_evt(self, start, end):
    # This process is responsible for event IDs from start to end.
    # All data of the same event ID will be used to create a graph.
    # This function collects all data based on event_id.seq_cnt into a python
    # list containing Panda DataFrames, one for a unique event ID.
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
      for group in self._data.keys():
        if idx_grp[group] >= len(self._seq_cnt[group][:, 0]):
          continue
        if idx == self._seq_cnt[group][idx_grp[group], 0]:
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

        idx_end = self._seq_cnt[group][idx_grp[group], 1] + idx_start[group]

        # dfs is a python list containing Panda DataFrame objects
        dfs = []
        for dataset in self._data[group].keys():
          # array elements from idx_start to idx_end of this dataset have the
          # event ID == idx
          data = self._data[group][dataset][idx_start[group] : idx_end]

          # create a Panda DataFrame to store the numpy array
          data_dataframe = pd.DataFrame(data, columns=self._cols(group, dataset))

          dfs.append(data_dataframe)

        # concatenate into the dictionary "ret" with group names as keys
        ret[group] = pd.concat(dfs, axis="columns")
        idx_start[group] += self._seq_cnt[group][idx_grp[group], 1]
        idx_grp[group] += 1

      # Add all dictionaries "ret" into a list.
      # Each of them corresponds to the data of one single event ID
      ret_list.append(ret)

    # print("start=",start," end=",end," num=",end-start+1," num miss IDs=",num_miss)
    return ret_list

