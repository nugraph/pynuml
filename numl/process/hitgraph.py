import pandas as pd, torch, torch_geometric as tg
from ..core.file import NuMLFile
from ..labels import *
from ..graph import *
from ..core.out import PTOut, H5Out
from mpi4py import MPI
import numpy as np
import sys

edep1_t = 0.0
edep2_t = 0.0
hit_merge_t = 0.0
torch_t = 0.0
plane_t = 0.0
label_t = 0.0
knn_t = 0.0
my_num_graphs = 0
profiling = False

def single_plane_graph(event_id, evt, l=ccqe.hit_label, e=edges.knn, **edge_args):
  """Process an event into graphs"""

  # skip any events with no simulated hits
  # if (hit.index==key).sum() == 0: return
  # if (edep.index==key).sum() == 0: return

  global edep1_t, edep2_t, hit_merge_t, torch_t, plane_t, label_t, knn_t
  global my_num_graphs
  global profiling
  if profiling:
    start_t = MPI.Wtime()

  # get energy depositions, find max contributing particle, and ignore any evt_hits with no truth
  evt_edep = evt["edep_table"]

  # evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
  # below is faster than above
  evt_edep = evt_edep.sort_values(by=['energy_fraction'], ascending=False, kind='mergesort').drop_duplicates('hit_id')

  if profiling:
    end_t = MPI.Wtime()
    edep1_t += end_t - start_t
    start_t = end_t

  evt_hit = evt_edep.merge(evt["hit_table"], on="hit_id", how="inner").drop("energy_fraction", axis=1)

  if profiling:
    end_t = MPI.Wtime()
    edep2_t += end_t - start_t
    start_t = end_t

  # skip events with fewer than 50 simulated hits in any plane
  for i in range(3):
    if (evt_hit.global_plane==i).sum() < 50: return

  # get labels for each evt_particle
  evt_part = l(evt["particle_table"])

  if profiling:
    end_t = MPI.Wtime()
    label_t += end_t - start_t
    start_t = end_t

  # join the dataframes to transform evt_particle labels into hit labels
  evt_hit = evt_hit.merge(evt_part.drop(["parent_id", "type"], axis=1), on="g4_id", how="inner")

  if profiling:
    end_t = MPI.Wtime()
    hit_merge_t += end_t - start_t
    start_t = end_t

  # draw graph edges
  ret = []
  for p, plane in evt_hit.groupby("local_plane"):

    # Reset indices
    plane = plane.reset_index(drop=True).reset_index()

    pos = plane[["global_wire", "global_time"]].values / torch.tensor([0.5, 0.075])[None, :].float()
    node_feats = ["global_plane", "global_wire", "global_time", "tpc",
      "local_plane", "local_wire", "local_time", "integral", "rms"]

    if profiling:
      end_t = MPI.Wtime()
      plane_t += end_t - start_t
      start_t = end_t

    data = tg.data.Data(
      x=torch.tensor(plane[node_feats].values).float(),
      y=torch.tensor(plane["label"].values).long(),
      pos=pos,
    )

    if profiling:
      end_t = MPI.Wtime()
      torch_t += end_t - start_t
      start_t = end_t

    data = e(data, **edge_args)
    ret.append([f"r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}_p{p}", data])
    my_num_graphs += 1

    if profiling:
      end_t = MPI.Wtime()
      knn_t += end_t - start_t
      start_t = end_t

  return ret

def process_file(out, fname, g=single_plane_graph, l=ccqe.hit_label, e=edges.delaunay, p=None, use_seq=False, profile=False):
  comm = MPI.COMM_WORLD
  nprocs = comm.Get_size()
  rank = comm.Get_rank()

  global profiling
  profiling = profile
  if profiling:
    comm.Barrier()
    start_t = MPI.Wtime()
    timing = start_t

  """Process all events in a file into graphs"""
  if rank == 0:
    print("------------------------------------------------------------------")
    print(f"Processing input file: {fname}")
    if isinstance(out, PTOut): print(f"Output folder: {out.outdir}")
    if isinstance(out, H5Out): print(f"Output file: {out.fname}")

  # open input file and read dataset "/event_table/event_id.seq_cnt"
  f = NuMLFile(fname)

  # only use the following groups and datasets in them
  f.add_group("hit_table")
  f.add_group("particle_table", ["g4_id", "parent_id", "type"])
  f.add_group("edep_table")

  # number of unique event IDs in the input file
  event_id_len = len(f)

  # Calculate the start and end evt.seq id for each process
  starts = []
  ends = []
  _count = event_id_len // nprocs
  for j in range(event_id_len % nprocs):
    starts.append(_count * j + j)
    ends.append(starts[j] + _count)

  for j in range(event_id_len % nprocs, nprocs):
    starts.append(_count * j + event_id_len % nprocs)
    ends.append(starts[j] + _count - 1)

  # This process is assigned event IDs of range from my_start to my_end
  my_start = starts[rank]
  my_end   = ends[rank]
  # print("rank ",rank," my_start=",my_start," my_end=",my_end)

  # read data of the event IDs assigned to this process
  f.read_data(starts, ends, use_seq=use_seq, profile=profiling)

  if profiling:
    read_time = MPI.Wtime() - timing
    comm.Barrier()
    timing = MPI.Wtime()

  # organize the data into a list based on event IDs, so data corresponding to
  # one event ID can be used to create a graph. A graph will be stored as a
  # dataframe.
  evt_list = f.build_evt(my_start, my_end)
  # print("len(evt_list)=", len(evt_list))

  if profiling:
    build_list_time = MPI.Wtime() - timing
    comm.Barrier()
    write_time = 0
    graph_time = 0
    local_size = np.zeros(2)

  # num_planes = 3
  # num_evts = len(evt_list) * num_planes
  # edge_index = np.empty((num_evts, 2), int)
  # print("edge_index.ndim=\n", edge_index.ndim)
  # print("edge_index.shape=\n", edge_index.shape)
  # pos = np.empty((num_evts, 2), int)
  # x   = np.empty((num_evts, 2), int)
  # y   = np.empty((num_evts, 1), int)

  # j = 0
  # Iterate through event IDs, construct graphs and save them in files
  for i in range(len(evt_list)):
    if profiling:
      timing = MPI.Wtime()
      for group in evt_list[i].keys():
        if group != "index":
          local_size[0] += sys.getsizeof(evt_list[i][group])

    # retrieve event sequence ID
    idx = evt_list[i]["index"]
    event_id = f.index(idx)

    # avoid overwriting to already existing files
    if isinstance(out, PTOut):
      import os.path as osp
      if osp.exists(f"{out.outdir}/r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}_p0.pt"):
        # print(f"{rank}: skipping event ID {event_id}")
        continue

    tmp = g(event_id, evt_list[i], l, e)

    if profiling:
      graph_time += MPI.Wtime() - timing
      timing = MPI.Wtime()

    if tmp is not None:
      for name, data in tmp:
        # print("saving", name)
        out.save(data, name)
        # print("type of data =", type(data))
        # edge_index[j] = list(data.edge_index.size())
        # pos[j] = list(data.pos.size())
        # x[j] = list(data.x.size())
        # y[j] = list(data.y.size())
        # j += 1

    # if j == 1: break

    if profiling:
      write_time += MPI.Wtime() - timing

  # print("edge_index=\n", edge_index)
  # print("pos=\n", pos)
  # print("x=\n", x)
  # print("y=\n", y)
  # print("j=\n", j)

  if profiling:
    total_time = MPI.Wtime() - start_t

    global edep1_t, edep2_t, hit_merge_t, torch_t, plane_t, label_t, knn_t
    global my_num_graphs

    total_t = np.array([read_time, build_list_time, graph_time, write_time, total_time, edep1_t, edep2_t, label_t, hit_merge_t, plane_t, torch_t, knn_t])
    max_total_t = np.zeros(12)
    comm.Reduce(total_t, max_total_t, op=MPI.MAX, root = 0)
    min_total_t = np.zeros(12)
    comm.Reduce(total_t, min_total_t, op=MPI.MIN, root = 0)

    num_graphs = np.array([my_num_graphs], dtype=np.int)
    sum_num_graphs = np.zeros(1, dtype=np.int)
    comm.Reduce(num_graphs, sum_num_graphs, op=MPI.SUM, root = 0)

    local_size[1] = num_graphs
    max_size = np.zeros(2)
    comm.Reduce(local_size, max_size, op=MPI.MAX, root = 0)
    min_size = np.zeros(2)
    comm.Reduce(local_size, min_size, op=MPI.MIN, root = 0)

    if rank == 0:
      print("---- Timing break down of graph creation phase (in seconds) ------")
      print("edep grouping   time MAX=%8.2f  MIN=%8.2f" % (max_total_t[5], min_total_t[5]))
      print("edep merge      time MAX=%8.2f  MIN=%8.2f" % (max_total_t[6], min_total_t[6]))
      print("labelling       time MAX=%8.2f  MIN=%8.2f" % (max_total_t[7], min_total_t[7]))
      print("hit_table merge time MAX=%8.2f  MIN=%8.2f" % (max_total_t[8], min_total_t[8]))
      print("plane build     time MAX=%8.2f  MIN=%8.2f" % (max_total_t[9], min_total_t[9]))
      print("torch_geometric time MAX=%8.2f  MIN=%8.2f" % (max_total_t[10], min_total_t[10]))
      print("edge knn        time MAX=%8.2f  MIN=%8.2f" % (max_total_t[11], min_total_t[11]))
      print("(MAX and MIN timings are among %d processes)" % nprocs)
      print("------------------------------------------------------------------")
      print("Number of MPI processes = ", nprocs)
      print("Number of event IDs     = ", event_id_len)
      print("Total no. of graphs     = ", sum_num_graphs[0])
      print("Local no. of graphs  MAX= %6d   MIN= %6d" % (max_size[1], min_size[1]))
      print("Local graph size     MAX=%8.2f  MIN=%8.2f (MiB)" % (max_size[0]/1048576.0, min_size[0]/1048576.0))
      print("---- Top-level timing breakdown (in seconds) ---------------------")
      print("read from file  time MAX=%8.2f  MIN=%8.2f" % (max_total_t[0], min_total_t[0]))
      print("build dataframe time MAX=%8.2f  MIN=%8.2f" % (max_total_t[1], min_total_t[1]))
      print("graph creation  time MAX=%8.2f  MIN=%8.2f" % (max_total_t[2], min_total_t[2]))
      print("write to files  time MAX=%8.2f  MIN=%8.2f" % (max_total_t[3], min_total_t[3]))
      print("total           time MAX=%8.2f  MIN=%8.2f" % (max_total_t[4], min_total_t[4]))
      print("(MAX and MIN timings are among %d processes)" % nprocs)


