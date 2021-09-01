import pandas as pd, torch, torch_geometric as tg
from ..core.file import NuMLFile
from ..labels import *
from ..graph import *
from ..core.out import PTOut
from mpi4py import MPI
import numpy as np

def single_plane_graph(event_id, evt, l=ccqe.hit_label, e=edges.knn, **edge_args):
  """Process an event into graphs"""

  # skip any events with no simulated hits
  # if (hit.index==key).sum() == 0: return
  # if (edep.index==key).sum() == 0: return

  # get energy depositions, find max contributing particle, and ignore any evt_hits with no truth
  evt_edep = evt["edep_table"]
  evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
  evt_hit = evt_edep.merge(evt["hit_table"], on="hit_id", how="inner").drop("energy_fraction", axis=1)

  # skip events with fewer than 50 simulated hits in any plane
  for i in range(3):
    if (evt_hit.global_plane==i).sum() < 50: return

  # get labels for each evt_particle
  evt_part = l(evt["particle_table"])

  # join the dataframes to transform evt_particle labels into hit labels
  evt_hit = evt_hit.merge(evt_part.drop(["parent_id", "type"], axis=1), on="g4_id", how="inner")

  # draw graph edges
  ret = []
  for p, plane in evt_hit.groupby("local_plane"):

    # Reset indices
    plane = plane.reset_index(drop=True).reset_index()

    pos = plane[["global_wire", "global_time"]].values / torch.tensor([0.5, 0.075])[None, :].float()
    node_feats = ["global_plane", "global_wire", "global_time", "tpc",
      "local_plane", "local_wire", "local_time", "integral", "rms"]
    data = tg.data.Data(
      x=torch.tensor(plane[node_feats].values).float(),
      y=torch.tensor(plane["label"].values).long(),
      pos=pos,
    )
    data = e(data, **edge_args)
    ret.append([f"r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}_p0", data])
  return ret

def process_file(out, fname, g=single_plane_graph, l=ccqe.hit_label, e=edges.delaunay, p=None):
  comm = MPI.COMM_WORLD
  nprocs = comm.Get_size()
  rank = comm.Get_rank()

  start_t = MPI.Wtime()
  timing = start_t
  """Process all events in a file into graphs"""
  if rank == 0:
    print(f"Processing file: {fname}")
    print(f"Output folder: {out.outdir}")

  # open input file and read dataset "/event_table/event_id"
  f = NuMLFile(fname)

  # only use the following groups and datasets in them
  f.add_group("hit_table")
  f.add_group("particle_table", ["event_id.seq", "g4_id", "parent_id", "type"])
  f.add_group("edep_table")

  # number of unique event IDs in the input file
  event_id_len = len(f)
  if rank == 0:
    print("Size of event_table/event_id is ", event_id_len)

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
  f.read_data(starts, ends)

  read_time = MPI.Wtime() - timing
  timing = MPI.Wtime()

  # organize the data into a list based on event IDs, so data corresponding to
  # one event ID can be used to create a graph. A graph will be stored as a
  # dataframe.
  evt_list = f.build_evt(my_start, my_end)
  # print("rank ",rank, " len(evt_list)=", len(evt_list))

  build_list_time = MPI.Wtime() - timing
  write_time = 0
  graph_time = 0

  # Iterate through event IDs, construct graphs and save them in files
  for idx in range(len(evt_list)):
    timing = MPI.Wtime()
    # avoid overwriting to already existing files
    import os.path as osp
    event_id = f.index(my_start + idx)
    if osp.exists(f"{out.outdir}/r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}_p0.pt"):
      print(f"{rank}: skipping event ID {event_id}")
      continue
    tmp = g(event_id, evt_list[idx], l, e)
    graph_time += MPI.Wtime() - timing

    timing = MPI.Wtime()
    if tmp is not None:
      for name, data in tmp:
        # print("saving", name)
        out.save(data, name)
    write_time += MPI.Wtime() - timing

  total_time = MPI.Wtime() - start_t

  total_t = np.array([read_time, build_list_time, graph_time, write_time, total_time])
  max_total_t = np.zeros(5)
  comm.Reduce(total_t, max_total_t, op=MPI.MAX, root = 0)
  min_total_t = np.zeros(5)
  comm.Reduce(total_t, min_total_t, op=MPI.MIN, root = 0)

  if rank == 0:
    print("Number of MPI processes = ", nprocs)
    print("read file file  time MAX=%8.2f  MIN=%8.2f" % (max_total_t[0], min_total_t[0]))
    print("build dataframe time MAX=%8.2f  MIN=%8.2f" % (max_total_t[1], min_total_t[1]))
    print("graph creation  time MAX=%8.2f  MIN=%8.2f" % (max_total_t[2], min_total_t[2]))
    print("write to file   time MAX=%8.2f  MIN=%8.2f" % (max_total_t[3], min_total_t[3]))
    print("total           time MAX=%8.2f  MIN=%8.2f" % (max_total_t[4], min_total_t[4]))
    print("(MAX and MIN timings are among %d processes)" % nprocs)

