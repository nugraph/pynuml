import pandas as pd, torch, torch_geometric as tg
from ..core.file import NuMLFile
from ..labels import *
from ..graph import *

def single_plane_graph(f, idx, l=ccqe.hit_label, e=edges.knn, **edge_args):
  """Process an event into graphs"""

  key = f.index(idx)

  import os.path as osp
  if osp.exists(f"/data/uboone/pandora/processed_delaunay/r{key[0]}_sr{key[1]}_evt{key[2]}_p0.pt"):
    print(f"skipping {key}")
    return

  evt = f[idx]

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
    ret.append([f"r{key[0]}_sr{key[1]}_evt{key[2]}_p0", data])
  return ret

def process_file(out, fname, g=single_plane_graph, l=ccqe.hit_label, e=edges.knn, p=None):
  """Process all events in a file into graphs"""
  print(f"Processing {fname}")
  f = NuMLFile(fname)

  f.add_group("hit_table")
  f.add_group("particle_table", ["event_id", "g4_id", "parent_id", "type"])
  f.add_group("edep_table")

  if p is None:
    for idx in range(len(f)):
      tmp = g(f, idx, l, e)
      if tmp is not None:
        for name, data in tmp:
          print("saving", name)
          out.save(data, name)
  else:
    from functools import partial
    import multiprocessing as mp
    procs = [ None for i in range(p) ]
    idx = 0
    with mp.Pool(processes=p) as pool:
      while True:
        for i in range(p):
          if procs[i] is None:
            print("starting new process", i+1, "for input", idx)
            procs[i] = pool.apply_async(g, (f, idx, l, e))
            idx += 1
          elif procs[i].ready():
            print("got output back from process", i+1)
            tmp = procs[i].get()
            if tmp is not None:
              for name, data in tmp:
                print("saving", name)
                out.save(data, name)
            print("starting new process", i+1, "for input", idx)
            procs[i] = pool.apply_async(g, (f, idx, l, e))
            idx += 1
          if idx == len(f): return

    # func = partial(g, out=out, l=l, e=e)
    # with mp.Pool(processes=p) as pool: pool.map(func, gen)

  print('End processing ', fname)

