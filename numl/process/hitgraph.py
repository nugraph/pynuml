import pandas as pd, torch, torch_geometric as tg
from ..core.file import NuMLFile
from ..labels import *
from ..graph import *

def process_event_singleplane(out, key, hit, part, edep, sp, l=ccqe.semantic_label, e=edges.window, **edge_args):
  """Process an event into graphs"""
  # skip any events with no simulated hits
  print(f"processing event {key[0]}, {key[1]}, {key[2]}")

  if key not in hit.index or key not in edep.index: return

  # get energy depositions, find max contributing particle, and ignore any hits with no truth
  evt_edep = edep.loc[key].reset_index(drop=True)
  evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
  evt_hit = evt_edep.merge(hit.loc[key].reset_index(), on="hit_id", how="inner").drop("energy_fraction", axis=1)

  # skip events with fewer than 50 simulated hits in any plane
  for i in range(3):
    if (evt_hit.global_plane==i).sum() < 20: return

  # get labels for each particle
  evt_part = part.loc[key].reset_index(drop=True)
  evt_part = l(evt_part)

  # join the dataframes to transform particle labels into hit labels
  evt_hit = evt_hit.merge(evt_part, on="g4_id", how="inner")

  # draw graph edges
  for p, plane in evt_hit.groupby("local_plane"):

    # Reset indices
    plane = plane.reset_index(drop=True).reset_index()

    # Save to file
    node_feats = ["global_plane", "global_wire", "global_time", "tpc",
      "local_plane", "local_wire", "local_time", "integral", "rms"]
    data = tg.data.Data(
      x=torch.tensor(plane[node_feats].to_numpy()).float(),
      y_s=torch.tensor(plane["semantic_label"].to_numpy()).long(),
      pos=plane[["global_wire", "global_time"]].values / torch.tensor([0.5, 0.075])[None, :].float(),
    )
    if "instance_label" in plane.keys():
      data.y_i = torch.tensor(plane["instance_label"].to_numpy()).long()
    data = e(data, **edge_args)
    out.save(data, f"r{key[0]}_sr{key[1]}_evt{key[2]}_p{p}")

def process_event(out, key, hit, part, edep, sp, l=ccqe.semantic_label, e=edges.window, **edge_args):
  """Process an event into graphs"""
  # skip any events with no simulated hits
  if out.exists(f"r{key[0]}_sr{key[1]}_evt{key[2]}"):
    print(f"file r{key[0]}_sr{key[1]}_evt{key[2]} exists! skipping")
    return
  print(f"processing event {key[0]}, {key[1]}, {key[2]}")

  if key not in hit.index or key not in edep.index: return

  # get energy depositions, find max contributing particle, and ignore any hits with no truth
  evt_edep = edep.loc[key].reset_index(drop=True)
  evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
  evt_hit = evt_edep.merge(hit.loc[key].reset_index(), on="hit_id", how="inner").drop("energy_fraction", axis=1)

  # skip events with fewer than 50 simulated hits in any plane
  for i in range(3):
    if (evt_hit.global_plane==i).sum() < 20: return

  # get labels for each particle
  evt_part = part.loc[key].reset_index(drop=True)
  evt_part = l(evt_part)

  # join the dataframes to transform particle labels into hit labels
  evt_hit = evt_hit.merge(evt_part, on="g4_id", how="inner")

  planes = [ "_u", "_v", "_y" ]

  evt_sp = sp.loc[key].reset_index(drop=True)

  data = { "n_sp": evt_sp.shape[1] }

  # draw graph edges
  for p, plane in evt_hit.groupby("local_plane"):

    # Reset indices
    plane = plane.reset_index(drop=True).reset_index()

    # build 3d edges
    suffix = planes[p]
    plane_sp = evt_sp.rename(columns={"hit_id"+suffix: "hit_id"}).reset_index()
    plane_sp = plane_sp[(plane_sp.hit_id != -1)]
    k3d = ["index","hit_id"]
    edges_3d = pd.merge(plane_sp[k3d], plane[(plane.hit_id != -1)][k3d], on="hit_id", how="inner", suffixes=["_3d", "_2d"])
    blah = edges_3d[["index_2d", "index_3d"]].to_numpy()

    # Save to file
    tmp = tg.data.Data(
      pos=plane[["global_wire", "global_time"]].values / torch.tensor([0.5, 0.075])[None, :].float()
    )
    tmp = e(tmp, **edge_args)
    node_feats = ["global_plane", "global_wire", "global_time", "tpc",
      "local_plane", "local_wire", "local_time", "integral", "rms"]
    data["x"+suffix] = torch.tensor(plane[node_feats].to_numpy()).float()
    data["y"+suffix] = torch.tensor(plane["semantic_label"].to_numpy()).long()
    data["edge_index"+suffix] = tmp.edge_index
    data["edge_index_3d"+suffix] = torch.tensor(blah).transpose(0, 1).long()
  out.save(tg.data.Data(**data), f"r{key[0]}_sr{key[1]}_evt{key[2]}")

def process_file(fname, out, g=process_event, l=ccqe.semantic_label, e=edges.knn, p=None):
  """Process all events in a file into graphs"""
  try:
    f = NuMLFile(fname)

    evt = f.get_dataframe("event_table", ["event_id"])
    hit = f.get_dataframe("hit_table")
    part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type", "momentum", "start_process", "end_process"])
    edep = f.get_dataframe("edep_table")
    sp = f.get_dataframe("spacepoint_table")

    # loop over events in file
    for key in evt.index: g(out, key, hit, part, edep, sp, l, e)

    print('End processing ', fname)

  except OSError:
    print(f"Could not open file {fname}. Skipping.")

