import pandas as pd, torch, torch_geometric as tg
from ..core.file import NuMLFile
from ..labels import *
from ..graph import *

def single_plane_graph(out, key, hit, part, edep, sp, l=standard, e=edges.window_edges):
  """Process an event into graphs"""
  # skip any events with no simulated hits
  if (hit.index==key).sum() == 0: return
  if (edep.index==key).sum() == 0: return

  # get energy depositions, find max contributing particle, and ignore any hits with no truth
  evt_edep = edep.loc[key].reset_index(drop=True)
  evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
  evt_hit = evt_edep.merge(hit.loc[key].reset_index(), on="hit_id", how="inner").drop("energy_fraction", axis=1)

  # skip events with fewer than 50 simulated hits in any plane
  for i in range(3):
    if (evt_hit.global_plane==i).sum() < 50: return

  # get labels for each particle
  evt_part = part.loc[key].reset_index(drop=True)
  evt_part = l.semantic_label(evt_part)
  print(evt_part)

  # join the dataframes to transform particle labels into hit labels
  evt_hit = evt_hit.merge(evt_part.drop(["parent_id", "type"], axis=1), on="g4_id", how="inner")

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
    edge = e(plane)
    node_feats = ["global_plane", "global_wire", "global_time", "tpc",
      "local_plane", "local_wire", "local_time", "integral", "rms"]
    data["x"+suffix] = torch.tensor(plane[node_feats].to_numpy()).float()
    data["y"+suffix] = torch.tensor(plane["label"].to_numpy()).long()
    data["edge_index"+suffix] = torch.tensor(edge[["idx_1", "idx_2"]].to_numpy().T).long()
    data["edge_index_3d"+suffix] = torch.tensor(blah).transpose(0, 1).long()
    out.save(tg.data.Data(**data), f"r{key[0]}_sr{key[1]}_evt{key[2]}")

def process_file(out, fname, g=single_plane_graph, l=standard, e=edges.window_edges):
  """Process all events in a file into graphs"""
  f = NuMLFile(fname)
  evt = f.get_dataframe("event_table", ["event_id"])
  hit = f.get_dataframe("hit_table")
  part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type"])
  edep = f.get_dataframe("edep_table")
  sp = f.get_dataframe("spacepoint_table")

  # loop over events in file
  for key in evt.index: g(out, key, hit, part, edep, sp, l, e)

