import pandas as pd, torch, torch_geometric as tg
from ..core.file import NuMLFile
from ..labels import *
from ..graph import *

def process_event_singleplane(out, key, hit, part, edep, l=ccqe, e=edges.window_edges):
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
  evt_part = l.hit_label(evt_part)

  # join the dataframes to transform particle labels into hit labels
  evt_hit = evt_hit.merge(evt_part.drop(["parent_id", "type"], axis=1), on="g4_id", how="inner")

  # draw graph edges
  for p, plane in evt_hit.groupby("local_plane"):

    # Reset indices
    plane = plane.reset_index(drop=True).reset_index()

    # Build and label edges
    edge = e(plane)
    edge = l.edge_label(edge)

    # Save to file
    node_feats = ["global_plane", "global_wire", "global_time", "tpc",
      "local_plane", "local_wire", "local_time", "integral", "rms"]
    graph_dict = {
      "x": torch.tensor(plane[node_feats].to_numpy()).float(),
      "edge_index": torch.tensor(edge[["idx_1", "idx_2"]].to_numpy().T).long(),
      "y": torch.tensor(plane["label"].to_numpy()).long(),
      "y_edge": torch.tensor(edge["label"].to_numpy()).long()  
    }
    out.save(tg.data.Data(**graph_dict), f"r{key[0]}_sr{key[1]}_evt{key[2]}_p{p}")

def process_file(out, fname, g=process_event_singleplane, l=ccqe, e=edges.window_edges):
  """Process all events in a file into graphs"""
  f = NuMLFile(fname)

  evt = f.get_dataframe("event_table", ["event_id"])
  hit = f.get_dataframe("hit_table")
  part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type"])
  edep = f.get_dataframe("edep_table")

  # loop over events in file
  for key in evt.index: g(out, key, hit, part, edep, l, e)

