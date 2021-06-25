import pandas as pd, torch
from ..labels import ccqe
from ..graph import edges

node_feats = ["global_plane", "global_wire", "global_time", "tpc",
  "local_plane", "local_wire", "local_time", "integral", "rms"]

def ccqe_window_graph(key, out, hit, part, edep):

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
  evt_part = ccqe.hit_label(evt_part)

  # join the dataframes to transform particle labels into hit labels
  evt_hit = evt_hit.merge(evt_part.drop(["parent_id", "type"], axis=1), on="g4_id", how="inner")

  # draw graph edges
  for p, plane in evt_hit.groupby("local_plane"):

    # Reset indices
    plane = plane.reset_index(drop=True).reset_index()

    # Build and label edges
    edge = edges.window_edges(plane)
    edge = ccqe.edge_label(edge)

    # Save to file
    graph_dict = {
      "x": torch.tensor(plane[node_feats].to_numpy()).float(),
      "edge_index": torch.tensor(edge[["idx_1", "idx_2"]].to_numpy().T).long(),
      "y": torch.tensor(edge["label"].to_numpy()).long()
    }
    torch.save(graph_dict, f"{out}/r{key[0]}_sr{key[1]}_evt{key[2]}_p{p}.pt")
