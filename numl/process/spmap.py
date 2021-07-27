import pandas as pd, torch, torch_geometric as tg
from ..core.file import NuMLFile
from ..labels import *
from ..graph import *

def process_event(out, key, sp, hit, part, edep, l=standard):
  """Process an event into graphs"""
  # skip any events with no simulated hits
  if (hit.index==key).sum() == 0: return
  if (edep.index==key).sum() == 0: return

  # label true particles
  evt_part = part.loc[key].reset_index(drop=True)
  evt_part["label_semantic"] = l.semantic_label(evt_part)

  print(evt_part)

  # get energy depositions, find max contributing particle, and ignore any hits with no truth
  evt_edep = edep.loc[key].reset_index(drop=True)
#  print(evt_edep)

  def test(df):
    print(df)
#    df = df.groupby("g4_id")
#    print(df)
#    exit(0)

  grouped = evt_edep.groupby("hit_id")
#  grouped.agg(test)

#  evt_edep = evt_edep.groupby("hit_id") #pipe(test)
#  evt_edep.groupby("hit_id").agg(test)

  # now we need to turn this into a fractional ground truth
  # evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
  # evt_hit = evt_edep.merge(hit.loc[key].reset_index(), on="hit_id", how="inner").drop("energy_fraction", axis=1)

  evt_sp = sp.loc[key].reset_index(drop=True)

  # skip events with fewer than 50 simulated hits in any plane, or fewer than 50 spacepoints
  # for i in range(3):
  #   if (evt_hit.global_plane==i).sum() < 50: return
  # if evt_sp.shape[0] < 50: return

  # get labels for each particle
  # evt_part = part.loc[key].reset_index(drop=True)
  # evt_part = l.semantic_labels(evt_part)
  # evt_part = l.instance_labels(evt_part)

  # join the dataframes to transform particle labels into hit labels
  # evt_hit = evt_hit.merge(evt_part.drop(["parent_id", "type"], axis=1), on="g4_id", how="inner")

  # now we need to combine those
  # node_feats = ["global_plane", "global_wire", "global_time", "tpc",
  #   "local_plane", "local_wire", "local_time", "integral", "rms"]
  # graph_dict = {
  #   "x": torch.tensor(plane[node_feats].to_numpy()).float(),
  #   "edge_index": torch.tensor(edge[["idx_1", "idx_2"]].to_numpy().T).long(),
  #   "y": torch.tensor(plane["label"].to_numpy()).long(),
  #   "y_edge": torch.tensor(edge["label"].to_numpy()).long()  
  # }
  # out.save(tg.data.Data(**graph_dict))

def process_file(out, fname, p=process_event, l=standard):
  """Process all events in a file into graphs"""
  f = NuMLFile(fname)

  evt = f.get_dataframe("event_table", ["event_id"])
  sp = f.get_dataframe("spacepoint_table")
  hit = f.get_dataframe("hit_table")
#  part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type"])
  part = f.get_dataframe("particle_table")
  edep = f.get_dataframe("edep_table")

  # loop over events in file
  for key in evt.index: p(out, key, sp, hit, part, edep, l)

