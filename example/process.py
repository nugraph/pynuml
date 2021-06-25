import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, pandas as pd

f = numl.NuMLFile("nue_slice_x00.txt.h5")

evt = f.get_dataframe("event_table", ["event_id"])
hit = f.get_dataframe("hit_table")
part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type"])
edep = f.get_dataframe("edep_table")

# loop over events in file
for key, _ in evt.iterrows():
  numl.process.hitgraph.ccqe_window_graph(key, ".", hit, part, edep)
