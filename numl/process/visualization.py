import pandas as pd, torch, torch_geometric as tg
import numpy as np
import sys
sys.path.append("..")
from labels import *
from labels import standard
from graph import *
import plotly.express as px

def single_plane_graph_vis(key, hit, part, edep, sp, l=standard, e=edges.window_edges):
    """Process an event into graphs"""
    # skip any events with no simulated hits
    if (hit.index==key).sum() == 0: return
    if (edep.index==key).sum() == 0: return

    # get energy depositions, find max contributing particle, and ignore any hits with no truth
    evt_edep = edep.loc[key].reset_index(drop=True)
    evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
    evt_hit = evt_edep.merge(hit.loc[key].reset_index(), on="hit_id", how="inner")

    # skip events with fewer than 50 simulated hits in any plane
    for i in range(3):
        if (evt_hit.global_plane==i).sum() < 50: return

    # get labels for each particle
    evt_part = part.loc[key].reset_index(drop=True)
    evt_part = l.semantic_label(evt_part)

    parent_dict = {0 : ["No Info", "No Info"]}
    for g,t in zip(evt_part['g4_id'], evt_part['type']):
        if g not in parent_dict:
            parent_dict[g] = [t, "No Info"]

    for g,e in zip(evt_hit['g4_id'], evt_hit['energy_fraction']):
        parent_dict[g][1] = e

    # join the dataframes to transform particle labels into hit labels
    evt_hit = evt_hit.merge(evt_part, on="g4_id", how="inner")


    planes_suffixes = [ "_u", "_v", "_y" ]

    evt_sp = sp.loc[key].reset_index(drop=True)

    data = { "n_sp": evt_sp.shape[1] }

    planes = []
    # draw graph edges
    for p, plane in evt_hit.groupby("local_plane"):

        # Reset indices
        plane = plane.reset_index(drop=True).reset_index()

        plane['parent_type'] = plane.apply(lambda row: parent_dict[row['parent_id']][0], axis=1)
        plane['parent_energy_fraction'] = plane.apply(lambda row: parent_dict[row['parent_id']][1], axis=1)

        suffix = planes_suffixes[p]
        # Save to file
        node_feats = ["global_plane", "global_wire", "global_time", "tpc",
          "local_plane", "local_wire", "local_time", "integral", "rms"]
        data["x"+suffix] = torch.tensor(plane[node_feats].to_numpy()).float()
        data["y"+suffix] = torch.tensor(plane["semantic_label"].to_numpy()).long()

        planes.append(plane)

    return planes


def handle_planes(planes_arr):
    particle_dtype = pd.CategoricalDtype(["pion",
      "muon",
      "kaon",
      "hadron",
      "shower",
      "michel",
      "delta",
      "diffuse",
      "invisible"], ordered = True)

    for i in range(3):
        planes_arr[i].semantic_label = pd.Categorical(planes_arr[i].semantic_label).from_codes(codes = planes_arr[i].semantic_label, dtype = particle_dtype)
        planes_arr[i].start_process = planes_arr[i].start_process.map(lambda s : s.decode('utf-8'))
        planes_arr[i].end_process = planes_arr[i].end_process.map(lambda s : s.decode('utf-8'))

    return pd.concat(planes_arr)


def plot_event(df, print_out=True, write=False):
    color_dict = {"pion" : "yellow",
      "muon" : "green",
      "kaon" : "black",
      "hadron" : "blue",
      "shower" : "red",
      "michel" : "purple",
      "delta" : "pink",
      "diffuse" : "orange",
      "invisible" : "white"}
    fig = px.scatter(df, x="local_wire", y="local_time", color="semantic_label", color_discrete_map=color_dict, facet_col="local_plane",
                     hover_data=["g4_id","type", "energy_fraction", "start_process", "end_process", "parent_id", "parent_type", "parent_energy_fraction"])

    fig.update_layout(
        width = 1200,
        height = 500,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    if print_out:
        fig.show()

    if write:
        fig.write_html("events/nue_sample/%i_%i_%i.html" %(planes[0].iloc[0]['run'],planes[0].iloc[0]['subrun'],planes[0].iloc[0]['event']))
