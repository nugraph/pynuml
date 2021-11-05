import pandas as pd, torch, torch_geometric as tg
import numpy as np
import sys
sys.path.append("..")
from labels import *
from labels import standard
from graph import *
from core.file import NuMLFile
import plotly.express as px
    
def single_plane_graph_vis(evt, l=standard):
    """Process an event into graphs"""

    # get energy depositions, find max contributing particle, and ignore any hits with no truth
    evt_edep = evt["edep_table"]
    evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
    evt_hit = evt_edep.merge(evt["hit_table"], on="hit_id", how="inner")

    # skip events with fewer than 50 simulated hits in any plane
    for i in range(3):
        if (evt_hit.global_plane==i).sum() < 50: return

    # get labels for each particle
    evt_part = evt["particle_table"]
    evt_part = l.semantic_label(evt_part)

    parent_dict = {0 : ["No Info", "No Info"]}
    for g,t,m in zip(evt_part['g4_id'], evt_part['type'], evt_part['momentum']):
        if g not in parent_dict:
            parent_dict[g] = [t, m]

    # join the dataframes to transform particle labels into hit labels
    evt_hit = evt_hit.merge(evt_part, on="g4_id", how="inner")

    planes = []
    for p, plane in evt_hit.groupby("local_plane"):

        # Reset indices
        plane = plane.reset_index(drop=True).reset_index()

        plane['parent_type'] = plane.apply(lambda row: parent_dict[row['parent_id']][0], axis=1)
        plane['parent_momentum'] = plane.apply(lambda row: parent_dict[row['parent_id']][1], axis=1)

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


def plot_event(df, print_out=True, write=None):
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
                     hover_data=["g4_id","type", "momentum", "start_process", "end_process", "parent_id", "parent_type", "parent_momentum"])

    fig.update_layout(
        width = 1200,
        height = 500,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    if print_out:
        fig.show()

    if write:
        fig.write_html(f"{write}/{df[0].iloc[0]['run']}_{df[0].iloc[0]['subrun']}_{df[0].iloc[0]['event']}.html")

def concat_events(fname):
    f = NuMLFile(fname)
    evt = f.get_dataframe("event_table", ["event_id"])
    hit = f.get_dataframe("hit_table")
    part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type", "momentum", "start_process", "end_process"])
    edep = f.get_dataframe("edep_table")
    sp = f.get_dataframe("spacepoint_table")
    
    data = pd.DataFrame()
    
    for i,key in enumerate(evt.index):
        planes = single_plane_graph_vis(key, hit, part, edep, sp)
        if planes:
            planes = handle_planes(planes)
            data = pd.concat([data, planes], ignore_index=True)
            
    return data

def label_counts(data):
    counts = data['semantic_label'].value_counts()
    ax = counts.plot(kind='bar')
    ax.set_ylabel("count")

def histogram_slice(data, metric, log_scale=False, write=False):
    color_dict = {"pion" : "yellow",
      "muon" : "green",
      "kaon" : "black",
      "hadron" : "blue",
      "shower" : "red",
      "michel" : "purple",
      "delta" : "pink",
      "diffuse" : "orange",
      "invisible" : "white"}

    fig = px.histogram(data, x=metric, color="semantic_label", color_discrete_map=color_dict, facet_col="local_plane", log_y=log_scale)

    fig.update_layout(
        width = 1200,
        height = 500,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    fig.show()
    
    if write: fig.write_html("hist_nue.html")
