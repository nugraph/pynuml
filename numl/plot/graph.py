import torch, matplotlib as mpl, seaborn as sns, numpy as np
import matplotlib.pyplot as plt, matplotlib.collections as mc

def _init():
  """Default font formatting"""
  mpl.rc('font', weight='bold', size=36)
  return plt.subplots(figsize=[16,9])

def _format(ax):
  """Default plot formatting"""
  ax.autoscale()
  plt.xlabel("Wire")
  plt.ylabel("Time tick")
  plt.tight_layout()

def _get_lines(g, score):
  """Take a g object and return a list of LineCollection objects, one per class"""
  # wire = g.x[:,1]
  # time = g.x[:,2]
  wire = g["x"][:,1]
  time = g["x"][:,2]
  # lines = [ [ [ wire[edge[0]], time[edge[0]] ], [ wire[edge[1]], time[edge[1]] ] ] for edge in g.edge_index.T ]
  lines = [ [ [ wire[edge[0]], time[edge[0]] ], [ wire[edge[1]], time[edge[1]] ] ] for edge in g["edge_index"].T ]
  lines_class = [ [], [], [], [] ]
  colours = ['gainsboro', 'red', 'green', 'blue' ]
  for l, y in zip(lines, score): lines_class[y].append(l)
  return [ mc.LineCollection(lines_class[i], colors=colours[i], linewidths=2, zorder=1) for i in range(len(colours)) ]

def plot_node_score(g, y):
  """Plot graph nodes, colour-coded by node label"""
  fig, ax = _init()
  c = np.array(sns.color_palette())[y]
  plt.scatter(g["x"][:,1], g["x"][:,2], c=c, s=8)
  _format(ax)

def plot_edge_score(g, y):
  """Plot graph edges, colour-coded by edge score"""
  fig, ax = _init()
  lcs = _get_lines(g, y)
  for lc in lcs: ax.add_collection(lc)
  _format(ax)

def plot_edge_diff(g, y):
  """Plot graph edges, highlighting edges that were misclassified"""
  fig, ax = _init()
  y = (y != g.y)
  lcs = _get_lines(g, y)
  for lc in lcs: ax.add_collection(lc)
  _format(ax)

