import torch, matplotlib as mpl
import matplotlib.pyplot as plt, matplotlib.collections as mc

def _format():
  """Default plot formatting"""
  mpl.rc('font', weight='bold', size=36)

def _get_lines(g, score):
  """Take a g object and return a list of LineCollection objects, one per class"""
  _format()
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

def plot_edge_score(g, y):
  """Plot graph edges, colour-coded by edge score"""
  _format()
  fig, ax = plt.subplots(figsize=[16,9])
  lcs = _get_lines(g, y)
  for lc in lcs: ax.add_collection(lc)
  ax.autoscale()
  plt.xlabel("Wire")
  plt.ylabel("Time tick")
  plt.tight_layout()

def plot_edge_diff(g, y):
  """Plot graph edges, highlighting edges that were misclassified"""
  _format()
  fig, ax = plt.subplots(figsize=[16,9])
  y = (y != g.y)
  lcs = _get_lines(g, y)
  for lc in lcs: ax.add_collection(lc)
  ax.autoscale()
  plt.tight_layout()
