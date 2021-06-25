import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, glob, torch, matplotlib.pyplot as plt

graphs = glob.glob("*.pt")

for graph in graphs:
  data = torch.load(graph)
  numl.plot.graph.plot_edge_score(data, data["y"])
  plt.savefig(osp.splitext(graph)[0]+".png")
  plt.clf()
