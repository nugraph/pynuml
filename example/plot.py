
if __name__ == "__main__":
  import sys, os.path as osp
  sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
  import numl, glob, torch, matplotlib.pyplot as plt

  graphs = glob.glob(sys.argv[1]+"/*")

  for graph in graphs:
    print(f"plotting graph {osp.basename(graph)}")
    data = torch.load(graph)
    name = osp.splitext(osp.basename(graph))[0]
    numl.plot.graph.plot_edge_score(data, data["y_edge"])
    plt.savefig(name+"_edge.png")
    plt.close()
    numl.plot.graph.plot_node_score(data, data["y"])
    plt.savefig(name+"_node.png")
    plt.close()
