def window(data, wire_distance=3, time_distance=3):
  '''Form graph edges forming edges in a given window around each node'''
  import torch, pandas as pd

  df = pd.DataFrame(data.pos.numpy(), columns=["wire", "time"]).reset_index()
  df["dummy"] = 1
  df = df.merge(df, on="dummy", how="outer", suffixes=["_1","_2"]).drop("dummy", axis="columns")
  df = df[(abs(df.wire_1-df.wire_2)<wire_distance) & (abs(df.time_1-df.time_2)<time_distance) & (df.index_1 != df.index_2)]
  data.edge_index = torch.tensor(df[["index_1", "index_2"]].transpose().to_numpy())
  return data

def delaunay(data):
  '''Form graph edges using Delaunay triangulation'''
  import torch, torch_geometric as tg
  return tg.transforms.FaceToEdge()(tg.transforms.Delaunay()(data))


def radius(data, r=16, max_num_neighbours=32):
  '''Form graph edges using Radius Graph transformation'''
  import torch, torch_geometric as tg
  return tg.transforms.RadiusGraph(r=r, max_num_neighbors=max_num_neighbours)(data)


def knn(data, k=6):
  '''Form graph edges using KNN Graph transformation'''
  import torch, torch_geometric as tg
  return tg.transforms.KNNGraph(k=k)(data)
