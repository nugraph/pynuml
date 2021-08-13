def window(data, wire_distance=5, time_distance=50):
    '''Form graph edges forming edges in a given window around each node'''
    import torch 

    edge_index = []
    for node_1 in range(len(data)):
        for node_2 in range(len(data)):
            if node_1 == node_2: continue
            if abs(data.pos[node_1][0] - data.pos[node_2][0]) < wire_distance and\
                abs(data.pos[node_1][1] - data.pos[node_2][1] < time_distance):
                edge_index.append([node_1, node_2])

    edge_index = torch.tensor(edge_index).T
    
    data.edge_index = edge_index
    return data


def delaunay(data):
  '''Form graph edges using Delaunay triangulation'''
  import torch, torch_geometric as tg
  # data = tg.data.Data(pos=torch.tensor(hits[["global_wire", "global_time"]].values).float())
  return tg.transforms.FaceToEdge()(tg.transforms.Delaunay()(data))


def radius(data, r=16, max_num_neighbours=32):
    '''Form graph edges using Radius Graph transformation'''
    import torch, torch_geometric as tg
    return tg.transforms.RadiusGraph(r=r, max_num_neighbours=max_num_neighbours)(data)


def knn(data, k=6):
    '''Form graph edges using KNN Graph transformation'''
    import torch, torch_geometric as tg
    return tg.transforms.KNNGraph(k=k)(data)