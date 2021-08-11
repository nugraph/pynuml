def _merge_hits(hit):
    # prepare merged hits to then mask in edge forming
    hit = hit.rename(columns={"index":"idx"})
    edge = hit[["global_wire", "global_time", "idx", "g4_id", "label"]].copy()
    edge["dummy"] = 1
    edge = edge.merge(edge, on="dummy", how="outer", suffixes=["_1","_2"]).drop("dummy", axis=1)
    return edge


def window_edges(hit):
    # form edges

###    hit = hit.rename(columns={"index":"idx"})
###    edge = hit[["global_wire", "global_time", "idx", "g4_id", "label"]].copy()
###    edge["dummy"] = 1
###    edge = edge.merge(edge, on="dummy", how="outer", suffixes=["_1","_2"]).drop("dummy", axis=1)

    edge = _merge_hits(hit)
    mask_id = (edge.idx_1 != edge.idx_2)
    mask_wire = (abs(edge.global_wire_1-edge.global_wire_2) < 5)
    mask_time = (abs(edge.global_time_1-edge.global_time_2) < 50)
    return edge[mask_id & mask_wire & mask_time].copy()


def delaunay(hit):
    # performing delaunay transform
    from scipy.spatial import Delaunay
    import torch, pandas as pd
    from torch_geometric.utils import to_undirected

    hit = hit.sort_values(by='index')
    points = hit[['global_wire', 'global_time']].to_numpy(copy=True)
    face = torch.from_numpy(Delaunay(points, qhull_options='QJ').simplices.transpose()).long()
    edge_index = to_undirected(torch.cat([face[:2], face[1:], face[::2]], dim=1)).T.numpy()

    # merging hits to form edges
    edge = _merge_hits(hit)

    # cross-joining merged hits with delaunay edges
    edge["dummy"] = 1
    edge_index_df = pd.DataFrame(data=edge_index, columns=['edge_1', 'edge_2'])
    edge_index_df["dummy"] = 1
    edge = edge.merge(edge_index_df, on='dummy', how='outer').drop('dummy', axis=1)

    # selecting only delaunay edges
    mask_id = (edge.idx_1 != edge.idx_2)
    mask_edge_1 = (edge.idx_1 == edge.edge_1)
    mask_edge_2 = (edge.idx_2 == edge.edge_2)

    return edge[mask_id & mask_edge_1 & mask_edge_2].drop(['edge_1', 'edge_2'], axis=1).copy()


def radius_graph(hit):
    # specify the radius on the graph
    RADIUS = 16

    # forming edges within the distance specified in RADIUS
    edge = _merge_hits(hit)
    mask_id = (edge.idx_1 != edge.idx_2)
    mask_radius = (edge.global_wire_1-edge.global_wire_2)**2 +\
         (edge.global_time_1-edge.global_time_2) <= RADIUS**2

    return edge[mask_id & mask_radius].copy()


def knn_graph(hit):
    # specify the number of neighbours
    K = 6
    
    # constructing KNN graph edges
    import torch, torch_geometric as tg
    from torch_geometric.utils import to_undirected

    hit = hit.sort_values(by='index')
    points = hit[['global_wire', 'global_time']].to_numpy(copy=True)
    edge_index = to_undirected(tg.nn.knn_graph(
        torch.from_numpy(points),
        k=K
    )).T.numpy()
    
    # merging hits to form edges
    edge = _merge_hits(hit)

    # cross-joining merged hits with KNN graph edges
    edge["dummy"] = 1
    edge_index_df = pd.DataFrame(data=edge_index, columns=['edge_1', 'edge_2'])
    edge_index_df["dummy"] = 1
    edge = edge.merge(edge_index_df, on='dummy', how='outer').drop('dummy', axis=1)

    # selecting only KNN graph edges
    mask_id = (edge.idx_1 != edge.idx_2)
    mask_edge_1 = (edge.idx_1 == edge.edge_1)
    mask_edge_2 = (edge.idx_2 == edge.edge_2)

    return edge[mask_id & mask_edge_1 & mask_edge_2].drop(['edge_1', 'edge_2'], axis=1).copy()