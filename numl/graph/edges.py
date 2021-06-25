def window_edges(hit):
    # form edges
    hit = hit.rename(columns={"index":"idx"})
    edge = hit[["global_wire", "global_time", "idx", "g4_id", "label"]].copy()
    edge["dummy"] = 1
    edge = edge.merge(edge, on="dummy", how="outer", suffixes=["_1","_2"]).drop("dummy", axis=1)
    mask_id = (edge.idx_1 != edge.idx_2)
    mask_wire = (abs(edge.global_wire_1-edge.global_wire_2) < 5)
    mask_time = (abs(edge.global_time_1-edge.global_time_2) < 50)
    return edge[mask_id & mask_wire & mask_time].copy()
