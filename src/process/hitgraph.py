import pandas as pd
import pynuml
from mpi4py import MPI
import numpy as np
import sys

edep1_t = 0.0
edep2_t = 0.0
hit_merge_t = 0.0
torch_t = 0.0
plane_t = 0.0
label_t = 0.0
edge_t = 0.0
profiling = False

def process_event(event_id, evt, l, e, lower_bnd=20, **edge_args):
    """Process an event into graphs"""
    pynuml.util.requires_torch()
    pynuml.util.requires_pyg()

    # skip any events with no simulated hits

    global edep1_t, edep2_t, hit_merge_t, torch_t, plane_t, label_t, edge_t
    global profiling
    if profiling:
        start_t = MPI.Wtime()

    # get energy depositions, find max contributing particle, and ignore any hits with no truth
    evt_edep = evt["edep_table"]
    evt_edep = evt_edep.sort_values(by=['energy_fraction'], ascending=False, kind='mergesort').drop_duplicates('hit_id')

    if profiling:
        end_t = MPI.Wtime()
        edep1_t += end_t - start_t
        start_t = end_t

    evt_hit = evt_edep.merge(evt["hit_table"], on="hit_id", how="right")
    evt_hit['is_cosmic'] = False
    evt_hit.loc[evt_hit['energy_fraction'].isnull(), 'is_cosmic'] = True
    evt_hit = evt_hit.drop("energy_fraction", axis=1)

    if profiling:
        end_t = MPI.Wtime()
        edep2_t += end_t - start_t
        start_t = end_t

    # skip events with fewer than lower_bnd simulated hits in any plane
    for i in range(3): 
        #filter out cosmics
        if (evt_hit[~evt_hit['is_cosmic']].global_plane==i).sum() < lower_bnd: return

    # get labels for each particle
    evt_part = l(evt["particle_table"])

    if profiling:
        end_t = MPI.Wtime()
        label_t += end_t - start_t
        start_t = end_t

    # join the dataframes to transform particle labels into hit labels
    evt_hit = evt_hit.merge(evt_part, on="g4_id", how="left")

    if profiling:
        end_t = MPI.Wtime()
        hit_merge_t += end_t - start_t
        start_t = end_t

    planes = [ "_u", "_v", "_y" ]
    evt_sp = evt["spacepoint_table"]
    sim_hits = evt_hit["hit_id"].tolist()
    data = { "n_sp": evt_sp.shape[0] }

    # draw graph edges
    for p, plane in evt_hit.groupby("local_plane"):

        # Reset indices
        plane = plane.reset_index(drop=True).reset_index()

        # build 3d edges
        suffix = planes[p]
        plane_sp = evt_sp.rename(columns={"hit_id"+suffix: "hit_id"}).reset_index()
        plane_sp = plane_sp[(plane_sp.hit_id != -1)]
        k3d = ["index","hit_id"]
        edges_3d = pd.merge(plane_sp[k3d], plane[(plane.hit_id != -1)][k3d], on="hit_id", how="inner", suffixes=["_3d", "_2d"])
        blah = edges_3d[["index_2d", "index_3d"]].to_numpy()
     
        if profiling:
            end_t = MPI.Wtime()
            plane_t += end_t - start_t
            start_t = end_t

        # Save to file
        tmp = pyg.data.Data(
            pos=torch.tensor(plane[["global_wire", "global_time"]].values) * torch.tensor(np.array([0.3, 0.055]))[None, :]
        )
        node_feats = ["global_plane", "global_wire", "global_time", "tpc",
            "local_plane", "local_wire", "local_time", "integral", "rms"]
        data["x"+suffix] = torch.tensor(plane[node_feats].to_numpy()).float()
        data["y_c"+suffix] = torch.tensor(plane["is_cosmic"].to_numpy()).bool()
        data["y_s"+suffix] = torch.tensor(plane["semantic_label"].to_numpy()).long()[~data["y_c"+suffix]]
        data["y_i"+suffix] = torch.tensor(plane["instance_label"].to_numpy()).long()[~data["y_c"+suffix]]

        if profiling:
            end_t = MPI.Wtime()
            torch_t += end_t - start_t
            start_t = end_t

        tmp = e(tmp, **edge_args)
        data["edge_index"+suffix] = tmp.edge_index
        data["edge_index_3d"+suffix] = torch.tensor(blah).transpose(0, 1).long()

        if profiling:
            end_t = MPI.Wtime()
            edge_t += end_t - start_t
            start_t = end_t

    if data["y_s_u"].max() > 7 or data["y_s_v"].max() > 7 or data["y_s_y"].max() > 7:
        print("\n  error: hit with invisible label found! skipping event\n")
        return []

    return [[f"r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}", pyg.data.Data(**data)]]

def process_file(out, fname, g=process_event, l=pynuml.labels.standard,
    e=pynuml.graph.edges.delaunay, p=None, overwrite=True, use_seq_cnt=True, evt_part=2, profile=False):

    pynuml.util.requires_torch()

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    global profiling
    profiling = profile
    if profiling:
        comm.Barrier()
        start_t = MPI.Wtime()
        timing = start_t

    """Process all events in a file into graphs"""
    if rank == 0:
        print("------------------------------------------------------------------")
        print(f"Processing input file: {fname}")
        if isinstance(out, pynuml.core.PTOut): print(f"Output folder: {out.outdir}")
        if isinstance(out, pynuml.core.H5Out): print(f"Output file: {out.fname}")

    # open input file and read dataset "/event_table/event_id.seq_cnt"
    f = pynuml.File(fname)

    if profiling:
        open_time = MPI.Wtime() - timing
        comm.Barrier()
        timing = MPI.Wtime()

    # only use the following groups and datasets in them
    f.add_group("hit_table")
    f.add_group("particle_table", ["g4_id", "parent_id", "type", "momentum", "start_process", "end_process"])
    f.add_group("edep_table")
    f.add_group("spacepoint_table")

    # number of unique event IDs in the input file
    event_id_len = len(f)

    # Read all data associated with event IDs assigned to this process
    # Data read is stored as a python nested dictionary, f._data. Keys are group
    # names, values are python dictionaries, each has names of dataset in that
    # group as keys, and values storing dataset subarrays
    f.read_data_all(use_seq_cnt, evt_part, profiling)

    if profiling:
        read_time = MPI.Wtime() - timing
        comm.Barrier()
        timing = MPI.Wtime()

    # organize the assigned event data into a python list, evt_list, so data
    # corresponding to one event ID can be used to create a graph. Each element
    # in evt_list is a Pandas DataFrame. A graph will be created using data
    # stored in a Pandas dataframe.
    evt_list = f.build_evt()
    # print("len(evt_list)=", len(evt_list))

    if profiling:
        build_list_time = MPI.Wtime() - timing
        comm.Barrier()
        write_time   = 0
        graph_time   = 0
        num_evts     = 0            # no. events assigned to this process
        evt_size_max = 0            # max event size assigned to this process
        evt_size_min = sys.maxsize  # min event size assigned to this process
        evt_size_sum = 0            # sum of event sizes assigned to this process
        num_grps     = 0            # no. graphs created by this process
        grp_size_max = 0            # max graph size created by this process
        grp_size_min = sys.maxsize  # min graph size created by this process
        grp_size_sum = 0            # sum of graph sizes created by this process

    # Iterate through event IDs, construct graphs and save them in files
    for i in range(len(evt_list)):
        if profiling:
            timing = MPI.Wtime()
            evt_size = 0
            for group in evt_list[i].keys():
                if group != "index":
                    # size in bytes of a Pandas DataFrame
                    evt_size += sys.getsizeof(evt_list[i][group])
            num_evts += 1
            evt_size_sum += evt_size
            if evt_size > evt_size_max : evt_size_max = evt_size
            if evt_size < evt_size_min : evt_size_min = evt_size

        # retrieve event sequence ID
        idx = evt_list[i]["index"]
        event_id = f.index(idx)

        # avoid overwriting to already existing files
        if isinstance(out, pynuml.core.PTOut):
            import os.path as osp, os
            out_file = f"{out.outdir}/r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}.pt"
            if osp.exists(out_file):
                if overwrite:
                    os.remove(out_file)
                else:
                    print(f"Error: file already exists: {out_file}")
                    sys.stdout.flush()
                    MPI.COMM_WORLD.Abort(1)

        # create graphs using data in evt_list[i], a Pandas DataFrame
        # Note an event may create more than one graph
        tmp = g(event_id, evt_list[i], l, e)

        if profiling:
            graph_time += MPI.Wtime() - timing
            timing = MPI.Wtime()

        if tmp is not None:
            for name, data in tmp:
                # print("saving", name)
                out.save(data, name)
                if profiling:
                    grp_size = 0
                    for key, val in data:
                        # calculate size in bytes of val
                        if (isinstance(val, torch.Tensor)):
                            # val is a pytorch tensor
                            grp_size += val.element_size() * val.nelement()
                        else:
                            grp_size += sys.getsizeof(val)
                    num_grps += 1
                    grp_size_sum += grp_size
                    if grp_size > grp_size_max : grp_size_max = grp_size
                    if grp_size < grp_size_min : grp_size_min = grp_size

        if profiling:
            write_time += MPI.Wtime() - timing

    if profiling:
        total_time = MPI.Wtime() - start_t

        global edep1_t, edep2_t, hit_merge_t, torch_t, plane_t, label_t, edge_t

        my_t = np.array([open_time, read_time, build_list_time,
                                         graph_time, write_time, total_time, edep1_t, edep2_t,
                                         label_t, hit_merge_t, plane_t, torch_t, edge_t])
        all_t  = None
        if rank == 0:
            all_t  = np.empty([nprocs, 14], dtype=np.double)

        # root process gathers all timings from all processes
        comm.Gather(my_t, all_t, root=0)

        if rank == 0:
            # transport to 14 x nprocs
            all_t = all_t.transpose(1, 0)
            # sort along each row in order to get MAX, MIN, and Median
            all_t = np.sort(all_t)

        local_counts  = np.empty(6, dtype=np.int64)
        global_counts = np.empty(6, dtype=np.int64)

        local_counts[0] = num_evts
        local_counts[1] = evt_size_max
        local_counts[2] = num_grps
        local_counts[3] = grp_size_max
        local_counts[4] = evt_size_sum
        local_counts[5] = grp_size_sum
        comm.Reduce(local_counts, global_counts, op=MPI.MAX, root=0)
        num_evts_max     = global_counts[0]
        evt_size_max     = global_counts[1]
        num_grps_max     = global_counts[2]
        grp_size_max     = global_counts[3]
        evt_size_sum_max = global_counts[4]
        grp_size_sum_max = global_counts[5]

        local_counts[0] = num_evts
        local_counts[1] = evt_size_min
        local_counts[2] = num_grps
        local_counts[3] = grp_size_min
        local_counts[4] = evt_size_sum
        local_counts[5] = grp_size_sum
        comm.Reduce(local_counts, global_counts, op=MPI.MIN, root=0)
        num_evts_min     = global_counts[0]
        evt_size_min     = global_counts[1]
        num_grps_min     = global_counts[2]
        grp_size_min     = global_counts[3]
        evt_size_sum_min = global_counts[4]
        grp_size_sum_min = global_counts[5]

        local_counts[0] = num_evts
        local_counts[1] = evt_size_sum
        local_counts[2] = num_grps
        local_counts[3] = grp_size_sum
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        num_evts     = global_counts[0]
        evt_size_sum = global_counts[1]
        num_grps     = global_counts[2]
        grp_size_sum = global_counts[3]

        if rank == 0:
            print("------------------------------------------------------------------")
            print("Number of MPI processes          =%8d" % nprocs)
            print("Total no. event IDs              =%8d" % event_id_len)
            print("Total no. non-empty events       =%8d" % num_evts)
            print("Size of all events               =%10.1f MiB" % (evt_size_sum/1048576.0))
            if evt_part == 0:
                print("== Use event ID based data partitioning strategy ==")
            elif evt_part == 1:
                print("== Use event data amount based data partitioning strategy ==")
            elif evt_part == 2:
                print("== Use events in particle table to partition ==")
            if use_seq_cnt:
                print("== Use dataset 'event_id.seq_cnt' to calculate data partitioning ==")
            else:
                print("== Use dataset 'event_id.seq' to calculate data partitioning ==")
            print("Local no. events assigned     MAX=%8d   MIN=%8d   AVG=%10.1f"
                        % (num_evts_max, num_evts_min,num_evts/nprocs))
            print("Local indiv event size in KiB MAX=%10.1f MIN=%10.1f AVG=%10.1f"
                        % (evt_size_max/1024.0, evt_size_min/1024.0, evt_size_sum/1024.0/num_evts))
            print("Local sum   event size in MiB MAX=%10.1f MIN=%10.1f AVG=%10.1f"
                        % (evt_size_sum_max/1048576.0, evt_size_sum_min/1048576.0, evt_size_sum/1048576.0/nprocs))
            print("(MAX and MIN timings are among %d processes)" % nprocs)
            print("------------------------------------------------------------------")
            print("Total no.  of graphs             =%8d" % num_grps)
            print("Size of all graphs               =%10.1f MiB" % (grp_size_sum/1048576.0))
            print("Local no. graphs created      MAX=%8d   MIN=%8d   AVG=%10.1f"
                        % (num_grps_max, num_grps_min, num_grps/nprocs))
            print("Local indiv graph size in KiB MAX=%10.1f MIN=%10.1f AVG=%10.1f"
                        % (grp_size_max/1024.0, grp_size_min/1024.0,grp_size_sum/1024.0/num_grps))
            print("Local sum   graph size in MiB MAX=%10.1f MIN=%10.1f AVG=%10.1f"
                        % (grp_size_sum_max/1048576.0, grp_size_sum_min/1048576.0, grp_size_sum/1048576.0/nprocs))
            print("(MAX and MIN timings are among %d processes)" % nprocs)
            print("---- Timing break down of graph creation phase (in seconds) ------")
            sort_t = all_t[6]
            print("edep grouping               time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[7]
            print("edep merge                  time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[8]
            print("labelling                   time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[9]
            print("hit_table merge             time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[10]
            print("plane build                 time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[11]
            print("torch_geometric             time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[12]
            if e == pynuml.graph.edges.delaunay:
                print("edge indexing delaunay      time ", end='')
            elif e == pynuml.graph.edges.radius:
                print("edge indexing radius        time ", end='')
            elif e == pynuml.graph.edges.knn:
                print("edge indexing knn           time ", end='')
            elif e == pynuml.graph.edges.window:
                print("edge indexing window        time ", end='')
            else:
                print("edge indexing               time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            print("---- Top-level timing breakdown (in seconds) ---------------------")
            sort_t = all_t[0]
            print("file open                   time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[1]
            print("read from file              time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[2]
            print("build dataframe             time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[3]
            print("graph creation              time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[4]
            print("write to files              time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            sort_t = all_t[5]
            print("total                       time ", end='')
            print("MAX=%8.2f  MIN=%8.2f  MID=%8.2f" % (sort_t[nprocs-1], sort_t[0], sort_t[nprocs//2]))
            print("(MAX and MIN timings are among %d processes)" % nprocs)
