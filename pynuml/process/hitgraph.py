import sys
from typing import Callable, List
import numpy as np
import pandas as pd
from mpi4py import MPI

import torch
import torch_geometric as pyg

from .. import io, labels, graph
from ..util import requires_torch, requires_pyg

edep1_t = 0.0
edep2_t = 0.0
hit_merge_t = 0.0
torch_t = 0.0
plane_t = 0.0
label_t = 0.0
edge_t = 0.0
profiling = False

class HitGraphProducer:
    '''Process event into graphs'''

    requires_torch()
    import torch

    requires_pyg()
    import torch_geometric as pyg

    def __init__(self,
                 file: io.File,
                 labeller: Callable = None,
                 planes: List[str] = ['u','v','y'],
                 node_pos: List[str] = ['local_wire','local_time'],
                 pos_norm: List[float] = [0.3,0.055],
                 node_feats: List[str] = ['integral','rms'],
                 lower_bound: int = 20):

        requires_torch()
        import torch

        requires_pyg()
        import torch_geometric as pyg

        self.labeller = labeller
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = pos_norm
        self.node_feats = node_feats
        self.lower_bound = lower_bound

        # callback to add groups and columns to the input file
        file.add_group('hit_table', ['hit_id','local_plane','local_time','local_wire','integral','rms'])
        file.add_group('spacepoint_table', ['spacepoint_id','hit_id'])
        if self.labeller:
            file.add_group('particle_table', ['g4_id','parent_id','type','momentum','start_process','end_process'])
            file.add_group('edep_table')

    def __call__(self,
                 evt: dict) -> pyg.data.HeteroData:

        event_id = evt['index']
        name = f'r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}'

        hits = evt['hit_table']

        if self.labeller:
            edeps = evt['edep_table']
            edeps = edeps.sort_values(by=['energy_fraction'],
                                      ascending=False,
                                      kind='mergesort').drop_duplicates('hit_id')
            hits = edeps.merge(hits, on='hit_id', how='right')
            hits['filter_label'] = ~hits['energy_fraction'].isnull()
            hits = hits.drop('energy_fraction', axis='columns')

        # skip events with fewer than lower_bnd simulated hits in any plane.
        # note that we can't just do a pandas groupby here, because that will
        # skip over any planes with zero hits
        sig = hits[hits['filter_label']] if self.labeller else hits
        for i in range(len(self.planes)):
            if sig[sig.local_plane==i].shape[0] < self.lower_bound:
                return name, None

        # get labels for each particle
        if self.labeller:
            particles = self.labeller(evt['particle_table'])
            hits = hits.merge(particles, on='g4_id', how='left')
    
        spacepoints = evt['spacepoint_table'].reset_index(names='index_3d')

        data = pyg.data.HeteroData()

        # event metadata
        data['metadata'].run = event_id[0]
        data['metadata'].subrun = event_id[1]
        data['metadata'].event = event_id[2]

        # spacepoint nodes
        data['sp'].pos = torch.empty([spacepoints.shape[0], 0])

        # draw graph edges
        for i, plane_hits in hits.groupby('local_plane'):

            p = self.planes[i]
            plane_hits = plane_hits.reset_index(drop=True).reset_index(names='index_2d')

            # node position
            pos = torch.tensor(plane_hits[self.node_pos].values).float()
            norm = torch.tensor(self.pos_norm).float()
            data[p].pos = pos * norm[None,:]

            # node features
            data[p].x = torch.tensor(plane_hits[self.node_feats].values).float()

            # hit indices
            data[p].id = torch.tensor(plane_hits['hit_id'].values).long()

            # 2D edges
            graph.edges.delaunay(data[p])

            # 3D edges
            edge3d = spacepoints.merge(plane_hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                       on=f'hit_id_{p}',
                                       how='inner')
            edge3d = edge3d[[f'index_2d_{p}','index_3d']].values.transpose()
            data[p, 'forms', 'sp'].edge_index = torch.tensor(edge3d).long()

            # truth information
            if self.labeller:
                data[p].y_f = torch.tensor(plane_hits['filter_label'].values).bool()
                filtered = plane_hits[plane_hits['filter_label']]
                data[p].y_s = torch.tensor(filtered['semantic_label'].values).long()
                if data[p].y_s.min() < 0 or data[p].y_s.max() > 7:
                    raise Exception('invalid semantic label found!')
                data[p].y_i = torch.tensor(filtered['instance_label'].values).long()

        # if we have truth information, filter out background spacepoints
        if self.labeller:
            sp_filter = spacepoints
            for p in self.planes:
                sp_filter = sp_filter.merge(hits[['hit_id','filter_label']].add_suffix(f'_{p}'),
                                            on=f'hit_id_{p}',
                                            how='left')
                sp_filter.drop(f'hit_id_{p}', axis='columns', inplace=True) # don't need index any more
            sp_filter.fillna(True, inplace=True) # don't want to filter out based on nan values
            mask = sp_filter[[f'filter_label_{p}' for p in self.planes]].all(axis='columns')
            data['sp'].mask = torch.tensor(mask).bool()

            for p in self.planes:
                plane_graph = pyg.data.Data()
                plane_graph.pos = data[p].pos[data[p].y_f, :]
                data[p].edge_index_filtered = graph.edges.delaunay(plane_graph).edge_index

        return name, data

def process_file(out,
                 fname: str,
                 g,# = process_event,
                 l = labels.standard,
                 e = graph.edges.delaunay,
                 p: str = None,
                 overwrite: bool = True,
                 use_seq_cnt: bool = True,
                 evt_part: int = 2,
                 profile: bool = False) -> None:
    '''Loop over events in file and process each into an ML object'''

    requires_torch()
    import torch

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
        if isinstance(out, io.PTOut): print(f"Output folder: {out.outdir}")
        if isinstance(out, io.H5Out): print(f"Output file: {out.fname}")

    # open input file and read dataset "/event_table/event_id.seq_cnt"
    f = io.File(fname)

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
        if isinstance(out, io.PTOut):
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
            if e == graph.edges.delaunay:
                print("edge indexing delaunay      time ", end='')
            elif e == graph.edges.radius:
                print("edge indexing radius        time ", end='')
            elif e == graph.edges.knn:
                print("edge indexing knn           time ", end='')
            elif e == graph.edges.window:
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
