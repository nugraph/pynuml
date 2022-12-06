import sys
from typing import Any, Callable, List, NoReturn, Tuple
import numpy as np
import pandas as pd
from mpi4py import MPI

from .. import io, labels, graph

class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    import torch
    import torch_geometric as pyg

    def __init__(self,
                 file: io.File,
                 labeller: Callable = None,
                 planes: List[str] = ['u','v','y'],
                 node_pos: List[str] = ['local_wire','local_time'],
                 pos_norm: List[float] = [0.3,0.055],
                 node_feats: List[str] = ['integral','rms'],
                 lower_bound: int = 20):

        super(HitGraphProducer, self).__init__(file)

        import torch
        import torch_geometric as pyg

        self.labeller = labeller
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = pos_norm
        self.node_feats = node_feats
        self.lower_bound = lower_bound

    @property
    def columns(self) -> Dict[str, List[str]]:
        groups = {
            'hit_table': ['hit_id','local_plane','local_time','local_wire','integral','rms'],
            'spacepoint_table': ['spacepoint_id','hit_id']
        }
        if self.labeller:
            groups['particle_table'] = ['g4_id','parent_id','type','momentum','start_process','end_process']
            groups['edep_table'] = []
        return groups

    def __call__(self, evt: Any) -> Tuple[str, Any]:

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