import sys
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from mpi4py import MPI

import torch
import torch_geometric as pyg

from .. import io, labels, graph
from .base import ProcessorBase

class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    def __init__(self,
                 file: io.File,
                 labeller: Callable = None,
                 planes: List[str] = ['u','v','y'],
                 node_pos: List[str] = ['local_wire','local_time'],
                 pos_norm: List[float] = [0.3,0.055],
                 node_feats: List[str] = ['integral','rms'],
                 lower_bound: int = 20):

        self.labeller = labeller
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = torch.tensor(pos_norm).float()
        self.node_feats = node_feats
        self.lower_bound = lower_bound

        super(HitGraphProducer, self).__init__(file)

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

    def __call__(self, evt: io.Event) -> Tuple[str, Any]:

        event_id = evt.event_id
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
        for i in range(len(self.planes)):
            if hits[hits.local_plane==i].shape[0] < self.lower_bound:
                return evt.name, None

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
        data['sp'].num_nodes = spacepoints.shape[0]

        # draw graph edges
        for i, plane_hits in hits.groupby('local_plane'):

            p = self.planes[i]
            plane_hits = plane_hits.reset_index(drop=True).reset_index(names='index_2d')

            # node position
            pos = torch.tensor(plane_hits[self.node_pos].values).float()
            data[p].pos = pos * self.pos_norm[None,:]

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

        return evt.name, data
