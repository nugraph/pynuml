import sys
from typing import Any, Callable
import numpy as np
import pandas as pd
from mpi4py import MPI

import torch
import torch_geometric as pyg

from .base import ProcessorBase

class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    def __init__(self,
                 file: 'pynuml.io.File',
                 labeller: Callable = None,
                 planes: list[str] = ['u','v','y'],
                 node_pos: list[str] = ['local_wire','local_time'],
                 pos_norm: list[float] = [0.3,0.055],
                 node_feats: list[str] = ['integral','rms'],
                 lower_bound: int = 20,
                 filter_hits: bool = False):

        self.labeller = labeller
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = torch.tensor(pos_norm).float()
        self.node_feats = node_feats
        self.lower_bound = lower_bound
        self.filter_hits = filter_hits

        self.transform = pyg.transforms.Compose((
            pyg.transforms.Delaunay(),
            pyg.transforms.FaceToEdge()))

        super(HitGraphProducer, self).__init__(file)

    @property
    def columns(self) -> dict[str, list[str]]:
        groups = {
            'hit_table': ['hit_id','local_plane','local_time','local_wire','integral','rms'],
            'spacepoint_table': ['spacepoint_id','hit_id']
        }
        if self.labeller:
            groups['particle_table'] = ['g4_id','parent_id','type','momentum','start_process','end_process']
            groups['edep_table'] = []
        return groups

    @property
    def metadata(self):
        metadata = { 'planes': self.planes }
        if self.labeller is not None:
            metadata['classes'] = self.labeller.labels[:-1]
        return metadata

    def __call__(self, evt: 'pynuml.io.Event') -> tuple[str, Any]:

        event_id = evt.event_id
        name = f'r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}'

        hits = evt['hit_table']
        spacepoints = evt['spacepoint_table'].reset_index(drop=True)

        # handle energy depositions
        if self.filter_hits or self.labeller:
            edeps = evt['edep_table']
            energy_col = 'energy' if 'energy' in edeps.columns else 'energy_fraction' # for backwards compatibility
            edeps = edeps.sort_values(by=[energy_col],
                                      ascending=False,
                                      kind='mergesort').drop_duplicates('hit_id')
            hits = edeps.merge(hits, on='hit_id', how='right')

            # if we're filtering out data hits, do that
            if self.filter_hits:
                hitmask = hits[energy_col].isnull()
                filtered_hits = hits[hitmask].hit_id.tolist()
                hits = hits[~hitmask].reset_index(drop=True)
                # filter spacepoints from noise
                cols = [ f'hit_id_{p}' for p in self.planes ]
                spmask = spacepoints[cols].isin(filtered_hits).any(axis='columns')
                spacepoints = spacepoints[~spmask].reset_index(drop=True)

            hits['filter_label'] = ~hits[energy_col].isnull()
            hits = hits.drop(energy_col, axis='columns')

        # reset spacepoint index
        spacepoints = spacepoints.reset_index(names='index_3d')

        # skip events with fewer than lower_bnd simulated hits in any plane.
        # note that we can't just do a pandas groupby here, because that will
        # skip over any planes with zero hits
        for i in range(len(self.planes)):
            planehits = hits[hits.local_plane==i]
            nhits = planehits.filter_label.sum() if self.labeller else planehits.shape[0]
            if nhits < self.lower_bound:
                return evt.name, None

        # get labels for each particle
        if self.labeller:
            particles = self.labeller(evt['particle_table'])
            hits = hits.merge(particles, on='g4_id', how='left')
            mask = (~hits.g4_id.isnull()) & (hits.semantic_label.isnull())
            if mask.any():
                print(f'found {mask.sum()} orphaned hits.')
                return evt.name, None
            del mask

        data = pyg.data.HeteroData()

        # event metadata
        data['metadata'].run = event_id[0]
        data['metadata'].subrun = event_id[1]
        data['metadata'].event = event_id[2]

        # spacepoint nodes
        data['sp'].x = torch.empty(spacepoints.shape[0], 0)

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
            self.transform(data[p])
            data[p, 'plane', p].edge_index = data[p].edge_index
            del data[p].edge_index

            # 3D edges
            edge3d = spacepoints.merge(plane_hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                       on=f'hit_id_{p}',
                                       how='inner')
            edge3d = edge3d[[f'index_2d_{p}','index_3d']].values.transpose()
            edge3d = torch.tensor(edge3d) if edge3d.size else torch.empty((2,0))
            data[p, 'nexus', 'sp'].edge_index = edge3d.long()

            # truth information
            if self.labeller:
                data[p].y_f = torch.tensor(plane_hits['filter_label'].values).bool()
                filtered = plane_hits[plane_hits['filter_label']]
                data[p].y_s = torch.tensor(filtered['semantic_label'].values).long()
                self.labeller.validate(data[p].y_s)
                data[p].y_i = torch.tensor(filtered['instance_label'].values).long()

        return evt.name, data
