from typing import Any, Callable
import numpy as np
import pandas as pd

import torch
import torch_geometric as pyg

from .base import ProcessorBase

class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    def __init__(self,
                 file: 'pynuml.io.File',
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 planes: list[str] = ['u','v','y'],
                 node_pos: list[str] = ['local_wire','local_time'],
                 pos_norm: list[float] = [0.3,0.055],
                 node_feats: list[str] = ['integral','rms'],
                 lower_bound: int = 20,
                 filter_hits: bool = False):

        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = torch.tensor(pos_norm).float()
        self.node_feats = node_feats
        self.lower_bound = lower_bound
        self.filter_hits = filter_hits

        self.transform = pyg.transforms.Compose((
            pyg.transforms.Delaunay(),
            pyg.transforms.FaceToEdge()))

        super().__init__(file)

    @property
    def columns(self) -> dict[str, list[str]]:
        groups = {
            'hit_table': ['hit_id','local_plane','local_time','local_wire','integral','rms'],
            'spacepoint_table': ['spacepoint_id','hit_id']
        }
        if self.semantic_labeller:
            groups['particle_table'] = ['g4_id','parent_id','type','momentum','start_process','end_process']
            groups['edep_table'] = []
        if self.event_labeller:
            groups['event_table'] = ['is_cc', 'nu_pdg']
        if self.label_vertex:
            keys = ['nu_vtx_corr','nu_vtx_wire_pos','nu_vtx_wire_time']
            if 'event_table' in groups:
                groups['event_table'].extend(keys)
            else:
                groups['event_table'] = keys
        return groups

    @property
    def metadata(self):
        metadata = { 'planes': self.planes }
        if self.semantic_labeller is not None:
            metadata['semantic_classes'] = self.semantic_labeller.labels[:-1]
        if self.event_labeller is not None:
            metadata['event_classes'] = self.event_labeller.labels
        return metadata

    def __call__(self, evt: 'pynuml.io.Event') -> tuple[str, Any]:

        event_id = evt.event_id
        name = f'r{event_id[0]}_sr{event_id[1]}_evt{event_id[2]}'

        if self.event_labeller or self.label_vertex:
            event = evt['event_table'].squeeze()

        hits = evt['hit_table']
        spacepoints = evt['spacepoint_table'].reset_index(drop=True)

        # discard any events with pathologically large hit integrals
        # this is a hotfix that should be removed once the dataset is fixed
        if hits.integral.max() > 1e6:
            print('found event with pathologically large hit integral, skipping')
            return evt.name, None

        # handle energy depositions
        if self.filter_hits or self.semantic_labeller:
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
            nhits = planehits.filter_label.sum() if self.semantic_labeller else planehits.shape[0]
            if nhits < self.lower_bound:
                return evt.name, None

        # get labels for each particle
        if self.semantic_labeller:
            particles = self.semantic_labeller(evt['particle_table'])
            try:
                hits = hits.merge(particles, on='g4_id', how='left')
            except:
                print('exception occurred when merging hits and particles')
                print('hit table:', hits)
                print('particle table:', particles)
                print('skipping this event')
                return evt.name, None
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
            data[p, 'plane', p].edge_index = self.transform(data[p]).edge_index

            # 3D edges
            edge3d = spacepoints.merge(plane_hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                       on=f'hit_id_{p}',
                                       how='inner')
            edge3d = edge3d[[f'index_2d_{p}','index_3d']].values.transpose()
            edge3d = torch.tensor(edge3d) if edge3d.size else torch.empty((2,0))
            data[p, 'nexus', 'sp'].edge_index = edge3d.long()

            # truth information
            if self.semantic_labeller:
                data[p].y_semantic = torch.tensor(plane_hits['semantic_label'].fillna(-1).values).long()
                data[p].y_instance = torch.tensor(plane_hits['instance_label'].fillna(-1).values).long()
            if self.label_vertex:
                vtx_2d = torch.tensor([ event[f'nu_vtx_wire_pos_{i}'], event.nu_vtx_wire_time ]).float()
                data[p].y_vtx = vtx_2d * self.pos_norm[None,:]

        # event label
        if self.event_labeller:
            data['evt'].y = torch.tensor(self.event_labeller(event)).long()

        # save particle tree
        data['particles'].id = torch.tensor(particles.g4_id).int()
        data['particles'].parent = torch.tensor(particles.parent_id).int()
        data['particles'].type = torch.tensor(particles.type).int()
        data['particles'].momentum = torch.tensor(particles.momentum).float()
        data['particles'].semantic_label = torch.tensor(particles.semantic_label).int()
        data['particles'].instance_label = torch.tensor(particles.instance_label).int()

        # 3D vertex truth
        if self.label_vertex:
            vtx_3d = [ [ event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z ] ]
            data['evt'].y_vtx = torch.tensor(vtx_3d).float()

        return evt.name, data