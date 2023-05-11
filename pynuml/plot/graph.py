from typing import List

import pandas as pd
from torch_geometric.data import HeteroData
import plotly.express as px

class GraphPlot:
    def __init__(self,
                 planes: List[str],
                 classes: List[str]):
        self._planes = planes
        self._classes = classes
        self._labels = pd.CategoricalDtype(classes, ordered=True)
        self._cmap = { c: px.colors.qualitative.Plotly[i] for i, c in enumerate(self._classes) }
        self._data = None
        self._df = None

    def to_dataframe(self, data: HeteroData):
        def to_categorical(arr):
            return pd.Categorical.from_codes(codes=arr, dtype=self._labels)
        dfs = []
        for p in self._planes:
            plane = data[p].to_dict()
            df = pd.DataFrame(plane['id'], columns=['id'])
            df['plane'] = p
            df[['wire','time']] = plane['pos']
            df['y_f'] = plane['y_f']
            mask = df.y_f.values
            df.loc[mask, 'y_s'] = to_categorical(plane['y_s'])
            df.loc[mask, 'y_i'] = plane['y_i'].numpy()
            if 'x_s' in plane.keys():
                df['x_s'] = to_categorical(plane['x_s'].argmax(dim=-1).detach())
                df[self._classes] = plane['x_s'].detach()
            if 'x_f' in plane.keys():
                df['x_f'] = plane['x_f'].detach()
            dfs.append(df)
        return pd.concat(dfs)

    def plot(self,
             data: HeteroData,
             name: str,
             target: str = 'hits',
             how: str = 'none',
             filter: str = 'none',
             write_png: bool = True,
             write_html: bool = False):

        if data is not self._data:
            self._data = data
            self._df = self.to_dataframe(data)

        # no colour
        if target == 'hits':
            opts = {
                'title': 'Graph hits'
            }

        # semantic labels
        elif target == 'semantic':
            if how == 'true':
                opts = {
                    'title': 'True semantic labels',
                    'labels': { 'y_s': 'Semantic label' },
                    'color': 'y_s',
                    'color_discrete_map': self._cmap
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted semantic labels',
                    'labels': { 'x_s': 'Semantic label' },
                    'color': 'x_s',
                    'color_discrete_map': self._cmap
                }
            elif how in self._classes:
                opts = {
                    'title': f'Predicted semantic label strength for {how} class',
                    'labels': { how: f'{how} probability' },
                    'color': how,
                    'color_continuous_scale': px.colors.sequential.Reds
                }
            else:
                raise Exception('for semantic labels, "how" must be one of "true", "pred" or the name of a class.')

        # instance labels
        elif target == 'instance':
            if how == 'true':
                opts = {
                    'title': 'True instance labels',
                    'labels': { 'y_i': 'Instance label' },
                    'color': 'y_i'
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted instance labels',
                    'labels': { 'x_i': 'Instance label' },
                    'color': 'x_i'
                }
            else:
                raise Exception('for instance labels, "how" must be one of "true" or "pred".')

        # filter labels
        elif target == 'filter':
            if how == 'true':
                opts = {
                    'title': 'True filter labels',
                    'labels': { 'y_f': 'Filter label' },
                    'color': 'y_f',
                    'color_continuous_scale': px.colors.sequential.Reds
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted filter labels',
                    'labels': { 'x_f': 'Filter label' },
                    'color': 'x_f',
                    'color_continuous_scale': px.colors.sequential.Reds
                }
            else:
                raise Exception('for filter labels, "how" must be one of "true" or "pred".')

        else:
            raise Exception('"target" must be one of "hits", "semantic", "instance" or "filter".')

        if filter == 'none':
            df = self._df
        elif filter == 'true':
            df = self._df[self._df.y_f.values]
            opts['title'] += ' (filtered by truth)'
        elif filter == 'pred':
            df = self._df[self._df.x_f > 0.5]
            opts['title'] += ' (filtered by prediction)'
        else:
            raise Exception('"filter" must be one of "none", "true" or "pred.')

        fig = px.scatter(df, x='wire', y='time', facet_col='plane', **opts)
        fig.update_yaxes(matches=None)
        fig.update_xaxes(matches=None)
        for a in fig.layout.annotations:
            a.text = a.text.replace('plane=', '')
        if write_html:
            fig.write_html(f'{name}.html')
        if write_png:
            fig.write_image(f'{name}.png')