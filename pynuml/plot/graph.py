import pandas as pd
from torch_geometric.data import Batch, HeteroData
import plotly.express as px
from plotly.graph_objects import FigureWidget

class GraphPlot:
    def __init__(self,
                 planes: list[str],
                 classes: list[str]):
        self._planes = planes
        self._classes = classes
        self._labels = pd.CategoricalDtype(classes, ordered=True)
        self._cmap = { c: px.colors.qualitative.Plotly[i] for i, c in enumerate(classes) }
        self._data = None
        self._df = None

    def to_dataframe(self, data: HeteroData):
        def to_categorical(arr):
            return pd.Categorical.from_codes(codes=arr, dtype=self._labels)
        if isinstance(data, Batch):
            raise Exception('to_dataframe does not support batches!')
        dfs = []
        for p in self._planes:
            plane = data[p].to_dict()
            df = pd.DataFrame(plane['id'], columns=['id'])
            df['plane'] = p
            df[['wire','time']] = plane['pos']
            df['y_filter'] = plane['y_semantic'] != -1
            mask = df.y_filter.values
            df['y_semantic'] = to_categorical(plane['y_semantic'])
            df['y_instance'] = plane['y_instance'].numpy()
            if 'x_semantic' in plane.keys():
                df['x_semantic'] = to_categorical(plane['x_semantic'].argmax(dim=-1).detach())
                df[self._classes] = plane['x_semantic'].detach()
            if 'x_filter' in plane.keys():
                df['x_filter'] = plane['x_filter'].detach()
            dfs.append(df)
        df = pd.concat(dfs)
        md = data['metadata']
        df['run'] = md.run.item()
        df['subrun'] = md.subrun.item()
        df['event'] = md.event.item()
        return df

    def plot(self,
             data: HeteroData,
             target: str = 'hits',
             how: str = 'none',
             filter: str = 'none',
             width: int = None,
             height: int = None) -> FigureWidget:

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
                    'labels': { 'y_semantic': 'Semantic label' },
                    'color': 'y_semantic',
                    'color_discrete_map': self._cmap
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted semantic labels',
                    'labels': { 'x_semantic': 'Semantic label' },
                    'color': 'x_semantic',
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
                    'labels': { 'y_instance': 'Instance label' },
                    'color': 'y_instance'
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted instance labels',
                    'labels': { 'x_instance': 'Instance label' },
                    'color': 'x_instance'
                }
            else:
                raise Exception('for instance labels, "how" must be one of "true" or "pred".')

        # filter labels
        elif target == 'filter':
            if how == 'true':
                opts = {
                    'title': 'True filter labels',
                    'labels': { 'y_filter': 'Filter label' },
                    'color': 'y_filter',
                    'color_continuous_scale': px.colors.sequential.Reds
                }
            elif how == 'pred':
                opts = {
                    'title': 'Predicted filter labels',
                    'labels': { 'x_filter': 'Filter label' },
                    'color': 'x_filter',
                    'color_continuous_scale': px.colors.sequential.Reds
                }
            else:
                raise Exception('for filter labels, "how" must be one of "true" or "pred".')

        else:
            raise Exception('"target" must be one of "hits", "semantic", "instance" or "filter".')

        if filter == 'none':
            df = self._df
        elif filter == 'true':
            df = self._df[self._df.y_filter.values]
            opts['title'] += ' (filtered by truth)'
        elif filter == 'pred':
            df = self._df[self._df.x_filter > 0.5]
            opts['title'] += ' (filtered by prediction)'
        else:
            raise Exception('"filter" must be one of "none", "true" or "pred.')

        fig = px.scatter(df, x='wire', y='time', facet_col='plane',
                         width=width, height=height, **opts)
        fig.update_xaxes(matches=None)
        for a in fig.layout.annotations:
            a.text = a.text.replace('plane=', '')
        return FigureWidget(fig)