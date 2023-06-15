import pandas as pd

class FlavorLabels:
    def __init__(self):
        self._labels = (
            'cc_nue',
            'cc_numu',
            'nc')

    @property
    def labels(self):
        return self._labels

    def label(self, idx: int):
        if not 0 <= label < len(self._labels):
            raise Exception(f'index {idx} out of range for {len(self._labels)} labels.')
        return self._labels[idx]

    def index(self, name: str):
        if name not in self._labels:
            raise Exception(f'"{name}" is not the name of a class.')
        return self._labels.index(name)

    @property
    def cc_nue(self):
        return self.index('cc_nue')
    
    @property
    def cc_numu(self):
        return self.index('cc_numu')

    @property
    def nc(self):
        return self.index('nc')

    def __call__(self, events: pd.DataFrame):
        event = events.squeeze()
        if not event.is_cc:
            return self.nc
        pdg = abs(event.nu_pdg)
        if pdg == 12:
            return self.cc_nue
        if pdg == 14:
            return self.cc_numu
        raise Exception(f'PDG code {event.nu_pdg} not recognised.')