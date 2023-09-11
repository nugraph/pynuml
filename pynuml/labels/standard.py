import pandas as pd
import particle

class StandardLabels:

    def __init__(self,
                 gamma_threshold: float = 0.02,
                 hadron_threshold: float = 0.2):
        self._labels = [
            'pion',
            'muon',
            'kaon',
            'hadron',
            'shower',
            'michel',
            'delta',
            'diffuse',
            'invisible'
        ]
        self._gamma_threshold = gamma_threshold
        self._hadron_threshold = hadron_threshold

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
    def pion(self):
        return self.index('pion')

    @property
    def muon(self):
        return self.index('muon')

    @property
    def kaon(self):
        return self.index('kaon')

    @property
    def hadron(self):
        return self.index('hadron')

    @property
    def shower(self):
        return self.index('shower')

    @property
    def michel(self):
        return self.index('michel')

    @property
    def delta(self):
        return self.index('delta')

    @property
    def diffuse(self):
        return self.index('diffuse')
    
    @property
    def invisible(self):
        return self.index('invisible')

    def __call__(self,
                 part: pd.DataFrame):
        '''Standard labelling function.

        Pion, Muon, Kaon, Hadron, EM shower, Michel electron, Delta ray,
        diffuse activity.
        '''

        def walk(part, particles, depth, sl, il):
            def s(part, particles):
                sl, slc = -1, None # default semantic labels
                il, ilc = -1, None # default instance labels
                parent_type = 0 if part.parent_id == 0 else particles.type[part.parent_id]

                # charged pion labeller
                def pion_labeler(part, parent_type):
                    sl = self.pion
                    il = part.g4_id
                    # do not propagate labels to children
                    return sl, il, None, None

                # muon labeller
                def muon_labeler(part, parent_type):
                    sl = self.muon
                    il = part.g4_id
                    # do not propagate labels to children
                    return sl, il, None, None

                # tau labeller
                def tau_labeler(part, parent_type):
                    sl = self.tau
                    il = part.g4_id
                    # do not propagate labels to children
                    return sl, il, None, None

                # charged kaon labeller
                def kaon_labeler(part, parent_type):
                    sl = self.kaon
                    il = part.g4_id
                    # do not propagate labels to children
                    return sl, il, None, None

                # neutral pion + kaon labeller
                def neutral_pions_kaons_labeler(part, parent_type):
                    sl = self.invisible
                    # form no instance, and do not propagate labels to children
                    return sl, -1, None, None

                # electron + positron labeller
                def electron_positron_labeler(part, parent_type):

                    # label primary electrons as showers
                    if part.start_process == 'primary':
                        sl = self.shower
                        il = part.g4_id
                        # propagate labels to children
                        slc = self.shower
                        ilc = part.g4_id

                    # label michels
                    elif abs(parent_type) == 13 and part.start_process in ('muMinusCaptureAtRest','muPlusCaptureAtRest','Decay'):
                        sl = self.michel
                        il = part.g4_id
                        # do not propagate labels to children
                        slc = None
                        ilc = None

                    # conversions and Compton scatters can form showers...
                    elif part.start_process == 'conv' or part.end_process == 'conv' \
                        or part.start_process == 'compt' or part.end_process == 'compt':

                        # ...if they have enough momentum...
                        if part.momentum >= self._gamma_threshold:
                            sl = self.shower
                            il = part.g4_id
                            # propagate labels to children
                            slc = self.shower
                            ilc = part.g4_id
                        # ...otherwise we call them diffuse
                        else:
                            sl = self.diffuse
                            il = -1
                            # propagate labels to children
                            slc = self.diffuse
                            ilc = -1

                    # these processes tend to look diffuse
                    elif part.start_process == 'eBrem' or part.end_process == 'phot' \
                        or part.end_process == 'photonNuclear' or part.end_process == 'eIoni':
                        sl = self.diffuse
                        # form no instance, and do not propagate labels to children
                        il = -1
                        slc = None
                        ilc = None

                    # ionisation electrons are typically delta rays...
                    elif part.start_process == 'muIoni' or part.start_process == 'hIoni' \
                        or part.start_process == 'eIoni':

                        # ...but no point labelling them as such if they're too low-momentum to be resolved
                        if part.momentum <= 0.01:

                            # delta rays from muons
                            if part.start_process == 'muIoni':
                                sl = self.muon
                                il = part.parent_id
                                # do not propagate labels to children
                                slc = None
                                ilc = None

                            # delta rays from hadrons
                            elif part.start_process == 'hIoni':
                                if abs(parent_type) == 2212:
                                    sl = self.hadron
                                    il = part.parent_id
                                    # do not propagate labels to children
                                    slc = None
                                    ilc = None
                                    if part.momentum <= 0.0015:
                                        sl = self.diffuse
                                        # form no instance, and do not propagate labels to children
                                        il = -1
                                        slc = None
                                        ilc = None

                                # delta rays from pions
                                else:
                                    sl = self.pion
                                    il = part.parent_id
                                    # do not propagate labels to children
                                    slc = None
                                    ilc = None

                            # everything else is just called diffuse
                            else:
                                sl = self.diffuse
                                # form no instance, and do not propagate labels to children
                                il = -1
                                slc = None
                                ilc = None

                        # if higher momentum, explicitly label as delta
                        else:
                            sl = self.delta
                            il = part.g4_id
                            slc = self.delta
                            ilc = part.g4_id

                    # diffuse EM processes
                    elif part.end_process == 'StepLimiter' or part.end_process == 'annihil' \
                        or part.end_process == 'eBrem' or part.start_process == 'hBertiniCaptureAtRest' \
                        or part.end_process == 'FastScintillation':
                        sl = self.diffuse
                        slc = self.diffuse
                        # do not form instances for diffuse depositions
                        il = 1
                        ilc = -1
                    else:
                        raise Exception(f'labelling failed for electron with start process "{part.start_process}" and end process "{part.end_process}')

                    return sl, il, slc, ilc

                def gamma_labeler(part, parent_type):
                    if part.start_process == 'conv' or part.end_process == 'conv' \
                        or part.start_process == 'compt' or part.end_process == 'compt':
                        if part.momentum >= self._gamma_threshold:
                            sl = self.shower
                            il = part.g4_id
                            slc = self.shower
                            ilc = part.g4_id
                        else:
                            sl = self.diffuse
                            slc = self.diffuse
                            # do not form instances for diffuse depositions
                            il = -1
                            ilc = -1
                    elif part.start_process == 'eBrem' or part.end_process == 'phot' \
                        or part.end_process == 'photonNuclear':
                        sl = self.diffuse
                        slc = None
                        il = -1
                        ilc = None
                    else:
                        raise Exception(f'labelling failed for photon with start process "{part.start_process}" and end process "{part.end_process}')
                    return sl, il, slc, ilc

                def unlabeled_particle(part, parent_type):
                    raise Exception(f"particle not recognised! PDG code {part.type}, parent PDG code {parent_type}, start process {part.start_process}, end process {part.end_process}")

                particle_processor = {
                    211: pion_labeler,
                    221: pion_labeler,
                    331: pion_labeler,
                    223: pion_labeler,
                    13: muon_labeler,
                    321: kaon_labeler,
                    111: neutral_pions_kaons_labeler,
                    311: neutral_pions_kaons_labeler,
                    310: neutral_pions_kaons_labeler,
                    130: neutral_pions_kaons_labeler,
                    113: neutral_pions_kaons_labeler,
                    11: electron_positron_labeler,
                    22: gamma_labeler
                }

                if particle.pdgid.charge(part.type) == 0 and part.end_process == 'CoupledTransportation':
                    # neutral particle left the volume boundary
                    sl = self.invisible
                elif abs(part.type) in particle_processor.keys():
                    func = particle_processor.get(abs(part.type), lambda x, y: (-1, None))
                    ret = func(part, parent_type)
                    sl, il, slc, ilc = ret

                # baryon interactions - hadron or diffuse
                if (particle.pdgid.is_baryon(part.type) and particle.pdgid.charge(part.type) == 0) \
                    or particle.pdgid.is_nucleus(part.type):
                    sl = self.diffuse
                if particle.pdgid.is_baryon(part.type) and particle.pdgid.charge(part.type) != 0:
                    if abs(part.type) == 2212 and part.momentum >= self._hadron_threshold:
                        sl = self.hadron
                        il = part.g4_id
                    else:
                        sl = self.diffuse

                # call a charged tau highly ionising - should revisit this
                if abs(part.type) == 15:
                    sl = self.hadron

                # check to make sure particle was assigned
                if sl == -1:
                    unlabeled_particle(part, parent_type)

                return sl, il, slc, ilc

            if sl is not None:
                slc = sl
                ilc = il
            else:
                sl, il, slc, ilc = s(part, particles)

            ret = [ {
                "g4_id": part.g4_id,
                "parent_id": part.parent_id,
                "type": part.type,
                "start_process": part.start_process,
                "end_process": part.end_process,
                "momentum": part.momentum,
                "semantic_label": sl,
                "instance_label": il } ]
            for _, row in particles[(part.g4_id==particles.parent_id)].iterrows():
                ret += walk(row, particles, depth+1, slc, ilc)
            return ret

        ret = []
        part = part.set_index("g4_id", drop=False)
        primaries = part[(part.parent_id==0)]
        for _, primary in primaries.iterrows():
            ret += walk(primary, part, 0, None, None)
        if len(ret)==0: return
        labels = pd.DataFrame.from_dict(ret)
        instances = { val: i for i, val in enumerate(labels[(labels.instance_label>=0)].instance_label.unique()) }

        def alias_instance(row, instances):
            if row.instance_label == -1: return -1
            return instances[row.instance_label]

        labels["instance_label"] = labels.apply(alias_instance, args=[instances], axis="columns")

        return labels

    def validate(self, labels: pd.Series):
        mask = (labels < 0) | (labels >= len(self._labels) - 1)
        if mask.any():
            raise Exception(f'{mask.sum()} semantic labels are out of range: {labels[mask]}.')