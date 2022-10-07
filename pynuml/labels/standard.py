import enum
from typing import NoReturn
import pandas as pd

class label(enum.Enum):
    pion = 0
    muon = 1
    kaon = 2
    hadron = 3
    shower = 4
    michel = 5
    delta = 6
    diffuse = 7
    invisible = 8

def standard(part: pd.DataFrame,
             th_gamma: float = 0.02,
             th_hadr: float = 0.2):
    '''Standard labelling function.

    Pion, Muon, Kaon, Hadron, EM shower, Michel electron, Delta ray,
    diffuse activity.
    '''

    def walk(part, particles, depth, sl, il):
        def s(part, particles):
            import particle  # does this need to be in the closure?
            sl, slc = -1, None
            parent_g4_pdg = 0 if part.parent_id == 0 else particles.g4_pdg[part.parent_id]

            def pion_labeler(part, parent_g4_pdg):
                sl = label.pion.value
                slc = None
                return sl, slc

            def muon_labeler(part, parent_g4_pdg):
                sl = label.muon.value
                slc = None
                return sl, slc

            def kaon_labeler(part, parent_g4_pdg):
                sl = label.kaon.value
                slc = None
                return sl, slc

            def neutral_pions_kaons_labeler(part, parent_g4_pdg):
                sl = label.invisible.value
                slc = None
                return sl, slc

            def electron_positron_labeler(part, parent_g4_pdg):
                if part.start_process == b'primary':
                    sl = label.shower.value
                    slc = label.shower.value
                elif abs(parent_g4_pdg) == 13 and (part.start_process == b'muMinusCaptureAtRest' \
                    or part.start_process == b'muPlusCaptureAtRest' or part.start_process == b'Decay'):
                    sl = label.michel.value
                    slc = label.michel.value

                elif part.start_process == b'conv' or part.end_process == b'conv' \
                    or part.start_process == b'compt' or part.end_process == b'compt':
                    if part.momentum >=th_gamma:
                        sl = label.shower.value
                        slc = label.shower.value
                    else:
                        sl = label.diffuse.value
                        slc = label.diffuse.value
                elif part.start_process == b'eBrem' or part.end_process == b'phot' \
                    or part.end_process == b'photonNuclear':
                    sl = label.diffuse.value
                    slc = None
                elif part.start_process == b'muIoni' or part.start_process == b'hIoni' \
                    or part.start_process == b'eIoni':
                    if part.momentum <= 0.01:
                        if part.start_process == b'muIoni':
                            sl = label.muon.value
                            slc = None
                        elif part.start_process == b'hIoni':
                            if abs(parent_g4_pdg) == 2212:
                                sl = label.hadron.value
                                if part.momentum <= 0.0015: sl = label.diffuse.value
                            else:
                                sl = label.pion.value
                            slc = None
                        else:
                            sl = label.diffuse.value
                            slc = None
                    else:
                        sl = label.delta.value
                        slc = label.delta.value
                elif part.end_process == b'StepLimiter' or part.end_process == b'annihil' \
                    or part.end_process == b'eBrem' or part.start_process == b'hBertiniCaptureAtRest' \
                    or part.end_process == b'FastScintillation':
                    sl = label.diffuse.value
                    slc = label.diffuse.value
                else:
                    raise Exception('electron failed to be labeled as expected')

                return sl, slc

            def gamma_labeler(part, parent_g4_pdg):
                if part.start_process == b'conv' or part.end_process == b'conv' \
                    or part.start_process == b'compt' or part.end_process == b'compt':
                    if part.momentum >=th_gamma:
                        sl = label.shower.value
                        slc = label.shower.value
                    else:
                        sl = label.diffuse.value
                        slc = label.diffuse.value
                elif part.start_process == b'eBrem' or part.end_process == b'phot' \
                    or part.end_process == b'photonNuclear':
                    sl = label.diffuse.value
                    slc = None
                else:
                    raise Exception('gamma interaction failed to be labeled as expected')
                return sl, slc

            def unlabeled_particle(part, parent_g4_pdg):
                raise Exception(f"particle not recognised! PDG code {part.g4_pdg}, parent PDG code {parent_g4_pdg}, start process {part.start_process}, end process {part.end_process}")

            particle_processor = {
                211: pion_labeler,
                13: muon_labeler,
                321: kaon_labeler,
                111: neutral_pions_kaons_labeler,
                311: neutral_pions_kaons_labeler,
                310: neutral_pions_kaons_labeler,
                130: neutral_pions_kaons_labeler,
                11: electron_positron_labeler,
                22: gamma_labeler
            }

            if particle.pdgid.charge(part.g4_pdg) == 0 and part.end_process == b'CoupledTransportation':
                # neutral particle left the volume boundary
                sl = label.invisible.value
            else:
                func = particle_processor.get(abs(part.g4_pdg), lambda x ,y: (-1, None))
                sl, slc = func(part, parent_g4_pdg)

            # baryon interactions - hadron or diffuse
            if (particle.pdgid.is_baryon(part.g4_pdg) and particle.pdgid.charge(part.g4_pdg) == 0) \
                or particle.pdgid.is_nucleus(part.g4_pdg):
                sl = label.diffuse.value
            if particle.pdgid.is_baryon(part.g4_pdg) and particle.pdgid.charge(part.g4_pdg) != 0:
                if abs(part.g4_pdg) == 2212 and part.momentum >= th_hadr:
                    sl = label.hadron.value
                else:
                    sl = label.diffuse.value

            # check to make sure particle was assigned
            if sl == -1:
                unlabeled_particle(part, parent_g4_pdg)

            return sl, slc

        def i(part, particles, sl):
            il, ilc = -1, None
            if sl == label.muon.value and part.start_process == b'muIoni':
                il = part.parent_id
            elif (sl == label.pion.value or sl == label.hadron.value) and part.start_process == b'hIoni':
                il = part.parent_id
            elif sl != label.diffuse.value and sl != label.delta.value and sl != label.invisible.value:
                il = part.g4_id
                if sl == label.shower.value: ilc = il
                if sl == label.michel.value: ilc = il
            return il, ilc

        if sl is not None: slc = sl
        else: sl, slc = s(part, particles)

        if il is not None: ilc = il
        else: il, ilc = i(part, particles, sl)

        ret = [ { "g4_id": part.g4_id, "parent_id": part.parent_id, "g4_pdg": part.g4_pdg, "start_process": part.start_process, "end_process": part.end_process, "momentum": part.momentum, "semantic_label": sl, "instance_label": il } ]
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
