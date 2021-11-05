import enum
from functools import wraps
from time import time
from typing import NoReturn

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

def timing(f):
  @wraps(f)
  def wrap(*args, **kw):
    ts = time()
    result = f(*args, **kw)
    te = time()
    print('func:%r took: %2.4f sec' % \
    (f.__name__, te-ts))
    return result
  return wrap

@timing
def panoptic_label(part):
  def walk(part, particles, depth, sl, il):
    def s(part, particles):
      import particle  # does this need to be in the closure?
      sl, slc = -1, None
      parent_type = 0 if part.parent_id == 0 else particles.type[part.parent_id]

      def pion_labeler(part, parent_type):
        sl = label.pion.value
        slc = None
        return sl, slc

      def muon_labeler(part, parent_type):
        sl = label.muon.value
        slc = None
        return sl, slc

      def kaon_labeler(part, parent_type):
        sl = label.kaon.value
        slc = None
        return sl, slc

      def neutral_pions_kaons_labeler(part, parent_type):
        sl = label.invisible.value
        slc = None
        return sl, slc

      def electron_positron_labeler(part, parent_type):
        if part.start_process == b'primary':
          sl = label.shower.value
          slc = label.shower.value
        elif abs(parent_type) == 13 and (part.start_process == b'muMinusCaptureAtRest' \
          or part.start_process == b'muPlusCaptureAtRest' or part.start_process == b'Decay'):
          sl = label.michel.value
          slc = None
        elif part.start_process == b'muIoni' or part.start_process == b'hIoni' \
          or part.start_process == b'eIoni':
          if part.momentum <= 0.01:
            if part.start_process == b'muIoni':
              sl = label.muon.value
              slc = None
            elif part.start_process == b'hIoni':
              if abs(parent_type) == 2212:
                sl = label.hadron.value
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

      def gamma_labeler(part, parent_type):
        if part.end_process == b'conv':
          if part.momentum >=0.02:
            sl = label.shower.value
          else:
            sl = label.diffuse.value
        elif part.start_process == b'compt':
          if part.momentum >= 0.02:
            sl = label.shower.value
          else:
            sl = label.diffuse.value
        elif part.start_process == b'eBrem' or part.end_process == b'phot' \
          or part.end_process == b'photonNuclear':
          sl = label.diffuse.value
        else:
          raise Exception('gamma interaction failed to be labeled as expected')
        slc = sl
        return sl, slc

      def unlabeled_particle(part, parent_type):
        raise Exception(f"particle not recognised! PDG code {part.type}, parent type {parent_type}, start process {part.start_process}, end process {part.end_process}")

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

      if part.end_process == b'CoupledTransportation':
        # particle left the volume boundary
        sl = label.invisible.value
      else:
        func = particle_processor.get(abs(part.type), lambda x ,y: (-1, None))
        sl, slc = func(part, parent_type)

      # baryon interactions - hadron or diffuse
      if (particle.pdgid.is_baryon(part.type) and particle.pdgid.charge(part.type) != 0) \
        or particle.pdgid.is_nucleus(part.type):
        sl = label.hadron.value
      if particle.pdgid.is_baryon(part.type) and particle.pdgid.charge(part.type) == 0:
        if abs(part.type) == 2212 and part.momentum >=0.2:
          sl = label.hadron.value
        else:
          sl = label.diffuse.value

      # check to make sure particle was assigned
      if sl == -1:
        unlabeled_particle(part, parent_type)

      return sl, slc

    def i(part, particles, sl):
      il, ilc = -1, -1
      if sl != label.diffuse.value and sl != label.delta.value:
        il = part.g4_id
        if sl == label.shower.value: ilc = il
      return il, ilc

    if sl is not None: slc = sl
    else: sl, slc = s(part, particles)

    if il is not None: ilc = il
    else: il, ilc = i(part, particles, sl)

    ret = [ { "g4_id": part.g4_id, "parent_id": part.parent_id, "type": part.type, "start_process": part.start_process, "end_process": part.end_process, "momentum": part.momentum,                             "semantic_label": sl, "instance_label": il } ]
    for _, row in particles[(part.g4_id==particles.parent_id)].iterrows():
      ret += walk(row, particles, depth+1, slc, ilc)
    return ret

  ret = []
  part = part.set_index("g4_id", drop=False)
  primaries = part[(part.parent_id==0)]
  for _, primary in primaries.iterrows():
    ret += walk(primary, part, 0, None, None)
  import pandas as pd
  labels = pd.DataFrame.from_dict(ret)
  instances = { val: i for i, val in enumerate(labels[(labels.instance_label>=0)].instance_label.unique()) }

  def alias_instance(row, instances):
    if row.instance_label == -1: return -1
    return instances[row.instance_label]

  labels["instance_label"] = labels.apply(alias_instance, args=[instances], axis="columns")
  return labels

def semantic_label(part):
  return panoptic_label(part).drop("instance_label", axis="columns")

def instance_label(part):
  return panoptic_label(part).drop("semantic_label", axis="columns")
