import enum
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

from functools import wraps
from time import time

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
      import particle
      sl, slc = -1, None
      parent_type = 0 if part.parent_id == 0 else particles.type[part.parent_id]
      if abs(part.type) == 211: sl = label.pion.value
      if abs(part.type) == 13:  sl = label.muon.value
      if abs(part.type) == 321: sl = label.kaon.value
      if (particle.pdgid.is_baryon(part.type) and particle.pdgid.charge != 0) \
        or particle.pdgid.is_nucleus(part.type): sl = label.hadron.value
      if particle.pdgid.is_baryon(part.type) and particle.pdgid.charge(part.type) == 0:
        sl = label.diffuse.value
        slc = label.diffuse.value # propagate to children
      if part.type == 22:
        if part.end_process == b'conv':
          sl = label.shower.value
          slc = label.shower.value # propagate to children
        if part.start_process == b'eBrem' or part.end_process == b'phot' \
          or part.end_process == b'photonNuclear':
          sl = label.diffuse.value
          slc = label.diffuse.value #propagate to children
      if abs(part.type) == 11:
        if abs(parent_type) == 13 and (part.start_process == b'muMinusCaptureAtRest' \
          or part.start_process == b'muPlusCaptureAtRest' or part.start_process == b'Decay'):
          sl = label.michel.value
        if part.start_process == b'muIoni' or part.start_process == b'hIoni' \
          or part.start_process == b'eIoni':
          sl = label.delta.value
        if part.end_process == b'StepLimiter' or part.end_process == b'annihil' \
          or part.end_process == b'eBrem' or part.start_process == b'hBertiniCaptureAtRest' \
          or part.end_process == b'FastScintillation':
          sl = label.diffuse.value # is this right?
          slc = label.diffuse.value # propagate to children
      if part.type == 111 or abs(part.type) == 311 or abs(part.type) == 310 or abs(part.type) == 130: sl = label.invisible.value
      if sl == -1:
        raise Exception(f"particle not recognised! PDG code {part.type}, parent type {parent_type}, start process {part.start_process}, end process {part.end_process}")

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

    ret = [ { "g4_id": part.g4_id, "semantic_label": sl, "instance_label": il } ]
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

