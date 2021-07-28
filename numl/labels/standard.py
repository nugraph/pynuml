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
    print('func:%r args:[%r, %r] took: %2.4f sec' % \
      (f.__name__, args, kw, te-ts))
    return result
  return wrap

@timing
def semantic_label(part):

  def walk(part, particles, depth, parent_label):

    import particle
    l = -1 if parent_label is None else parent_label # inherit from parent
    if l == -1:
      parent_type = 0 if part.parent_id == 0 else particles.type[part.parent_id]
      if abs(part.type) == 211:  l = label.pion.value
      if abs(part.type) == 13:   l = label.muon.value
      if abs(part.type) == 321:  l = label.kaon.value
      if abs(part.type) == 2212 or particle.pdgid.is_nucleus(part.type): l = label.hadron.value
      if part.type == 2112:
        l = label.diffuse.value
        parent_label = label.diffuse.value # propagate to children
      if part.type == 22:
        if part.end_process == b'conv':
          l = label.shower.value
          parent_label = label.shower.value # propagate to children
        if part.start_process == b'eBrem' or part.end_process == b'phot':
          l = label.diffuse.value
          parent_label = label.diffuse.value #propagate to children
      if abs(part.type) == 11:
        if abs(parent_type) == 13 and (part.start_process == b'muMinusCaptureAtRest' or part.start_process == b'muPlusCaptureAtRest'):
          l = label.michel.value
        if part.end_process == b'muIoni' or part.end_process == b'hIoni' or part.end_process == b'eIoni':
          l = label.delta.value
        if part.end_process == b'StepLimiter' or part.end_process == b'annihil' or part.end_process == b'eBrem':
          l = label.diffuse.value # is this right?
      if part.type == 111: l = label.invisible.value
    offset = ""
    for i in range(depth): offset += " "
    if l == -1:
      print(f"{offset}depth {depth}: particle {part.g4_id} with label {l}, type {part.type}, start process {part.start_process}, end process {part.end_process}, momentum {part.momentum}")
      raise Exception("unknown particle!")
    ret = [ { "g4_id": part.g4_id, "semantic_label": l } ]
    for _, row in particles[(part.g4_id==particles.parent_id)].iterrows():
      ret += walk(row, particles, depth+1, parent_label)
    return ret
    
  ret = []
  part = part.set_index("g4_id", drop=False)
  primaries = part[(part.parent_id==0)]
  for _, primary in primaries.iterrows():
    ret += walk(primary, part, 0, None)
  import pandas as pd
  return part.reset_index(drop=True).merge(pd.DataFrame.from_dict(ret), on="g4_id", how="left")

