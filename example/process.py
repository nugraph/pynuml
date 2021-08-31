import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, glob

out = numl.core.out.PTOut("/data/pandora/processed")

for fname in glob.glob("/data/pandora/hdf5/*"):
  numl.process.hitgraph.process_file(out, fname)

