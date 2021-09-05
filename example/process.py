import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, glob

out = numl.core.out.PTOut("/files2/scratch/wkliao/Fermi/output")
# numl.process.hitgraph.process_file(out, "/files2/scratch/wkliao/Fermi/x1_key_off_len.h5")
numl.process.hitgraph.process_file(out, "/files2/scratch/wkliao/Fermi/x0123_key_off_len.h5")

