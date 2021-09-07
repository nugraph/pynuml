import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, glob

# output pytorch files, one graph per file
out = numl.core.out.PTOut("/files2/scratch/wkliao/Fermi/output")

# output HDF5 files, one graph per group
# TODO: use parallel HDF5
# out = numl.core.out.H5Out("/files2/scratch/wkliao/Fermi/output/x0123_out.h5")

numl.process.hitgraph.process_file(out, "/files2/scratch/wkliao/Fermi/x0123_seq_cnt.h5")
