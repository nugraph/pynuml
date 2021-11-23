import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, glob
import getopt

def main(argv):
  profiling    = False
  use_seq      = False
  inputfile    = ""
  outputfile   = ""
  output_h5    = False
  overwrite    = False
  use_evt_num  = True
  try:
    opts, args = getopt.getopt(argv,"hpsdf5i:o:",["ifile=","ofile="])
  except getopt.GetoptError:
    print("Usage: process.py [-p|-s|-d|-f|-5] -i <inputfile> -o <outputfile>")
    sys.exit(2)
  for opt, arg in opts:
    if opt == "-h":
      print("Usage: process.py [-p|-s|-d|-f|-5] -i <inputfile> -o <outputfile>")
      sys.exit()
    elif opt in ("-i", "--ifile"):  # input file name
      inputfile = arg
    elif opt in ("-o", "--ofile"):  # output file prefix name if '-5' is used
      outputfile = arg              # output folder name for pytorch files
    elif opt == "-5":   # output file in HDF5 files, one per MPI process
      output_h5 = True
    elif opt == "-p":   # enable timing profiling and outputs
      profiling = True
    elif opt == "-s":   # use partitioning dataset evt.seq instead of evt.seq_cnt
      use_seq = True
    elif opt == "-f":   # overwrite the output file, if exists
      overwrite = True
    elif opt == "-d":   # use event ID based data partitioning strategy
      use_evt_num = False

  if inputfile == "" or outputfile == "":
    print("Usage: process.py [-p|-s|-d|-f|-5] -i <inputfile> -o <outputfile>")
    sys.exit(2)

  if output_h5:
    # output HDF5 files, one graph per group, one file per MPI process
    out = numl.core.out.H5Out(outputfile, overwrite)
  else:
    # output pytorch files, one graph per file
    out = numl.core.out.PTOut(outputfile)

  numl.process.hitgraph.process_file(out, inputfile,
                                     l=numl.labels.standard.panoptic_label,
                                     use_seq=use_seq,
                                     use_evt_num_based=use_evt_num,
                                     profile=profiling)

if __name__ == "__main__":
   main(sys.argv[1:])

