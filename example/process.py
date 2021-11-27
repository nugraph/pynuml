import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, glob
import getopt

def main(argv):
  profiling     = False
  use_seq       = False
  inputfile     = ""
  outputfile    = ""
  output_h5     = False
  overwrite     = False
  evt_partition = 2
  usage_str     = "Usage: process.py [-p|-s|-f|-5|-d num] -i <inputfile> -o <outputfile>"
  try:
    opts, args = getopt.getopt(argv,"hpsf5d:i:o:",["ifile=","ofile="])
  except getopt.GetoptError:
    print(usage_str)
    sys.exit(2)
  for opt, arg in opts:
    if opt == "-h":
      print(usage_str)
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
    elif opt == "-d":   # event partition method.
      evt_partition = arg
                        # if arg == 0:   use event ID based
                        # elif arg == 1: use event amount based
                        # elif arg == 2: use event amount in paticle table

  if inputfile == "" or outputfile == "":
    print(usage_str)
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
                                     evt_part=evt_partition,
                                     profile=profiling)

if __name__ == "__main__":
   main(sys.argv[1:])

