import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import numl, glob
import getopt

def main(argv):
  profiling  = False
  use_seq    = False
  inputfile  = ""
  outputfile = ""
  output_h5  = False
  try:
    opts, args = getopt.getopt(argv,"hps5i:o:",["ifile=","ofile="])
  except getopt.GetoptError:
    print("Usage: process.py [-p|-s|-5] -i <inputfile> -o <outputfile>")
    sys.exit(2)
  for opt, arg in opts:
    if opt == "-h":
      print("Usage: process.py [-p|-s|-5] -i <inputfile> -o <outputfile>")
      sys.exit()
    elif opt in ("-i", "--ifile"):
      inputfile = arg
    elif opt in ("-o", "--ofile"):
      outputfile = arg
    elif opt == "-5":
      output_h5 = True
    elif opt == "-p":
      profiling = True
    elif opt == "-s":
      use_seq = True

  if inputfile == "" or outputfile == "":
    print("Usage: process.py [-p|-s|-5] -i <inputfile> -o <outputfile>")
    sys.exit(2)

  if output_h5:
    # output HDF5 files, one graph per group, one file per MPI process
    out = numl.core.out.H5Out(outputfile)
  else:
    # output pytorch files, one graph per file
    out = numl.core.out.PTOut(outputfile)

  numl.process.hitgraph.process_file(out, inputfile, l=numl.labels.standard.panoptic_label, use_seq=use_seq, profile=profiling)

if __name__ == "__main__":
   main(sys.argv[1:])

