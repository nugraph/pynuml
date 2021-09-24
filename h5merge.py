import sys, os
import getopt
import h5py

def merge(inputfile, outputfile):
  print("input file name =",inputfile)
  print("output file name =",outputfile)

  with open(inputfile) as f:
    in_files = f.read().splitlines()

  for fname in in_files:
    print("copy file:", fname)
    f = h5py.File(fname,'r')
    d_struct = {}
    d_struct[fname] = f.keys()
    for j in d_struct[fname]:
      # print("copy j:", j)
      os.system('h5copy -i %s -o %s -s %s -d %s' % (fname, outputfile, j, j))
    f.close()

if __name__ == "__main__":
  inputfile  = ""
  outputfile = ""

  try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
  except getopt.GetoptError:
    print("Usage: "+sys.argv[0]+" -i <inputfile> -o <outputfile>")
    sys.exit(2)
  for opt, arg in opts:
    if opt == "-h":
      print("Usage: "+sys.argv[0]+" -i <inputfile> -o <outputfile>")
      sys.exit()
    elif opt in ("-i", "--ifile"):
      inputfile = arg
    elif opt in ("-o", "--ofile"):
      outputfile = arg

  if inputfile == "" or outputfile == "":
    print("Usage: "+sys.argv[0]+" -i <inputfile> -o <outputfile>")
    sys.exit(2)

  merge(inputfile, outputfile)

