"""
PETRA HD5 to SDDS converter
"""

import h5py
import sys

import time
import argparse
import re
from datetime import datetime

USETURNS = 50000

BAD_BPMS = []

#BAD_BPMS = ["BPM_SWR_90", "BPM_WR_82", "BPM_WR126", "BPM_WL68", "BPM_WL_18", "BPM_NR_0", "BPM_NR_79", "BPM_NOR_23", "BPM_NOR_58", "BPM_NOR_85", "BPM_NOR_86", "BPM_OL_83", "BPM_NOR_104", "BPM_OL_118", "BPM_OL_65", "BPM_OL_58", "BPM_SOR_13", "BPM_WL_68", "BPM_WR_126", "BPM_NWR_31"]
#BAD_BPMS = ["BPM_SOR_133",
#            "BPM_SL_126",
#            "BPM_SL_140",
#            "BPM_SL_36",
#            "BPM_SWL_31",
#            "BPM_SWL_13",
#            "BPM_SWR_90",
#            "BPM_WR_82",
#            "BPM_WR126",
#            "BPM_WL68",
#            "BPM_WL_18",
#            "BPM_NR_0",
#            "BPM_NR_79",
#            "BPM_NOR_23",
#            "BPM_NOR_58",
#            "BPM_NOR_85",
#            "BPM_NOR_86",
#            "BPM_OL_83",
#            "BPM_NOR_104",
#            "BPM_OL_118",
#            "BPM_OL_65",
#            "BPM_OL_58",
#            "BPM_SOR_13",
#            "BPM_WL_68",
#            "BPM_WR_126",
#            "BPM_NWR_31"]

parser = argparse.ArgumentParser(description='Converting hd5 files to ASCII sdds')
parser.add_argument('-i', dest="input", default="/media/awegsche/HDD/files/learning/37_PETRA/HD5_data/05_h_gain_v2.h5")
parser.add_argument('-o', dest="output", default=None)
parser.add_argument('-x0', dest="x0", default=0)
parser.add_argument('-x1', dest="x1", default=1e12)

args = parser.parse_args()

fext = re.compile("\\.\\w+")

f = h5py.File(args.input, "r")
if args.output is None:
    args.output = fext.sub(".sdds", args.input)

i = 0
for _name, bunch in f.iteritems():
    if _name == "timestamp":
        continue
    
    outfile = open(args.output + "_" + _name + ".sdds", "w")
    outfile.write("""# title
# hd5 converted
# author: Andreas Wegscheider
# date: {:%d.%m.%y}
# removed {:d} bad BPMs: """.format(datetime.now(), len(BAD_BPMS)))
    for bbpm in BAD_BPMS:
        outfile.write(bbpm + " ")
    outfile.write("\n")

    lastp = 0

    for (name, block) in bunch.iteritems():
        if name not in twiss.NAME or name in BAD_BPMS:
            continue
        i = i + 1
        perc = float(i)/len(bunch)
        if perc - lastp > 10:
            lastp = perc
            print("{:.2f}".format(perc))
#        progressLumi(perc)
        index = twiss.indx[name]
        s_ = twiss.S[index]
        
        if "xcor" in block:
            X_KEY = "xcor"
        else:
            X_KEY = "x"
        
        if "ycor" in block:
            Y_KEY = "ycor"
        else:
            Y_KEY = "y"
        
        usable_turns =  min(len(block.get(X_KEY)), len(block.get(Y_KEY)))
        
        uset = min(usable_turns, USETURNS)
        begin = args.x0
        end = min(args.x1, uset)
        bl_x = block.get(X_KEY)[begin:end]
        bl_y = block.get(Y_KEY)[begin:end]
    
    
        stime = time.time()
    
        outfile.write("0 {0:s} {1:f} ".format(name, s_))
        outfile.write(" ".join(map(str, bl_x)))
        outfile.write("\n")
        outfile.write("1 {0:s} {1:f} ".format(name, s_))
        outfile.write(" ".join(map(str, bl_y)))
        outfile.write("\n")
    
        etime = time.time()
    outfile.close()
    

f.close()

print("--- Done")

