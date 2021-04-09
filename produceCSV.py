import uproot
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input')
parser.add_argument('--output', dest='output')
#parser.add_argument('--isSignal', action='store_true')
args = parser.parse_args()

f = uproot.open(args.input)
tree = f["DVAnalyzer/tree_DV"]
var = ['vtx_track_size','vtx_dBV','vtx_sigma_dBV','vtx_x','vtx_y','vtx_z','evt']
#var = ['vtx_dBV','vtx_sigma_dBV' ]
variable_total = np.array([])
for v in var:
    variable = np.array([])
    values = tree.array(v)
    i = 0
    for value in values:
    #for i in range(0,len(values)):
        #print(value)
        variable = np.hstack((variable, np.array(value)))
        i+= 1
        #if(i>5):
        #    break
    variable = np.reshape(variable, (len(variable), 1))

    if(len(variable_total)==0):
        variable_total = variable
    else:
        variable_total = np.hstack((variable_total, variable))
    #print(variable_total)
with open(args.output,'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(var)
    for iv in range(len(variable_total)):
        writer.writerow(variable_total[iv])
