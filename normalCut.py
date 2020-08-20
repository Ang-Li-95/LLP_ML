
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def passCut(data):
  afterCut = data[(data['vtx_track_size']>=5) 
                & (data['vtx_dBV']>=0.01) 
                & (data['vtx_x']*data['vtx_x'] + data['vtx_y']*data['vtx_y']<4.3681)
                & (data['vtx_sigma_dBV']<0.0025)]
  return len(afterCut)


var = ['vtx_track_size', 'vtx_dBV', 'vtx_sigma_dBV', 'vtx_x', 'vtx_y', 'vtx_z']
signal_raw = pd.read_csv('2018MinoAOD/csvFiles/testggToNN_800M_1m.csv')[var]
print("after Cut: {0} vertices".format(passCut(signal_raw)))
bkg_name = ['2018MinoAOD/csvFiles/testQCD_HT700to1000.csv', 
            '2018MinoAOD/csvFiles/testQCD_HT1000to1500.csv',
            '2018MinoAOD/csvFiles/testQCD_HT1500to2000.csv', 
            '2018MinoAOD/csvFiles/testQCD_HT2000toInf.csv', 
            '2018MinoAOD/csvFiles/testTTJets_HT600To800.csv', 
            '2018MinoAOD/csvFiles/testTTJets_HT800To1200.csv', 
            '2018MinoAOD/csvFiles/testTTJets_HT1200To2500.csv',
            '2018MinoAOD/csvFiles/testTTJets_HT2500ToInf.csv']

#bkg = []
for b in bkg_name:
    #bkg.append(pd.read_csv(b)[var])
    print(b)
    print("after cut: {0} vertices".format(passCut(pd.read_csv(b)[var])))

