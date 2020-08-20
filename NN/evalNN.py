
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, '/uscms/home/ali/nobackup/LLP/DVAnalyzer/ML')
from utils.NNUtils import *

def NNscore(y_pred):
  label = ["signal","QCD700-1000", "QCD1000-1500", "QCD1500-2000", "QCD2000-Inf",
          "TT600-800", "TT800-1200", "TT1200-2500", "TT2500-Inf"]
  for i in range(len(y_pred)):
    print(returnYield(y_pred[i],0.9955))
    plt.hist(y_pred[i], bins=100, alpha=0.5, density = True, label=label[i])

  plt.legend()
  #plt.yscale('log')
  plt.savefig("NNscore.png")
  plt.close()
  return

def returnYield(y, cut):
  return len(y[y>=cut])


#scaler = load('scaler.bin')
var = ['vtx_track_size', 'vtx_dBV', 'vtx_sigma_dBV', 'vtx_x', 'vtx_y', 'vtx_z']
signal_raw, bkg = importTestData('2018MinoAOD/csvFiles/', var)
#signal_raw = pd.read_csv('2018MinoAOD/csvFiles/testggToNN_800M_1m.csv')[var]
#signal_raw = scaler.transform(signal_raw)
#s_label = np.ones(signal_raw.shape[0])
#bkg_name = ['2018MinoAOD/csvFiles/testQCD_HT700to1000.csv', 
#            '2018MinoAOD/csvFiles/testQCD_HT1000to1500.csv',
#            '2018MinoAOD/csvFiles/testQCD_HT1500to2000.csv', 
#            '2018MinoAOD/csvFiles/testQCD_HT2000toInf.csv', 
#            '2018MinoAOD/csvFiles/testTTJets_HT600To800.csv', 
#            '2018MinoAOD/csvFiles/testTTJets_HT800To1200.csv', 
#            '2018MinoAOD/csvFiles/testTTJets_HT1200To2500.csv',
#            '2018MinoAOD/csvFiles/testTTJets_HT2500ToInf.csv']
#bkg = []
#for b in bkg_name:
#    bkg_raw = pd.read_csv(b)[var]
#    bkg_raw = scaler.transform(bkg_raw)
#    bkg.append(bkg_raw)

y_pred = []
model =  make_model(input_dim=signal_raw.shape[-1])
model.load_weights('weights.h5')
y_pred.append(model.predict(signal_raw))

for b in bkg:
  y_pred.append(model.predict(b))
NNscore(y_pred)

