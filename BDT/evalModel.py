import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/uscms/home/ali/nobackup/LLP/DVAnalyzer/ML')
from utils.NNUtils import *

def BDTscore(y_pred):
  label = ["signal","QCD700-1000", "QCD1000-1500", "QCD1500-2000", "QCD2000-Inf",
          "TT600-800", "TT800-1200", "TT1200-2500", "TT2500-Inf"]
  for i in range(len(y_pred)):
    print(returnYield(y_pred[i],0.6446))
    plt.hist(y_pred[i], bins=100, range=(0,1), alpha=0.5, density = True, label=label[i])

  plt.legend()
  plt.savefig("score.png")
  plt.close()
  return

def returnYield(y, cut):
  return len(y[y>=cut])



var = ['vtx_track_size', 'vtx_dBV', 'vtx_sigma_dBV', 'vtx_x', 'vtx_y', 'vtx_z']
#signal_raw = pd.read_csv('2018MinoAOD/csvFiles/testggToNN_800M_1m.csv')[var]
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
#    bkg.append(pd.read_csv(b)[var])

signal_raw, bkg = importTestData('2018MinoAOD/csvFiles/', var, False)

y_pred = []
model = xgb.XGBClassifier()
model.load_model('BDT_bestmodel.json')
y_pred.append(model.predict_proba(signal_raw)[:,1])

for b in bkg:
  y_pred.append(model.predict_proba(b)[:,1])
BDTscore(y_pred)

