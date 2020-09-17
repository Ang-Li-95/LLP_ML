
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


scaler = load('scaler.bin')
var = ['vtx_track_size', 'vtx_dBV', 'vtx_sigma_dBV', 'vtx_x', 'vtx_y', 'vtx_z', 'evt']
var_NN = ['vtx_track_size', 'vtx_dBV', 'vtx_sigma_dBV', 'vtx_x', 'vtx_y', 'vtx_z']
label = ["ggToNN","QCD700-1000", "QCD1000-1500", "QCD1500-2000", "QCD2000-Inf",
        "TT600-800", "TT800-1200", "TT1200-2500", "TT2500-Inf"]
signal_raw, bkg = importData('2018MinoAOD/csvFiles/withEvt/', var, False)

y_pred = []
model =  make_model(input_dim=len(var_NN))
model.load_weights('weights.h5')
s_NN = scaler.transform(signal_raw[var_NN])
print(s_NN.shape)
signal_raw["NNScore"] = model.predict(s_NN)
signal_raw.to_csv(label[0]+"_withNNScore.csv")

for i in range(len(bkg)):
  b = bkg[i]
  b_NN = scaler.transform(b[var_NN])
  print(b_NN.shape)
  b["NNScore"] = model.predict(b_NN)
  b.to_csv(label[i+1]+"_withNNScore.csv")


