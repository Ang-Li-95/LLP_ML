import pandas as pd
import numpy as np
from joblib import dump, load
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras



def importTrainData(path, var, testDataSize, scale=True):
  '''
  import csv data for training
  path: file paths
  var: variables to fed into training
  testDataSize: number of samples from each data to evaluate the model
  scale: whether to scale the data
  '''
  if(path[-1]!='/'):
    path += '/'

  signal = pd.read_csv(path + 'trainggToNN_800M_1m.csv')[var]
  signal = signal.dropna()
  signal = shuffle(signal)
  signal_raw = signal[:-testDataSize]
  s_label = np.ones(signal_raw.shape[0])
  signal_raw_test = signal[-testDataSize:]
  s_label_test = np.ones(signal_raw_test.shape[0])
  bkg_name = [path + 'trainQCD_HT700to1000.csv', 
              path + 'trainQCD_HT1000to1500.csv',
              path + 'trainQCD_HT1500to2000.csv', 
              path + 'trainQCD_HT2000toInf.csv', 
              path + 'trainTTJets_HT600To800.csv', 
              path + 'trainTTJets_HT800To1200.csv', 
              path + 'trainTTJets_HT1200To2500.csv',
              path + 'trainTTJets_HT2500ToInf.csv']
  for b in bkg_name:
      bkg = pd.read_csv(b)[var]
      bkg = bkg.dropna()
      bkg = shuffle(bkg)
      try:
          bkg_raw
      except NameError:
        bkg_raw = bkg[:-testDataSize]
      else:
        bkg_raw = bkg_raw.append(bkg[:-testDataSize])
      try:
          bkg_raw_test
      except NameError:
        bkg_raw_test = bkg[-testDataSize:]
      else:
        bkg_raw_test = bkg_raw_test.append(bkg[-testDataSize:])
          
  #pos_weight = bkg_raw.shape[0]/signal_raw.shape[0]
  b_label = np.zeros(bkg_raw.shape[0])
  nsig = signal_raw.shape[0]
  nbkg = bkg_raw.shape[0]
  
  b_label_test = np.zeros(bkg_raw_test.shape[0])
  data_train = signal_raw.append(bkg_raw)
  label_train = np.concatenate((s_label,b_label))
  
  data_test = signal_raw_test.append(bkg_raw_test)
  label_test = np.concatenate((s_label_test, b_label_test))
  
  data_train = data_train.to_numpy()
  data_test = data_test.to_numpy()
  data_train, label_train = shuffle(data_train, label_train)
  data_test, label_test = shuffle(data_test, label_test)

  print("Training: signal: {0} bkg: {1}".format(signal_raw.shape[0], bkg_raw.shape[0]))
  print("Testing: signal: {0} bkg: {1} ".format(signal_raw_test.shape[0], bkg_raw_test.shape[0]))
  print(data_train.shape)
  print(label_train.shape)
  if(scale):
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    dump(scaler,'scaler.bin')
  return data_train, label_train, data_test, label_test, nsig, nbkg


def importTestData(path, var, scale=True):
  '''
  import csv data for testing
  path: file paths
  var: variables to fed into training
  scale: whether to scale the data
  '''
  if path[-1]!='/':
    path += '/'
  if(scale):
    scaler = load('scaler.bin')
  signal_raw = pd.read_csv(path + 'testggToNN_800M_1m.csv')[var]
  if(scale):
    signal_raw = scaler.transform(signal_raw)
  bkg_name = [path + 'testQCD_HT700to1000.csv', 
              path + 'testQCD_HT1000to1500.csv',
              path + 'testQCD_HT1500to2000.csv', 
              path + 'testQCD_HT2000toInf.csv', 
              path + 'testTTJets_HT600To800.csv', 
              path + 'testTTJets_HT800To1200.csv', 
              path + 'testTTJets_HT1200To2500.csv',
              path + 'testTTJets_HT2500ToInf.csv']
  bkg = []
  for b in bkg_name:
      bkg_raw = pd.read_csv(b)[var]
      if(scale):
        bkg_raw = scaler.transform(bkg_raw)
      bkg.append(bkg_raw)

  return signal_raw, bkg


def importData(path, var, scale=True):
  '''
  import csv data
  path: file paths
  var: variables to fed into training
  scale: whether to scale the data
  '''
  if path[-1]!='/':
    path += '/'
  if(scale):
    scaler = load('scaler.bin')
  signal_raw = pd.read_csv(path + 'ggToNN_800M_1mm_wEvt.csv')[var]
  if(scale):
    signal_raw = scaler.transform(signal_raw)
  bkg_name = [path + 'QCD_HT700to1000_wEvt.csv', 
              path + 'QCD_HT1000to1500_wEvt.csv',
              path + 'QCD_HT1500to2000_wEvt.csv', 
              path + 'QCD_HT2000toInf_wEvt.csv', 
              path + 'TTJets_HT600To800_wEvt.csv', 
              path + 'TTJets_HT800To1200_wEvt.csv', 
              path + 'TTJets_HT1200To2500_wEvt.csv',
              path + 'TTJets_HT2500ToInf_wEvt.csv']
  bkg = []
  for b in bkg_name:
      bkg_raw = pd.read_csv(b)[var]
      if(scale):
        bkg_raw = scaler.transform(bkg_raw)
      bkg.append(bkg_raw)

  return signal_raw, bkg

def make_model(input_dim, metrics=None, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  if not metrics:
    metrics = [
          keras.metrics.BinaryAccuracy(name='accuracy'),
          #keras.metrics.Precision(name='precision'),
          #keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
    ]

  model = keras.models.Sequential()
  model.add(keras.Input(shape=(input_dim,)))
  #model.add(keras.layers.Dense(1024, activation='relu'))
  #model.add(keras.layers.Dense(512, activation='relu'))
  #model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.BatchNormalization())
  #model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(32, activation='relu'))
  model.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))

  model.compile(
      #optimizer=keras.optimizers.SGD(learning_rate=1),
      optimizer=keras.optimizers.Adam(lr=1e-3),
      #optimizer=keras.optimizers.Adadelta(),
      #loss='binary_crossentropy',
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  model.summary()
  return model


def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.plot(history.epoch,  history.history['loss'],
               label='Train '+label)
  plt.plot(history.epoch,  history.history['val_loss'],
               label='Val '+label)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()

  plt.savefig('NN_loss.png')
  plt.close()

  return

def plot_metrics(history):
  #metrics =  ['loss', 'auc', 'precision', 'recall']
  metrics =  ['loss', 'auc']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    #if metric == 'loss':
    #  plt.ylim([0, plt.ylim()[1]])
    #elif metric == 'auc':
    #  plt.ylim([0.8,1])
    #else:
    #  plt.ylim([0,1])

    plt.legend()
    plt.savefig('NN_{0}.png'.format(metric))
    plt.close()
  return

def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  plt.savefig('NN_ROC.png')
  plt.close()
  return
