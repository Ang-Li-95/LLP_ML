import pandas as pd
import numpy as np
from joblib import dump
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/uscms/home/ali/nobackup/LLP/DVAnalyzer/ML')
from utils.NNUtils import *

#def make_model(input_dim, metrics, output_bias=None):
#  if output_bias is not None:
#    output_bias = tf.keras.initializers.Constant(output_bias)
#  model = keras.models.Sequential()
#  model.add(keras.Input(shape=(input_dim,)))
#  model.add(keras.layers.Dense(input_dim, activation='relu'))
#  model.add(keras.layers.Dense(512, activation='relu'))
#  model.add(keras.layers.BatchNormalization())
#  model.add(keras.layers.Dense(256, activation='relu'))
#  model.add(keras.layers.Dense(128, activation='relu'))
#  #model.add(keras.layers.Dropout(0.5))
#  model.add(keras.layers.Dense(32, activation='relu'))
#  model.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))
#
#  #model = keras.Sequential([
#  #    keras.layers.Dense(
#  #        16, activation='relu',
#  #        input_dim=input_dim),
#  #    keras.layers.Dropout(0.5),
#  #    keras.layers.Dense(1, activation='sigmoid',
#  #                       bias_initializer=output_bias),
#  #])
#
#  model.compile(
#      #optimizer=keras.optimizers.SGD(learning_rate=1),
#      optimizer=keras.optimizers.Adam(lr=1e-3),
#      #optimizer=keras.optimizers.Adadelta(),
#      #loss='binary_crossentropy',
#      loss=keras.losses.BinaryCrossentropy(),
#      metrics=metrics)
#
#  return model


#def auc( y_true, y_pred ) :
#      score = tf.py_function( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
#                          [y_true, y_pred],
#                          'float32',
#                          name='sklearnAUC' )
#      return score

# *** 1. Import data and check stuff
testingFraction = 0.3
testDataSize = 2000
var = ['vtx_track_size', 'vtx_dBV', 'vtx_sigma_dBV', 'vtx_x', 'vtx_y', 'vtx_z']
#var = ['vtx_dBV', 'vtx_sigma_dBV']

# *** A. Import Dataset
#signal = pd.read_csv('2018MinoAOD/csvFiles/trainggToNN_800M_1m.csv')[var]
#signal = signal.dropna()
#signal = shuffle(signal)
#signal_raw = signal[:-testDataSize]
#signal_raw = signal_raw
#s_label = np.ones(signal_raw.shape[0])
#signal_raw_test = signal[-testDataSize:]
#s_label_test = np.ones(signal_raw_test.shape[0])
#bkg_name = ['2018MinoAOD/csvFiles/trainQCD_HT700to1000.csv', 
#            '2018MinoAOD/csvFiles/trainQCD_HT1000to1500.csv',
#            '2018MinoAOD/csvFiles/trainQCD_HT1500to2000.csv', 
#            '2018MinoAOD/csvFiles/trainQCD_HT2000toInf.csv', 
#            '2018MinoAOD/csvFiles/trainTTJets_HT600To800.csv', 
#            '2018MinoAOD/csvFiles/trainTTJets_HT800To1200.csv', 
#            '2018MinoAOD/csvFiles/trainTTJets_HT1200To2500.csv',
#            '2018MinoAOD/csvFiles/trainTTJets_HT2500ToInf.csv']
#for b in bkg_name:
#    bkg = pd.read_csv(b)[var]
#    bkg = bkg.dropna()
#    bkg = shuffle(bkg)
#    try:
#        bkg_raw
#    except NameError:
#      bkg_raw = bkg[:-testDataSize]
#    else:
#      bkg_raw = bkg_raw.append(bkg[:-testDataSize])
#    try:
#        bkg_raw_test
#    except NameError:
#      bkg_raw_test = bkg[-testDataSize:]
#    else:
#      bkg_raw_test = bkg_raw_test.append(bkg[-testDataSize:])
#        
#bkg_raw = bkg_raw
#pos_weight = bkg_raw.shape[0]/signal_raw.shape[0]
#b_label = np.zeros(bkg_raw.shape[0])
#
#b_label_test = np.zeros(bkg_raw_test.shape[0])
#data_train = signal_raw.append(bkg_raw)
#label_train = np.concatenate((s_label,b_label))
#data_train, label_train = shuffle(data_train, label_train)
#
#data_test = signal_raw_test.append(bkg_raw_test)
#label_test = np.concatenate((s_label_test, b_label_test))
#data_test, label_test = shuffle(data_test, label_test)

#data_train, data_test, label_train, label_test = train_test_split(data_train, label_train, test_size=testingFraction, shuffle=True)

#nsig = signal_raw.shape[0]
#nbkg = bkg_raw.shape[0]
#weight_0 = (1.0 / nbkg)*(nbkg+nsig)/2.0
#weight_1 = (1.0 / nsig)*(nbkg+nsig)/2.0
#class_weight = {0: weight_0, 1: weight_1}
#
#print("Training: signal: {0} bkg: {1} fraction: {2}".format(signal_raw.shape[0], bkg_raw.shape[0], pos_weight))
#print("Testing: signal: {0} bkg: {1} ".format(signal_raw_test.shape[0], bkg_raw_test.shape[0]))
#print(data_train.shape)
#print(label_train.shape)
#
##data_train = scale(data_train)
##data_test = scale(data_test)
#scaler = StandardScaler()
#data_train = scaler.fit_transform(data_train)
#data_test = scaler.transform(data_test)
#dump(scaler,'scaler.bin')

data_train, label_train, data_test, label_test, nsig, nbkg = importTrainData('2018MinoAOD/csvFiles/', var, testDataSize)
weight_0 = (1.0 / nbkg)*(nbkg+nsig)/2.0
weight_1 = (1.0 / nsig)*(nbkg+nsig)/2.0
class_weight = {0: weight_0, 1: weight_1}

dropout=0.5
metrics = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      #keras.metrics.Precision(name='precision'),
      #keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
initial_bias = np.log([(1.0*nsig)/nbkg])
model = make_model(input_dim=data_train.shape[-1], metrics=metrics, output_bias = initial_bias)
model.summary()

##Create model
#model = Sequential()
##model.add(Dense(175, activation='relu', input_dim=data_train.shape[1], kernel_regularizer=l2_reg))
#model.add(Dense(4, activation='relu', input_dim=data_train.shape[1]))
#model.add(Dropout(dropout))
#model.add(BatchNormalization())
#model.add(Dense(2, activation='relu'))
#
#model.add(Dense(1, activation='sigmoid'))
## Compile model
#model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy', auc, keras.metrics.AUC(name='auc_tf')])
# Setup callbacks
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=20,
                               restore_best_weights=True)

checkpoint = ModelCheckpoint('weights.h5', monitor='val_auc',
                             mode='max', save_best_only=True, verbose=1)

lr_decay = LearningRateScheduler(schedule=lambda epoch: 1e-3 * (0.9 ** epoch))

# Train model
history = model.fit(data_train, label_train, validation_data=(data_test, label_test), epochs=200,
                    callbacks=[es,checkpoint,lr_decay], batch_size=2048, class_weight=class_weight)
plot_metrics(history)
plot_loss(history, "NN", 0)
