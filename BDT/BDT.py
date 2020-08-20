from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.utils import shuffle

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math as M
import pandas as pd


# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import sys
sys.path.insert(0, '/uscms/home/ali/nobackup/LLP/DVAnalyzer/ML')
from utils.NNUtils import *

def trainBDT(X, y, X_val, y_val, param):

    evallist = [(X, y), (X_val, y_val)]
    model = xgb.XGBClassifier(**param)
    model.fit(X, y.ravel(), eval_set=evallist, verbose=True, early_stopping_rounds=50)
    results = model.evals_result()
    ypred = model.predict(X_val)
    model.save_model("BDT_bestmodel.json")
    predictions = [round(value) for value in ypred]
    accuracy = accuracy_score(y_val, predictions)
    print("The training accuaracy is: {}".format(accuracy))
    conf_matrix = confusion_matrix(y_val, predictions)
    #print("The confusion matrix: {}".format(conf_matrix))
    #print("The precision is: {}".format(precision_score(y_val, predictions)))
    #print("The eval_result is: {}".format(model.get_booster().best_iteration))
    #print("The eval_result is: {}".format(model.get_booster().best_score))
    plot_BDTScore(X_val.copy(), y_val.copy(), model)
    plt.plot(results['validation_0']['auc'], label='Train')
    plt.plot(results['validation_1']['auc'], label='Test')
    plt.legend()
    plt.title("learning curve")
    plt.savefig("LC.png")
    plt.close()
    return


def plot_BDTScore(X_val, y_val, model):
    sig_index = np.asarray(np.where(y_val==1))[0,:]
    bkg_index = np.asarray(np.where(y_val==0))[0,:]
    X_sig = X_val[sig_index,:]
    X_bkg = X_val[bkg_index,:]
    pred_sig = model.predict_proba(X_sig)[:,1]
    pred_bkg = model.predict_proba(X_bkg)[:,1]
    #returnBestCutValue('BDT',pred_sig.copy(), pred_bkg.copy(), _testingFraction=0.3)
    plt.hist(pred_sig, bins=100, alpha=0.5, density=True, label="signal")
    plt.hist(pred_bkg, bins=100, alpha=0.5, density=True, label="background")
    plt.legend(loc="best")
    plt.title("BDT score")
    plt.savefig("ROC.png")
    plt.close()
    return

    
def plot_learning_curve(X, y, param, nClus):
    train_sizes, train_scores, test_scores = learning_curve(xgb.XGBClassifier(**param), X, y, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing score")
    plt.legend(loc="best")
    plt.title("{} clusters learning curve".format(nClus))
    plt.savefig("learningCurve.png")
    plt.close()
    return
    
def plotCostFunc_kClus(data, nClus):
    #find out the "best" value of n_clusters to perform k-means clustering
    cost = []
    for i in range(1,nClus+1):
        ki = KMeans(n_clusters=i, random_state=0).fit(data)
        cost.append(ki.inertia_)
    plt.plot(range(1,nClus+1),cost, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('cost function')
    plt.show()
    

def KClustering(X, y, X_test, y_test, nClusters, usePCA, n_vars):
    if(usePCA):
        #process data with PCA
        #find the number of features that keep 95% variance
        print("Doing PCA...")
        variance_threshold = 0.95
        num_components = n_vars
        pca_trail = PCA()
        pca_trail.fit(X)
        var = np.cumsum(pca_trail.explained_variance_ratio_)
        for n_com in range(1,len(var)-1):
            if(var[n_com]>variance_threshold):
                num_components = n_com
                break

        print("Doing k-means clustering with {0} features...".format(num_components))
        pca = PCA(n_components=num_components)
        pca.fit(X)
        X_train_pca = pca.transform(X)
        X_test_pca = pca.transform(X_test)
        print("Shape of new training dataset: {}".format(X_train_pca.shape))
        print("Shape of new testing dataset: {}".format(X_test_pca.shape))
        #do the k-means clustering
        kmeans = KMeans(n_clusters=nClusters, random_state=0, verbose=0).fit(X_train_pca)
        score_train = kmeans.transform(X_train_pca)
        score_test = kmeans.transform(X_test_pca)
    else:
        #do k-means clustering
        print("Doing k-means clustering...")
        kmeans = KMeans(n_clusters=nClusters, random_state=0, verbose=0).fit(X)
        score_train = kmeans.transform(X)
        score_test = kmeans.transform(X_test)
        
    score_train_norm = scale(score_train)
    score_test_norm = scale(score_test)
    y_np = y.to_numpy()
    y_test_np = y_test.to_numpy()
    print("Finished clustering. :)")
    
    return score_train_norm, y_np, score_test_norm, y_test_np

def trainBDT_pyswarm(X_train, y_train, X_val, y_val, param):

    evallist = [(X_train, y_train), (X_val, y_val)]
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train.ravel(), eval_set=evallist, verbose=False, early_stopping_rounds=30)
    best_score = 1.0-model.get_booster().best_score
    return best_score

def f_BDT(x,data,labels):
    # *** 1. split our data set to training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.25, shuffle=True, random_state=7)
    n_particles = x.shape[0]
    BDT_result = []    #used to store the BDT training results for each particle
    for i in range (n_particles):
        # *** 2. set hyper parameters
        param = {
            'n_jobs': -1,
            'eta': 0.3,
            'n_estimators': 5000,
            'max_depth': int(x[i,0]*7+2),
            'min_child_weight': x[i,1],
            'subsample': x[i,2],
            'colsample_bytree': x[i,3],
            'gamma': x[i,4],
            'reg_alpha': x[i,5], 
            'reg_lambda': x[i,6],
            'scale_pos_weight': 1,
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'random_state': 27
        }
        BDT_result.append(trainBDT_pyswarm(X_train, y_train, X_val, y_val, param))
    return np.array(BDT_result)


# *** 1. Import data and check stuff
testingFraction = 0.3
testDataSize = 2000
var = ['vtx_track_size', 'vtx_dBV', 'vtx_sigma_dBV', 'vtx_x', 'vtx_y', 'vtx_z']

# *** A. Import Dataset
#signal = pd.read_csv('2018MinoAOD/csvFiles/trainggToNN_800M_1m.csv')[var]
#signal = shuffle(signal)
#signal_raw = signal[:-testDataSize]
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
#pos_weight = bkg_raw.shape[0]/signal_raw.shape[0]
#b_label = np.zeros(bkg_raw.shape[0])
#
#b_label_test = np.zeros(bkg_raw_test.shape[0])
#data_train = signal_raw.append(bkg_raw)
#label_train = np.concatenate((s_label,b_label))
#
#data_test = signal_raw_test.append(bkg_raw_test)
#label_test = np.concatenate((s_label_test, b_label_test))
#
#print("Training: signal: {0} bkg: {1} fraction: {2}".format(signal_raw.shape[0], bkg_raw.shape[0], pos_weight))
#print("Testing: signal: {0} bkg: {1} ".format(signal_raw_test.shape[0], bkg_raw_test.shape[0]))
#print(data_train.shape)
#print(label_train.shape)


# *** 2. Make mix of dihiggs and QCD for specified variables
#variables = ['deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'hh_mass', 'h1_mass', 'h2_mass','hh_pt', 'h1_pt', 'h2_pt', 'scalarHT']
#variables = ['hh_mass', 'h1_mass', 'h2_mass']

data_train, label_train, data_test, label_test, nsig, nbkg = importTrainData('2018MinoAOD/csvFiles/', var, testDataSize, False)
pos_weight = nbkg/nsig

#data_train, data_test, labels_train, labels_test = train_test_split(data, label, test_size=testingFraction, shuffle= True, random_state=30)

#data_train_norm = scale(data_train)
#data_test_norm = scale(data_test)



# *** Define parameters for BDT
param = {
        #'n_jobs': -1,
        'eta': 0.3,
        'n_estimators': 1000,
        'max_depth': 9,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0, 
        'reg_lambda': 1.5,
        'scale_pos_weight': pos_weight,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'random_state': 27
}

trainBDT(data_train.copy(), label_train.copy(), data_test.copy(), label_test.copy(), param)


