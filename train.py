#!/usr/bin/python3
# train.py
# Xavier Vasques 13/04/2021

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import preprocessing

training = "./data/train.csv"
df_train = pd.read_csv(training)
df_train['diagnosis'].replace('M', 1, inplace=True)
df_train['diagnosis'].replace('B', 0, inplace=True)

def train():
    # Load, read and normalize training data
    training = "./data/train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['diagnosis'].values
    X_train = data_train.drop(data_train.loc[:, 'id':'diagnosis'].columns, axis = 1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Models training
    
    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)
    
    # Save model
    from joblib import dump
    dump(clf_lda, './models/Inference_lda.joblib')
        
    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(
        solver='adam', 
        activation='relu', 
        alpha=0.0001, 
        hidden_layer_sizes=(500,), 
        random_state=0, 
        max_iter=1000
        )
    
    clf_NN.fit(X_train, y_train)
       
    # Save model
    from joblib import dump
    dump(clf_NN, './models/Inference_NN.joblib')
        
if __name__ == '__main__':
    train()