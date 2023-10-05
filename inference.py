#!/usr/bin/python3


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import pandas as pd
from joblib import load
from sklearn import preprocessing

def inference():

    # Load, read and normalize training data
    testing = "./data/test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['diagnosis'].values
    X_test = data_test.drop(data_test.loc[:, 'id':'diagnosis'].columns, axis = 1)
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # Run model
    clf_lda = load('./models/Inference_lda.joblib')
    print("LDA score and classification:")
    print(clf_lda.score(X_test, y_test))
    print(clf_lda.predict(X_test))
        
    # Run model
    clf_nn = load('./models/Inference_NN.joblib')
    print("NN score and classification:")
    print(clf_nn.score(X_test, y_test))
    print(clf_nn.predict(X_test))
    
if __name__ == '__main__':
    inference()