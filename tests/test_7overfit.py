import pytest
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from model.model import X_train, X_test, y_train, y_test
   
    
def test_overfitRTC():
    rfc = pickle.load(open(os.path.join(os.path.dirname(__file__),'../model/RFC_model.pkl'), 'rb'))
    pred_train = np.round(rfc.predict(X_train))
    pred_test = np.round(rfc.predict(X_test))
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
                                        
    assert (acc_train - acc_test)/acc_train <= 0.1