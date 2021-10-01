import pytest
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from model.model import X_train, X_test, y_train, y_test


def test_depthRFC():
    accuracy = []
    for depth in range(4, 12):
        rfc = RandomForestClassifier(max_depth= depth, random_state = 43)
        rfc.fit(X_train, y_train)
        pred = np.round(rfc.predict(X_train))
        accuracy.append(accuracy_score(y_train, pred))
        
    assert sorted(accuracy) == accuracy
    

def RF_accuracy(n):
    rfc = RandomForestClassifier(random_state = 42, max_depth=n)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    acc=accuracy_score(y_test, predictions)
    return acc


def test_RFCdepth():    
    assert  [RF_accuracy(i)<RF_accuracy(i+1) for i in range (6,10)]