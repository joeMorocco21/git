import os
import pytest
import pandas as pd
import numpy as np

from sklearn.metrics import  accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.model import X_train, X_test, y_train, y_test


df = pd.read_csv(os.path.join(os.path.dirname(__file__),"../data/clean/cleanTitanic.csv"), sep=",")


def test_accRFC():
    dtc = DecisionTreeClassifier(random_state = 42)
    dtc.fit(X[:801], y[:801])
    dtc_pred = dtc.predict(X[801:])
    dtc_acc = accuracy_score(y[801:], dtc_pred)
    acc_list = []
    for tree in [1, 4, 8, 20]:
        rfc = RandomForestClassifier(n_estimators= tree, random_state = 42)
        rfc.fit(X[:801], y[:801])
        rfc_pred = rfc.predict(X[801:])
        acc_list.append(accuracy_score(X[801:], rfc_pred))
        
    assert min(acc_list) >= dtc_acc
    
    
def test_accDTC():
    dtc = DecisionTreeClassifier(random_state = 10, max_depth= 8)
    dtc.fit(df.iloc[:801, 1:], df.iloc[:801, :1])
    dtc_pred = dtc.predict(df.iloc[:, 1:])
    dtc_acc = accuracy_score(df.iloc[801:, :1], dtc_pred)
    rfc = RandomForestClassifier(n_estimators= 7, max_depth= 8, random_state = 42)
    rfc.fit(df.iloc[:801, 1:], df.iloc[:801, :1])
    rfc_pred = np.round(rfc.predict(df.iloc[801:, 1:]))
    dtc_acc = accuracy_score(df.iloc[801:, :1], rfc_pred)
        
    assert rfc_acc >= dtc_acc