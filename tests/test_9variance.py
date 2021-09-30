import os
import pytest
import pandas as pd
import numpy as np

import pickle
from src.model import X_test, X


def test_sex():
    rfc = pickle.load(open(os.path.join(os.path.dirname(__file__),"../src/'RFC_model.pkl', 'rb'))
    df_female = X_test.copy()
    df_male = X_test.copy()
    df_female["Sex"]=1
    df_male["Sex"]=0
    prob_female = rfc.predict_proba(df_female)[:, 0]
    prob_male = rfc.predict_proba(df_male)[:, 0]
    nb_female = np.count_nonzero((prob_female > 0.5))
    nb_male = np.count_nonzero((prob_male > 0.5))
    assert nb_female  >= nb_male
    assert (nb_female - nb_male)/(nb_female + nb_male) > 90/100
    assert [prob_female[i] >= prob_male[i] for i in range(prob_female.shape[0])]


# Check survival probability variance (nbr)
def test_class():
    rfc = pickle.load(open(os.path.join(os.path.dirname(__file__),"../src/'RFC_model.pkl', 'rb'))
    df_class = X_test[X_test["Pclass"]==3].copy()
    class3 = rfc.predict_proba(df_class)[0]
    df_class['Pclass'] = 2
    class2 = rfc.predict_proba(df_class)[0]
    df_class['Pclass'] = 1
    class1 = rfc.predict_proba(df_class)[0]
    assert class3 > class2
    assert class2 > class1


def test_fare():
    rfc = pickle.load(open(os.path.join(os.path.dirname(file),"../src/'RFC_model.pkl', 'rb'))
    df_fare = X_test[X_test["Fare"]==8].copy()
    fare8 = rfc.predict_proba(df_fare)[0]
    df_fare['Fare'] = 30
    fare30 = rfc.predict_proba(df_fare)[0]
    df_fare['Fare'] = 80
    fare80 = rfc.predict_proba(df_fare)[0]
    assert fare8 > fare30
    assert fare30 > fare80
                                        
    nb_8 = np.count_nonzero((fare8 > 0.5))
    nb_26 = np.count_nonzero((fare26 > 0.5))
    nb_13 = np.count_nonzero((fare13 > 0.5))
    assert nb_8  >= nb_26
    assert nb_13 >= nb_26
    assert nb_8 >= nb_13          
                                        
                                        
def test_port():
    rfc = pickle.load(open(os.path.join(os.path.dirname(__file__),"../src/'RFC_model.pkl', 'rb'))
    s_df = X[X["Embarked"]=='S'].copy()
    s_prob = rfc.predict_proba(s_df)[0]
    c_df = X[X["Embarked"]=='S'].copy()
    c_df['Embarked'] = 'C'
    c_prob = rfc.predict_proba(c_df)[0]
    q_df = X[X["Embarked"]=='S'].copy()
    q_df['Embarked'] = 'Q'
    q_prob = rfc.predict_proba(q_df)[0]
    ns = np.count_nonzero((s_prob > 0.5))
    nc = np.count_nonzero((c_prob > 0.5))
    nq = np.count_nonzero((q_prob > 0.5))
    
    assert ns  <= nc
    assert nq  >= nc
    assert ns  <= nq