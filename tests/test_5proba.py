import pytest
import pandas as pd
import numpy as np
import pickle


entree = np.array([[1., 24., 15.2458, 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.],
 [0., 31., 10.5 , 1., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
 [0., 31., 37.0042, 0., 0., 1., 0., 0., 1., 0., 1., 0., 0.],
 [0., 20., 4.0125, 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.]])


def test_proba():
    """ Testing probabilites are within [0,1]"""
    rfc = pickle.load(open(os.path.join(os.path.dirname(__file__),"../src/'RFC_model.pkl', 'rb'))
    proba = rfc.predict_proba(entree)
    pb=pd.DataFrame(proba, columns=['yes','no'])
    sum_pb = pb['yes']+pb['no']
    assert [pb[col].between(0, 1, inclusive=True).any() for col in pb.columns]
    assert [sum_pb.between(0, 1, inclusive=True).any()]
    assert (proba <= 1).all() & (proba >= 0).all()