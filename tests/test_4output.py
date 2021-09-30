import pytest
import pandas as pd
import numpy as np
import pickle
from src.model import X 


def test_check():
    entree = X.iloc[502:505,:]
    rfc = pickle.load(open(os.path.join(os.path.dirname(__file__),"../src/'RFC_model.pkl', 'rb'))
    sortie = rfc.predict(entree)
    assert sortie.shape[0] == entree.shape[0]