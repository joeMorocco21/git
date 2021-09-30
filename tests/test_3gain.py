import os
import pytest
import pandas as pd
import numpy as np
from src.gini import gini
from src.gain import gain

df = pd.read_csv(os.path.join(os.path.dirname(__file__),"../data/clean/cleanTitanic.csv"), encoding="UTF8", sep=',')
#df = pd.read_csv(r'../src/cleanTitanic.csv', encoding="UTF8", sep=',')


def test_gain():
    assert round(gain(df, "Survived", "Age"), 3) == 0.078
    
                 
def test_gain():
    target = pd.Series([1, 0, 1, 0, 1, 0, 1, 1], name="target")
    child = pd.Series(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'a'], name="child")
    df = pd.concat([target, child], axis=1)
    assert round(gain(df, "target", "child"), 3) == 0.031

    
def test_gain():
    target = pd.Series([1, 0, 1, 0, 1, 0, 1, 1], name="target")
    child = pd.Series(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'a'], name="child")
    df = pd.concat([target, child], axis=1)
    assert round(gain(df, "target", "child"), 3) == 0.031