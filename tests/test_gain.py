import pytest
import pandas as pd
import numpy as np

import sys
sys.path.append('../model')
from gain import gain
from gain import gini_split    
def test_gain():
    target = pd.Series([1, 1, 1, 1, 0, 0, 0, 0], name="target")
    child = pd.Series(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'a'], name="child")
    df = pd.concat([target, child], axis=1)
    assert round(gain(df, "target", "child"), 3) == 0.125
                 
                 
def test_gain():
    target = pd.Series([1, 0, 1, 0, 1, 0, 1, 1], name="target")
    child = pd.Series(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'a'], name="child")
    df = pd.concat([target, child], axis=1)
    assert round(gain(df, "target", "child"), 3) == 0.031