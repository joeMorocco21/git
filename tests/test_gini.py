import pytest
import pandas as pd
import numpy as np
import math
import os
import sys
sys.path.append('../model')
from gini import Gini
df = pd.read_csv(os.path.join(os.path.dirname(__file__),"../data/clean/cleanTitanic.csv"), encoding="UTF8", sep=',')
    #df = pd.read_csv("cleanTitanic.csv")

def test_gini():
            """Unit test to assess on gini score."""
            expected_gini = 0.475
            gini_score = Gini.gini(df.Survived)
            assert math.fabs(gini_score - expected_gini) <= 0.001
            assert Gini.gini(pd.Series([1, 0, 0, 0, 0, 1])) == 0.444
            assert Gini.gini(pd.Series([1, 0, 0, 0, 0, 0])) == 0.278
            assert round(Gini.gini(pd.Series([1, 1, 1, 1, 1, 1, 1, 1])), 3) == 0
            assert round(Gini.gini(pd.Series([1, 1, 1, 1, 1, 1, 0, 0])), 3) == 0.375
            assert round(Gini.gini(pd.Series([1, 1, 1, 1, 0, 0, 0, 0])), 3) == 0.500