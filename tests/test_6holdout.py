import pytest
import pandas as pd
import numpy as np
from src.model import X, y, X_train, y_train, X_test, y_test


def test_holdout():
    test_size = round(X_test.shape[0] / X.shape[0], 1)
    assert test_size == 0.2
    assert X_train.shape[1] == X_test.shape[1]
    assert (y_train.shape[0] + y_test.shape[0]) == y.shape[0]
    assert (list(X_test.columns)) == (list(X_train.columns))