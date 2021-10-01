import numpy as np
import pandas as pd
import pytest

class Test():
    def test_NbrColumns():
            assert (list(df.columns)) == ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # Check if Gender type
    def test_Gender():
        assert sorted(df['Sex'].unique().tolist()) == ['female', 'male']

    # Check Fare consistency
    def test_Fare():
        assert df[df['Fare'] > 0].shape[0]  == 0

    # Check Pclass unique values
    def test_PropertyArea():
        assert sorted(df['Pclass'].unique().tolist()) == [1, 2, 3]

    # Check Age consistency (positif and below 81
    def test_Age():
        assert df[(df['Age'] < 0) | (df['Age'] > 81)].shape[0] != 0