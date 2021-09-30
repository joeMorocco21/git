import os
import pytest
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__),"../data/clean/cleanTitanic.csv"), sep=",")
#df = pd.read_csv(r'../src/cleanTitanic.csv', encoding="UTF8", sep=',')

def test_NbrColumns(): # Check number of columns
    assert (len(df.columns)) == 8
    
def test_NbrColumns(): # Check column names sequence
    assert (list(df.columns)) == ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def test_Gender(): # Check if Gender type
    assert sorted(df['Sex'].unique().tolist()) == ['female', 'male']

def test_Fare(): # Check Fare consistency
    assert df[df['Fare'] < 0].shape[0]  == 0
    assert df[df['Fare'] > 0].shape[0]  != 0

def test_Pclass(): # Check Pclass unique values
    assert sorted(df['Pclass'].unique().tolist()) == [1, 2, 3]

def test_Age(): # Check Age consistency (positive and below 81)
    assert df[(df['Age'] < 0) | (df['Age'] > 81)].shape[0] == 0
    assert df[(df['Age'] > 0) & (df['Age'] < 81)].shape[0] != 0