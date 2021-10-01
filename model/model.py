# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

raw = pd.read_csv(os.path.join(os.path.dirname(__file__),"../data/row/Titanic.csv"), encoding="UTF8", sep=';')
df = raw.copy()

df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
columns = ['Survived', 'Pclass', 'Sex', 'Embarked']


for c in columns:
    df[c] = df[c].astype('category')
    
    
for g in list(df.Sex.unique()):
    for c in list(df.Pclass.unique()):
        df.update(df.loc[(df.Sex==g) & (df.Pclass==c), 
                         'Age'].fillna(value=round(df.loc[(df.Sex==g) & (df.Pclass==c), 'Age'].mean())))
        

df.to_csv(os.path.join(os.path.dirname(__file__),"../data/clean/Titanic_cleaned.csv"),index=False)


df['Age'] = np.round(df['Age'])
df['Fare'] = np.round(df['Fare'])
df = df[df["Fare"]>0]
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])


df['Age'] = df.apply(lambda r: 'child' if r['Age']<10 else ('young' if r['Age']<18  else ('adult' if r['Age']<36 else ('senior' if r['Age']<55 else 'old'))), axis=1)

df['Fare'] = df.apply(lambda r: 'cheap' if r['Fare']<10 else ('low' if r['Fare']<20 else ('hilow' if r['Fare']<30 else ('fair' if r['Fare']<40 else ('middle' if r['Fare']<50 else ('lowhi' if r['Fare']<80 else ('hi' if r['Fare']<100 else 'huge')))))), axis=1)

dfs = df.dropna(axis=0)

labelencoder = LabelEncoder()
dfs["Sex"] = labelencoder.fit_transform(dfs["Sex"])
dfs["Embarked"] = labelencoder.fit_transform(dfs["Embarked"])
dfs["Age"] = labelencoder.fit_transform(dfs["Age"])
dfs["Fare"] = labelencoder.fit_transform(dfs["Fare"])


dfs.to_csv(os.path.join(os.path.dirname(__file__),"../data/clean/Titanic_dummied.csv"),index=False)

X = dfs.iloc[:, 1:]
y = dfs.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y, shuffle=True)

rfc = RandomForestClassifier(criterion="gini", random_state = 43)
rfc.fit(X_train, y_train)

dtc = DecisionTreeClassifier(criterion="gini", random_state = 10)
dtc.fit(X_train, y_train)



with open(os.path.join(os.path.dirname(__file__),'../model/DTC_model.pkl'), 'wb') as file:
    pickle.dump(dtc, file)
    
with open(os.path.join(os.path.dirname(__file__),'../model/RFC_model.pkl'), 'wb') as file:
    pickle.dump(rfc, file)
print(dfs.iloc[:801, 1:])
print(dfs.iloc[:801, :1])     