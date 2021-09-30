import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
class Model():
    def models(depth,tree):
        df_clean = pd.read_csv(os.path.join(os.path.dirname(__file__),"../data/clean/cleanTitanic.csv"), sep=",")
        columns = ['Survived', 'Pclass', 'Sex', 'Embarked']
        for c in columns:
            df_clean[c] = df_clean[c].astype('category')
        df_categ = df_clean[['Sex', 'Embarked']]
        df_categ = pd.get_dummies(data=df_categ, prefix=["sex", "embark"])
        num_cols = df_clean.select_dtypes(exclude=['category']).columns
        df_num = df_clean[num_cols]
        X = pd.concat([df_clean.Pclass, pd.concat([df_categ, df_num], axis=1)], axis=1)
        y = df_clean.Survived
        df_clean.select_dtypes(exclude=['category']).columns
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        sc = StandardScaler()
        X_train = pd.concat([X_train.iloc[:,:6],pd.DataFrame(sc.fit_transform(X_train.iloc[:,6:]), index=X_train.index, 
                                                     columns=["Age","SibSp","Parch","Fare"])], axis=1)
        X_test = pd.concat([X_test.iloc[:,:6],pd.DataFrame(sc.fit_transform(X_test.iloc[:,6:]), index=X_test.index,
                                                   columns=["Age","SibSp","Parch","Fare"])], axis=1)
        rfc = RandomForestClassifier(criterion="gini",random_state = 43)
        rfc.fit(X_train, y_train)
        dtc = DecisionTreeClassifier(criterion="gini",random_state = 10)
        dtc.fit(X_train, y_train)
