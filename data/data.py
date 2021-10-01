import pandas as pd
import numpy as np
import os
class Clean():
    def cleaning():
        df_raw = pd.read_csv(os.path.join(os.path.dirname(__file__),"row/Titanic.csv"), sep=";")
        df_raw.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
        columns = ['Survived', 'Pclass', 'Sex', 'Embarked']

        for c in columns:
            df_raw[c] = df_raw[c].astype('category')
                        
        for g in list(df_raw.Sex.unique()):
            for c in list(df_raw.Pclass.unique()):
                df_raw.update(df_raw.loc[(df_raw.Sex==g) & (df_raw.Pclass==c), 
                'Age'].fillna(value=round(df_raw.loc[(df_raw.Sex==g) & (df_raw.Pclass==c), 'Age'].mean())))
            
        df_raw['Age'] = np.round(df_raw['Age'])
        df_raw['Fare'] = np.round(df_raw['Fare'])
        df_raw = df_raw[df_raw["Fare"]>0]
        df_raw['Embarked'] = df_raw['Embarked'].fillna(df_raw['Embarked'].value_counts().index[0])
        df_raw.to_csv(os.path.join(os.path.dirname(__file__),"clean/cleanTitanic.csv"),index=False)
        