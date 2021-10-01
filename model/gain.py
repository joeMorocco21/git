import pandas as pd
import numpy as np
from gini import Gini

def gini_split(df, target, child):
    """Argument: df= DataFrame, target = panda.Series, child = panda.Series"""
    values = df[child].value_counts()
    split_gini = 0 
    for idx in values.keys():
        df_k = df[target][df[child]==idx]
        split_gini = split_gini + (values.loc[idx] / df[child].shape[0]) * Gini.gini(df_k)
    return split_gini

def gain(df, target, child):
    """Argument: df= DataFrame, target = panda.Series, child = panda.Series"""
    g = Gini.gini(df[target]) - gini_split(df, target, child)
    return round(g, 3)