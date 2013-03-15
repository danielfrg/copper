# coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd

'''
Util for a pandas Dataframe
'''

def percent_missing(frame, ascending=False):
    '''
    Generetas a Series with the percent of missing values of each column

    Parameters
    ----------
        ascending: boolean, sort the returned Series on this direction

    Returns
    -------
        pandas.Series
    '''
    return (1 - (frame.count() / len(frame))).order(ascending=ascending)

def unique_values(frame, ascending=False):
    '''
    Generetas a Series with the number of unique values of each column.
    Note: Excludes NA

    Parameters
    ----------
        ascending: boolean, sort the returned Series on this direction

    Returns
    -------
        pandas.Series
    '''
    ans = pd.Series(index=frame.columns)
    for col in frame.columns:
        ans[col] = len(frame[col].value_counts())
    return ans.order(ascending=ascending)