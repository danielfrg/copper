# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd

from sklearn import decomposition

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

def PCA(data, n_components, ret_model=False):
    X = None
    if type(data) is copper.Dataset:
        X = copper.transform.inputs2ml(data)
    elif type(data) is pd.DataFrame:
        X = data.values

    model = decomposition.PCA(n_components=n_components)
    model.fit(X)
    if ret_model:
        return model.transform(X), model
    else:
        return model.transform(X)

# -------------------------------------------------------------------------------------------
#                                          OUTLIERS

def outlier_rows(series_or_frame, width=1.5):
    '''
    Get the outliers filter array [Trues and Falses]
    '''
    if type(series_or_frame) is pd.Series:
        q1 = series_or_frame.describe()[4]
        q3 = series_or_frame.describe()[6]
        iqr = q3 - q1
        lower_limit = q1 - width * iqr
        upper_limit = q3 + width * iqr
        return (series_or_frame < lower_limit) | (series_or_frame > upper_limit)
    elif type(series_or_frame) is pd.DataFrame:
        ans = pd.Series(np.zeros(len(series_or_frame)), dtype=object)
        ans[:] = False
        for col, series in series_or_frame.iteritems():
            ans = ans | outlier_rows(series, width=width)
        return ans

def outliers(series, width=1.5):
    '''
    Get the outliers of that column
    '''
    return series[outlier_rows(series, width=width)]

def outlier_count(series, width=1.5):
    '''
    Get the number of outliers of that colums
    '''
    return len(outliers(series, width=width))