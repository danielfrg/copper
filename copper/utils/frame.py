# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd

from sklearn import decomposition
from sklearn.feature_selection import RFE, SelectPercentile, f_classif
'''
Utils for a pandas Dataframe
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

def PCA(data, ret_model=False, **args):
    ''' Calculates the PCA Decomposition of the frame
    '''
    X = data.values
    model = decomposition.PCA(**args)
    model.fit(X)
    transformed = pd.DataFrame(model.transform(X), index=data.index)
    return (transformed, model) if ret_model else transformed

def features_weight(X, y, ascending=False):
    '''
    Paremeters
    ----------
        X: pd.DataFrame
        y: pd.Series
    '''
    selector = SelectPercentile(f_classif)
    selector.fit(X.values, y.values)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    ans = pd.Series(scores, index=X.columns)
    return ans.order(ascending=ascending)

def rce_rank(X, y, n_features_to_select=None, estimator=None):
    '''
    Paremeters
    ----------
        X: pd.DataFrame
        y: pd.Series
    '''
    if estimator is None:
        from sklearn.svm import SVC
        estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select, step=1)
    selector = selector.fit(X.values, y.values)
    ans = pd.Series(selector.ranking_, index=X.columns)
    return ans.order(ascending=False)
    

#-----------------------  OUTLIERS  --------------------------------------------

def outlier_rows(data, width=1.5):
    ''' Returna a series/frame with the outliers filter array [Trues and Falses]

    Parameters
    ----------
        data: pd.Series or pd.Frame
    '''
    if type(data) is pd.Series:
        q1 = data.describe()[4]
        q3 = data.describe()[6]
        iqr = q3 - q1
        lower_limit = q1 - width * iqr
        upper_limit = q3 + width * iqr
        return (data < lower_limit) | (data > upper_limit)
    elif type(data) is pd.DataFrame:
        ans = pd.DataFrame(index=data.index)
        for col in data.columns:
            new = outlier_rows(data[col], width=width)
            ans = ans.join(new)
        return ans

def outliers(series, width=1.5):
    '''
    Get the outliers of that column
    '''
    return series[outlier_rows(series, width=width)]

def outlier_count(data, width=1.5, ascending=False):
    ''' Returns a series/int with the number of outliers of each column
    
    Parameters
    ----------
        data: pd.Series or pd.DataFrame
    '''
    if type(data) is pd.Series:
        return len(outliers(data, width=width))
    elif type(data) is pd.DataFrame:
        ans = pd.Series(index=data.columns)
        for col in data.columns:
            ans[col] = outlier_count(data[col], width=width)
        return ans.order(ascending=ascending)