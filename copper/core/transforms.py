# coding=utf-8
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing


# -----------------------------------------------------------------------------
#                                  PANDAS API
# -----------------------------------------------------------------------------

_numberRE = re.compile('[0-9.]+')
_numberREComp = re.compile(_numberRE)

def to_number(x):
    try:
        return float(_numberRE.search(x).group())
    except:
        return np.nan

'''
def to_number(series, RE='[0-9.]+'):
    _numberRE = re.compile(RE)

    def _numberREFun(x):
        try:
            return float(_numberRE.search(x).group())
        except:
            return np.nan

    return series.apply(_numberREFun)
'''

# def strptime(x, format='%m/%d/%Y'):
def strptime(x, *args):
    try:
        return datetime.strptime(x, ''.join(args))
    except:
        return np.nan

start_date = datetime(1970, 1, 1)

def date_to_number(x):
    try:
        return (x - start_date).days
    except:
        return np.nan

'''
def date2number(series, start_date=None):
    if start_date is None:
        start_date = datetime(1970, 1, 1)
    def _date2number(date):
        try:
            return (date - start_date).days
        except:
            return np.nan
    return series.apply(_date2number)
'''


# -----------------------------------------------------------------------------

def category2ml(series):
    '''
    Converts a Series with category format to a format for machine learning
    Represents the same information on different columns of ones and zeros

    Note: Fill/impute/drop missing values before using this.

    Parameters
    ----------
        series: pandas.Series, target to convert

    Returns
    -------
        pandas.DataFrame with the converted data
    '''
    ans = pd.DataFrame(index=series.index)
    categories = list(set(series))
    categories.sort()
    for category in categories:
        n_col = pd.Series(np.zeros(len(series)), index=series.index, dtype=int)
        n_col.name = '%s [%s]' % (series.name, category)
        n_col[series == category] = 1
        ans = ans.join(n_col)
    return ans

def category2number( series):
    '''
    Convert a Series with categorical information to a Series of numbers
    using the scikit-learn LabelEncoder

    Parameters
    ----------
        series: pandas.Series, target to convert

    Returns
    -------
        pandas.Series with the converted data
    '''
    le = preprocessing.LabelEncoder()
    le.fit(series.values)
    vals = le.transform(series.values)
    return pd.Series(vals, index=series.index, name=series.name, dtype=float)

def category_labels( series):
    '''
    Return the labels for a Series with categorical values

    Parameters
    ----------
        series: pandas.Series, target to convert

    Returns
    -------
        list, labels of the series
    '''
    le = preprocessing.LabelEncoder()
    le.fit(series.values)
    return le.classes_