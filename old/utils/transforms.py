from __future__ import division
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing

# ---------------------    Pandas.apply API    ---------------------------------

numberRE = re.compile('[0-9.]+')
def to_number(x):
    ''' Uses a regular expression to extract the first number of a string

    Usage:
    df[col].apply(copper.transform.to_number)
    '''
    try:
        return float(numberRE.search(x).group())
    except:
        return np.nan

def strptime(x, *args):
    ''' Extracts a date from a string

    Usage:
    df[col].apply(copper.transform.strptime, args='%d/%m/%y')
    '''
    try:
        return datetime.strptime(x, ''.join(args))
    except Exception as e:
        print(e)
        return np.nan


start_date = datetime(1970, 1, 1)
def date_to_number(x):
    ''' Converts a date to a number
    Default start date = 1970/1/1

    Usage:
    df[col].apply(copper.transform.date_to_number)

    To modify the start date (before calling apply):
    copper.transform.start_date = datetime(2000, 1, 1)
    '''
    try:
        return (x - start_date).days
    except:
        return np.nan

# ---------------------    MACHINE LEARNING    ---------------------------------

def ml_input_labels(ds):
    ans = []
    for col in ds.filter(role=ds.INPUT, type=ds.NUMBER, ret_cols=True):
        ans.append(col)
    for col in ds.filter(role=ds.INPUT, type=ds.CATEGORY, ret_cols=True):
        if ds.type[col] == ds.CATEGORY:
            categories = list(set(ds[col]))
            categories.sort()
            for category in categories:
                new = '%s#%s' % (col, category)
                ans.append(new)
    return ans

def category2ml(series):
    ''' Converts a Series with category format to a format for machine learning
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
        n_col.name = '%s#%s' % (series.name, category)
        n_col[series == category] = 1
        ans = ans.join(n_col)
    return ans

def category2number(series):
    ''' Convert a Series with categorical information to a Series of numbers
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

def category_labels(series):
    ''' Return the labels for a Series with categorical values

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

def inputs2ml(ds):
    ''' Takes a Dataset inputs and generates a Dataframe with values ready for
    doing machine learning.
    '''
    num_options = (np.int64, np.float64)
    ans = pd.DataFrame(index=ds.index)
    numcols = ds.filter(role=ds.INPUT, type=ds.NUMBER, ret_cols=True)
    ans = ans.join(ds.frame[numcols])
    catcols = ds.filter(role=ds.INPUT, type=ds.CATEGORY, ret_cols=True)
    for catcol in catcols:
        new_cols = category2ml(ds.frame[catcol])
        ans = ans.join(new_cols)
    return ans

def target2ml(ds, which=0):
    ''' Takes a Dataset target and generates a Dataframe with values ready for
    doing machine learning.

    Return
    ------
        pandas.Series
    '''
    col = ds.filter(role=ds.TARGET, ret_cols=True)
    if col:
        col = col[which]
        if ds.type[col] == ds.CATEGORY:
            ans = category2number(ds.frame[col])
        else:
            ans = ds.frame[col]
        # ans.name = 'Target'
        return ans
    else:
        return None