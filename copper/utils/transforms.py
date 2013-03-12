# coding=utf-8
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing


# -----------------------------------------------------------------------------
#                                  PANDAS API
# -----------------------------------------------------------------------------

numberRE = re.compile('[0-9.]+')

def to_number(x):
    try:
        return float(numberRE.search(x).group())
    except:
        return np.nan

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

def category2number(series):
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

def category_labels(series):
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

def inputs2ml(ds):
    ans = pd.DataFrame(index=ds.frame.index)

    for col in ds.filter(role=ds.INPUT, ret_cols=True):
        if ds.type[col] == ds.NUMBER and \
                          ds.frame[col].dtype in (np.int64, np.float64):
            ans = ans.join(ds.frame[col])
        elif ds.type[col] == ds.NUMBER and \
                                        ds.frame[col].dtype == object:
            ans = ans.join(ds.frame[col].apply(to_number))
        elif ds.type[col] == ds.CATEGORY and \
                        ds.frame[col].dtype in (np.int64, np.float64):
            # new_cols = category2number(ds.frame[col])
            new_cols = category2ml(ds.frame[col])
            ans = ans.join(new_cols)
        elif ds.type[col] == ds.CATEGORY and \
                                        ds.frame[col].dtype == object:
            # new_cols = category2number(ds.frame[col])
            new_cols = category2ml(ds.frame[col])
            ans = ans.join(new_cols)
        else:
            # Crazy stuff TODO: generate error
            pass
    return ans

def target2ml(ds, which=0):
    col = ds.filter(role=ds.TARGET, ret_cols=True)[which]
    if ds.type[col] == ds.CATEGORY:
        ans = category2number(ds.frame[col])
    else:
        ans = ds.frame[col]
    # ans.name = 'Target'
    return ans