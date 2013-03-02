# coding=utf-8
from __future__ import division
import os
import copper
import numpy as np
import pandas as pd

try :
    import pandas.rpy.common as com
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr
except:
    pass


def impute(dataframe, col, method='knn'):
    ''' Imputes data using R imputation package
    This requires the R package imputation to be installed:
        R: install.packages("imputation")

    file shouldbe on the project/data directory

    Parameters
    ----------
        filename on the data folder of the project
        method: str, method to use: knn only for now

    Returns
    -------
        pandas.Dataframe with the imputed data
    '''
    if method == 'knn':
        return _imputeKNN(filename)

def imputeKNN(dataframe):
    filename = 'impute.csv'
    filepath = os.path.join('/tmp/', filename)
    dataframe.to_csv(filepath)

    importr("imputation")
    r('data = read.csv("%s")' % filepath)
    r('data.imputed = kNNImpute(data, 2, verbose=F)')
    r('data.imputed = data.imputed[[1]]')
    # print com.load_data('data.imputed') # NOTE: pandas api not working
    r('write.csv(data.imputed, "%s")' % filepath)

    return pd.read_csv(filepath)

