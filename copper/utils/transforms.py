import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

intRE = re.compile('[-+]?[0-9]+')
floatRE = re.compile('[-+]?[0-9]*\.?[0-9]+')


def to_int(x):
    """ Convert from string to int. Usefull when used inside pandas.apply

    Regular expression: ``[-+]?[0-9]+``

    Examples
    --------
    >>> data = pd.Series(['1', 'a2', '(-3)', '___4___', '$31', '15%'])
    >>> data
        0          1
        1         a2
        2       (-3)
        3    ___4___
        4        $31
        5        15%
dtype: object
    >>> data.apply(copper.t.to_int)
        0     1
        1     2
        2    -3
        3     4
        4    31
        5    15
        dtype: int64
    """

    match = intRE.search(x)
    return np.nan if match is None else int(match.group())


def to_float(x):
    """ Convert from string to floats. Usefull when used inside pandas.apply

    Regular expression: ``[-+]?[0-9]*\.?[0-9]+``

    Examples
    --------
    >>> df = pd.Series(['1', 'a2', '(-3.5)', '___4.00001______', '$31.312', '15%'])
    >>> df
        0                       1
        1                      a2
        2                  (-3.5)
        3    ___4.00001___
        4                 $31.312
        5                     15%
    >>> df.apply(copper.t.to_int)
        0     1.00000
        1     2.00000
        2    -3.50000
        3     4.00001
        4    31.31200
        5    15.00000
        dtype: float64
    """
    match = floatRE.search(x)
    return np.nan if match is None else float(match.group())


def cat_encode(values):
    """ Encodes a category into multiple columns ready for machine learning

    Parameters
    ----------
    values: np.array or list

    Returns
    -------
    np.array

    Examples
    --------
    >>> cat_encode(np.array(['z', 'a', 'h', 'z', 'h']))
        [[ 0.  0.  1.]
        [ 1.  0.  0.]
        [ 0.  1.  0.]
        [ 0.  0.  1.]
        [ 0.  1.  0.]]
    """
    if type(values) is list:
        values = np.array(values)
    labels = list(set(values))
    labels.sort()
    ans = np.zeros((len(values), len(labels)))
    for i, label in enumerate(labels):
        ans[:, i][values == label] = 1
    return ans


def ml_inputs(dataset):
    """ Takes a dataset and retuns the inputs in a numpy.array ready for
    machine learning.
    Mainly transforms non-numerical variables(columns) to numbers.

    Parameters
    ----------
    copper.Dataset

    Returns
    -------
    np.array
    """
    columns = dataset.filter_cols(role=dataset.INPUT)
    assert len(columns) > 0, 'No input variables on Dataset'

    ans = np.zeros((len(dataset), 1))
    for column in columns:
        if dataset.type[column] == dataset.NUMBER:
            ans = np.hstack((ans, dataset[[column]].values))
        elif dataset.type[column] == dataset.CATEGORY:
            ans = np.hstack((ans, cat_encode(dataset[column])))
    ans = np.delete(ans, 0, axis=1)
    return ans


def ml_target(dataset):
    """ Takes a dataset and retuns the target in a numpy.array ready for
    machine learning.
    Mainly transforms non-numerical variables(columns) to numbers.

    Parameters
    ----------
    copper.Dataset

    Returns
    -------
    np.array

    Notes
    -----
    If dataset has more than one variable with role=TARGET then the first one
    is selected.
    """
    cols = dataset.filter_cols(role=dataset.TARGET)
    assert len(cols) > 0, 'No target variables on Dataset'
    if len(cols) > 1:
        import warnings
        warnings.warn("Dataset contains more than one target, %s was choosed" % cols[0])

    if dataset[cols[0]].dtype in (np.int, np.float):
        return dataset[cols[0]].values
    else:
        return LabelEncoder().fit_transform(dataset[cols[0]].values)


# TESTS
import copper
import pandas as pd
from copper.tests.utils import eq_
# from copper.utils import transforms
from nose.tools import raises



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
