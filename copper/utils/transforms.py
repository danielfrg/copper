import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

intRE = re.compile('[-+]?[0-9]+')
floatRE = re.compile('[-+]?[0-9]*\.?[0-9]+')


def to_int(x):
    match = intRE.search(x)
    return np.nan if match is None else int(match.group())


def to_float(x):
    match = floatRE.search(x)
    return np.nan if match is None else float(match.group())


def cat_encode(strings):
    labels = list(set(strings))
    labels.sort()
    ans = np.zeros((len(strings), len(labels)))
    for i, label in enumerate(labels):
        ans[:, i][strings == label] = 1
    return ans


def ml_inputs(dataset):
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
