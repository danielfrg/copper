from __future__ import division
import math
import random
import copper
import numpy as np
import pandas as pd

from nose.tools import raises
from copper.tests.utils import eq_


def test_transform_int_regex():
    eq_(copper.t.to_int('0.2'), 0)
    eq_(copper.t.to_int('.8753'), 8753)
    eq_(copper.t.to_int('3.2'), 3)
    eq_(copper.t.to_int('1.2'), 1)
    eq_(copper.t.to_int('1.0'), 1)
    eq_(copper.t.to_int('NN1.0'), 1)
    eq_(copper.t.to_int('NN3.4DD'), 3)
    eq_(copper.t.to_int('(35.2)'), 35)
    eq_(copper.t.to_int('FAKE') is np.nan, True)
    eq_(copper.t.to_int('FAKE.') is np.nan, True)
    eq_(copper.t.to_int('FAKE.321'), 321)
    eq_(copper.t.to_int('FAKE.321.111'), 321)


def test_transform_float_regex():
    eq_(copper.t.to_float('0.2'), 0.2)
    eq_(copper.t.to_float('.8753'), 0.8753)
    eq_(copper.t.to_float('3.2'), 3.2)
    eq_(copper.t.to_float('1.2'), 1.2)
    eq_(copper.t.to_float('1.0'), 1.0)
    eq_(copper.t.to_float('NN1.0'), 1.0)
    eq_(copper.t.to_float('NN3.4DD'), 3.4)
    eq_(copper.t.to_float('(35.2)'), 35.2)
    eq_(copper.t.to_float('FAKE') is np.nan, True)
    eq_(copper.t.to_float('FAKE.') is np.nan, True)
    eq_(copper.t.to_float('FAKE.321'), 0.321)
    eq_(copper.t.to_float('FAKE.321.111'), 0.321)


def test_transform_int():
    array = np.arange(10)
    strings = []
    for i, item in enumerate(array):
        strings.append("STRING(%i)" % item)
    ser = pd.Series(strings)
    sol = pd.Series(array)
    eq_(ser.apply(copper.t.to_int), sol)


def test_transform_float():
    array = np.arange(10) / 10
    strings = []
    for i, item in enumerate(array):
        strings.append("STRING(%f)" % item)
    ser = pd.Series(strings)
    sol = pd.Series(array)
    eq_(ser.apply(copper.t.to_float), sol)


def test_cat_encode_simple():
    strings = np.array(['1', '2', '1', '3', '5', '2', '1', '5'])
    sol = np.array([[1., 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
                    [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0],
                    [1, 0, 0, 0], [0, 0, 0, 1]])
    eq_(copper.t.cat_encode(strings), sol)


def test_cat_encode_simple_list():
    strings = ['1', '2', '1', '3', '5', '2', '1', '5']
    sol = np.array([[1., 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
                    [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0],
                    [1, 0, 0, 0], [0, 0, 0, 1]])
    eq_(copper.t.cat_encode(strings), sol)


def test_cat_encode_big():
    abc = 'abcdefghijklmnopqrstuvwxyz'
    array = np.floor(np.random.rand(100000) * 26)
    strings = np.array([abc[int(i)] for i in array])
    ans = copper.t.cat_encode(strings)
    eq_(len(ans), 100000)
    eq_(ans.sum(axis=1), np.ones(100000))
    eq_(ans.sum(), 100000)


def test_ml_inputs_simple():
    df = pd.DataFrame(np.random.rand(8, 6))
    strings = ['1', '2', '1', '3', '5', '2', '1', '5']
    df[1] = np.array(strings)
    df[3] = np.array(strings)
    ds = copper.Dataset(df)
    ds.type[[1, 3]] = ds.CATEGORY

    ans = copper.t.ml_inputs(ds)
    eq_(ans.shape, (8, 6 - 2 + 4 * 2))
    eq_(ans[:, 0], df[0].values)
    eq_(ans[:, [1, 2, 3, 4]], copper.t.cat_encode(df[1].values))
    eq_(ans[:, 5], df[2].values)
    eq_(ans[:, [6, 7, 8, 9]], copper.t.cat_encode(df[3].values))
    eq_(ans[:, 10], df[4].values)
    eq_(ans[:, 11], df[5].values)


def test_ml_inputs_simple_with_target():
    df = pd.DataFrame(np.random.rand(8, 6))
    strings = ['1', '2', '1', '3', '5', '2', '1', '5']
    df[1] = np.array(strings)
    df[3] = np.array(strings)
    ds = copper.Dataset(df)
    ds.type[[1, 3]] = ds.CATEGORY
    ds.role[[2]] = ds.TARGET

    ans = copper.t.ml_inputs(ds)
    eq_(ans.shape, (8, 5 - 2 + 4 * 2))
    eq_(ans[:, 0], df[0].values)
    eq_(ans[:, [1, 2, 3, 4]], copper.t.cat_encode(df[1].values))
    eq_(ans[:, [5, 6, 7, 8]], copper.t.cat_encode(df[3].values))
    eq_(ans[:, 9], df[4].values)
    eq_(ans[:, 10], df[5].values)


def test_ml_inputs_simple_with_ignore():
    df = pd.DataFrame(np.random.rand(8, 6))
    strings = ['1', '2', '1', '3', '5', '2', '1', '5']
    df[1] = np.array(strings)
    df[3] = np.array(strings)
    ds = copper.Dataset(df)
    ds.type[[1, 3]] = ds.CATEGORY
    ds.role[[2]] = ds.IGNORE

    ans = copper.t.ml_inputs(ds)
    eq_(ans.shape, (8, 5 - 2 + 4 * 2))
    eq_(ans[:, 0], df[0].values)
    eq_(ans[:, [1, 2, 3, 4]], copper.t.cat_encode(df[1].values))
    eq_(ans[:, [5, 6, 7, 8]], copper.t.cat_encode(df[3].values))
    eq_(ans[:, 9], df[4].values)
    eq_(ans[:, 10], df[5].values)


def test_ml_inputs_big():
    abc = 'abcdefghijklmnopqrstuvwxyz'
    m, n = 1000, 10
    array = np.floor(np.random.rand(m) * 26)
    strings = np.array([abc[int(i)] for i in array])
    df = pd.DataFrame(np.random.rand(m, 100))
    abc_cols = np.arange(n) * 10
    for col in abc_cols:
        df[col] = strings
    ds = copper.Dataset(df)
    ds.type[abc_cols.tolist()] = ds.CATEGORY

    ans = copper.t.ml_inputs(ds)
    eq_(ans.shape, (m, 100 - n + 26 * n))
    encoded = copper.t.cat_encode(strings)
    for i, abc_col in enumerate(abc_cols):
        s = abc_col + 25 * i
        f = abc_col + 25 * i + 26
        eq_(ans[:, s:f], encoded)


@raises(Exception)
def test_ml_target_error():
    df = pd.DataFrame(np.random.rand(8, 6))
    ds = copper.Dataset(df)
    copper.t.ml_target(ds)


def test_ml_target_number():
    df = pd.DataFrame(np.random.rand(8, 6))
    ds = copper.Dataset(df)

    target_col = math.floor(random.random() * 6)
    ds.role[target_col] = ds.TARGET

    le, target = copper.t.ml_target(ds)
    eq_(target, ds[target_col].values)
    eq_(le, None)


def test_ml_target_string():
    df = pd.DataFrame(np.random.rand(6, 6))
    strings = ['z', 'h', 'z', 'c', 'h', 'c']
    sol = [2, 1, 2, 0, 1, 0]
    df['T'] = strings

    ds = copper.Dataset(df)
    ds.role['T'] = ds.TARGET

    le, target = copper.t.ml_target(ds)
    eq_(target, np.array(sol))
    eq_(le.classes_.tolist(), ['c', 'h', 'z'])


def test_ml_target_more_than_one():
    df = pd.DataFrame(np.random.rand(8, 6))
    ds = copper.Dataset(df)

    ds.role[3] = ds.TARGET
    ds.role[5] = ds.TARGET

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        le, target = copper.t.ml_target(ds)
        eq_(le, None)
        eq_(target, ds[3].values)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
