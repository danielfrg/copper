import numpy as np
import pandas as pd
import numpy.testing as np_test
import pandas.util.testing as pd_test
from nose.tools import eq_ as nose_eq


def eq_(ans, sol, digits=0):
    if type(ans) == np.ndarray and type(sol) == np.ndarray:
        array_eq(ans, sol, digits)
    elif type(ans) == pd.Series and type(sol) == pd.Series:
        series_eq(ans, sol)
    elif type(ans) == pd.TimeSeries and type(sol) == pd.TimeSeries:
        series_eq(ans, sol, digits)
    elif type(ans) == pd.DataFrame and type(sol) == pd.DataFrame:
        frame_eq(ans, sol, digits)
    else:
        nose_eq(ans, sol, digits)


def array_eq(ans, sol, digits=0):
    if digits == 0:
        np_test.assert_array_equal(ans, sol)
    else:
        np_test.assert_array_almost_equal(ans, sol, digits)


def series_eq(ans, sol, digits=0):
    if digits == 0:
        pd_test.assert_series_equal(ans, sol, digits)
    else:
        nose_eq(ans.name, sol.name)
        np_test.assert_array_almost_equal(ans.values, sol.values, digits)


def frame_eq(ans, sol, digits=0):
    if digits == 0:
        pd_test.assert_frame_equal(ans, sol)
    else:
        nose_eq(ans.index.name, sol.index.name)
        nose_eq(ans.columns.name, sol.columns.name)
        np_test.assert_array_almost_equal(ans.values, sol.values, digits)
