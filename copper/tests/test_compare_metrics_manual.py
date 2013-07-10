from __future__ import division
import math
import random
import copper
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

from nose.tools import raises, ok_
from copper.tests.utils import eq_


def get_train():
    X = np.ones((12, 3))
    y = ['b', 'z', 'b', 'g', 'g', 'z', 'b', 'z', 'g', 'b', 'g', 'z']
    #   [ 0,   2,   0,   1,   1,   2,   0,   2,   1,   0,   1,   0]
    df = pd.DataFrame(X)
    df['target'] = y
    ds = copper.Dataset(df)
    ds.role['target'] = ds.TARGET
    return ds

class BaseFake():
    def fit(self, X, t):
        pass

    def predict(self, X):
        return np.array(self.encode)

class FakePerfect(BaseFake):
    labels = ['b', 'z', 'b', 'g', 'g', 'z', 'b', 'z', 'g', 'b', 'g', 'z']
    encode = [ 0,   2,   0,   1,   1,   2,   0,   2,   1,   0,   1,   2]

class Fake1(BaseFake):
    labels = ['g', 'z', 'b', 'z', 'g', 'z', 'b', 'b', 'g', 'b', 'g', 'z']
    encode = [ 1,   2,   0,   2,   1,   2,   0,   0,   1,   0,   1,   2]

class Fake2(BaseFake):
    labels = ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'b', 'g', 'g', 'g']
    encode = [ 1,   1,   1,   1,   1,   1,   1,   1,   0,   1,   1,   1]

def get_mc():
    mc = copper.ModelComparison()
    mc.train = get_train()
    mc.test = get_train()

    mc['perf'] = FakePerfect()
    mc['f1'] = Fake1()
    mc['f2'] = Fake2()
    mc.fit()
    return mc


# -----------------------------------------------------------------------------

def test_classes():
    mc = get_mc()
    eq_(mc.le.classes_, np.array(['b', 'g', 'z']))

def test_target_values():
    mc = get_mc()
    eq_(mc.y_train, np.array([0, 2, 0, 1, 1, 2, 0, 2, 1, 0, 1, 2]))

def test_accuracy():
    mc = get_mc()
    scores = mc.accuracy_score()
    eq_(scores['perf'], 1.0)
    eq_(scores['f1'], 0.75)
    eq_(scores['f2'], 0.25)

def test_f1_score_avg_none():
    mc = get_mc()
    scores = mc.f1_score(average=None)
    eq_(scores['perf (b)'], 1.0)
    eq_(scores['perf (g)'], 1.0)
    eq_(scores['perf (z)'], 1.0)
    eq_(scores['f1 (b)'], 0.75)
    eq_(scores['f1 (g)'], 0.75)
    eq_(scores['f1 (z)'], 0.75)
    eq_(scores['f2 (b)'], 0)
    eq_(scores['f2 (g)'], 0.4, 1)
    eq_(scores['f2 (z)'], 0)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
