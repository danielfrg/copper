from __future__ import division

from copper.ml.dbn.dbn import DBN


def get_iris():
    from sklearn import datasets
    iris = datasets.load_iris()

    X = iris.data
    Y = iris.target
    return X, Y


def test_basic():
    ''' sklearn API: not functionality
    '''
    dbn = DBN([5])
    X, y = get_iris()
    dbn.fit(X, y)
    dbn.predict_proba(X)
    dbn.predict(X)


def test_instance():
    from sklearn.base import BaseEstimator
    dbn = DBN([5])
    assert isinstance(dbn, BaseEstimator)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
