import os
import numpy as np
from copper.ml.dbn.dbn import DBN
from copper.tests.utils import eq_


def get_iris():
    from sklearn import datasets
    iris = datasets.load_iris()

    X = iris.data
    Y = iris.target
    return X, Y


def test_sklearn_api():
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


def test_reproducible():
    X, y = get_iris()

    dbn1 = DBN([5], random_state=123)
    dbn1.fit(X, y)
    pred1 = dbn1.predict(X)
    prob1 = dbn1.predict_proba(X)

    dbn2 = DBN([5], random_state=123)
    dbn2.fit(X, y)
    pred2 = dbn2.predict(X)
    prob2 = dbn2.predict_proba(X)

    eq_(dbn1.coef_, dbn2.coef_)
    eq_(pred1, pred2)
    eq_(prob1, prob2)


def test_coef_eq_layers_0():
    dbn = DBN([5], pretrain_epochs=0, finetune_epochs=0, random_state=1234)
    X, y = get_iris()
    dbn.fit(X, y)

    eq_(dbn.coef_[:5], dbn.layers[0].b)
    eq_(dbn.coef_[5:25], dbn.layers[0].W.reshape(-1))
    eq_(dbn.coef_[25:28], dbn.layers[1].b)
    eq_(dbn.coef_[28:], dbn.layers[1].W.reshape(-1))


def test_coef_eq_layers_1():
    dbn = DBN([5], pretrain_epochs=0, finetune_epochs=1, random_state=1234)
    X, y = get_iris()
    dbn.fit(X, y)

    eq_(dbn.coef_[:5], dbn.layers[0].b)
    eq_(dbn.coef_[5:25], dbn.layers[0].W.reshape(-1))
    eq_(dbn.coef_[25:28], dbn.layers[1].b)
    eq_(dbn.coef_[28:], dbn.layers[1].W.reshape(-1))


def test_coef_eq_layers_change():
    # Test that the coef_ and layers weights are connected in memory
    dbn = DBN([5], pretrain_epochs=0, finetune_epochs=0, random_state=1234)
    X, y = get_iris()
    dbn.fit(X, y)

    eq_(dbn.layers[0].b, np.zeros(5))
    eq_(dbn.layers[1].b, np.zeros(3))

    dbn.coef_[:] = np.ones(len(dbn.coef_))
    eq_(dbn.layers[0].b, np.ones(5))
    eq_(dbn.layers[0].W, np.ones((4, 5)))
    eq_(dbn.layers[1].b, np.ones(3))
    eq_(dbn.layers[1].W, np.ones((5, 3)))

    eq_(dbn.layers[0].b[0], 1)
    dbn.coef_[0] = 2
    eq_(dbn.layers[0].b[0], 2)

    eq_(dbn.coef_[1], 1)
    dbn.layers[0].b[1] = 3
    eq_(dbn.coef_[1], 3)


def test_iris_accuracy():
    dbn = DBN([25], pretrain_epochs=0, finetune_epochs=10, finetune_batch_size=10, random_state=1)
    X, y = get_iris()
    dbn.fit(X, y)

    acc = (dbn.predict(X) == y).mean()
    eq_(acc, 0.95333, 5)


def test_save_load_weights():
    import tempfile
    tempdir = tempfile.gettempdir()
    tempfile = os.path.join(tempdir, 'w.txt')

    dbn1 = DBN([5], random_state=1234)
    X, y = get_iris()
    dbn1.fit(X, y)
    pred1 = dbn1.predict(X)
    prob1 = dbn1.predict_proba(X)

    np.savetxt(tempfile, dbn1.coef_)

    dbn2 = DBN([5])
    dbn2.load_weights(tempfile, X.shape[1], len(np.unique(y)))
    pred2 = dbn2.predict(X)
    prob2 = dbn2.predict_proba(X)

    eq_(dbn1.coef_, dbn2.coef_)
    for i, layer in enumerate(dbn1.layers):
        eq_(dbn1.layers[i].W, dbn2.layers[i].W)

    eq_(pred1, pred2)
    eq_(prob1, prob2)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
