from __future__ import division
import copper
import pandas as pd
from nose.tools import raises
from copper.tests.utils import eq_


def get_iris():
    from sklearn import datasets
    iris = datasets.load_iris()

    X = iris.data
    Y = iris.target
    return X, Y


def get_iris_ds():
    X, Y = get_iris()
    df = pd.DataFrame(X)
    df['Target'] = pd.Series(Y, name='Target')

    ds = copper.Dataset(df)
    ds.role['Target'] = ds.TARGET
    return ds


def get_mc():
    ds = get_iris_ds()
    mc = copper.ModelComparison()
    mc.train_test_split(ds, random_state=0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    mc['LR'] = LogisticRegression()
    mc['SVM'] = SVC(probability=True)
    mc.fit()
    return mc

# -----------------------------------------------------------------------------


def test_accuracy_score(mc=None):
    mc = get_mc() if mc is None else mc
    # print mc.accuracy_score()
    score = mc.accuracy_score()
    eq_(score['SVM'], 0.973684, 6)
    eq_(score['LR'], 0.868421, 6)


@raises(ValueError)  # Not binary classification
def test_auc_score(mc=None):
    mc = get_mc() if mc is None else mc
    mc.auc_score()


@raises(ValueError)  # Not binary classification
def test_average_precision_score(mc=None):
    mc = get_mc() if mc is None else mc
    mc.average_precision_score()


def test_f1_score(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.f1_score()
    eq_(score['SVM'], 0.973952, 6)
    eq_(score['LR'], 0.870540, 6)


def test_fbeta_score(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.fbeta_score(beta=0.1)
    eq_(score['SVM'], 0.976249, 6)
    eq_(score['LR'], 0.914067, 6)
    score = mc.fbeta_score()
    eq_(score['SVM'], 0.973952, 6)
    eq_(score['LR'], 0.870540, 6)


def test_hinge_loss(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.hinge_loss()
    eq_(score['SVM'], 0.342105, 6)
    eq_(score['LR'], 0.342105, 6)


def test_matthews_corrcoef(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.matthews_corrcoef()
    eq_(score['SVM'], 0.978391, 6)
    eq_(score['LR'], 0.916242, 6)


def test_precision_score(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.precision_score()
    eq_(score['SVM'], 0.976316, 6)
    eq_(score['LR'], 0.915414, 6)


def test_recall_score(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.recall_score()
    eq_(score['SVM'], 0.973684, 6)
    eq_(score['LR'], 0.868421, 6)


def test_zero_one_loss(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.zero_one_loss()
    eq_(score['SVM'], 0.026316, 6)
    eq_(score['LR'], 0.131579, 6)


def test_recall_score_average_none(mc=None):
    mc = get_mc() if mc is None else mc
    score = mc.recall_score(average=None)
    eq_(score['LR (2)'], 1, 6)
    eq_(score['LR (0)'], 1, 6)
    eq_(score['SVM (2)'], 1, 6)
    eq_(score['SVM (0)'], 1, 6)
    eq_(score['SVM (1)'], 0.9375, 4)
    eq_(score['LR (1)'], 0.6875, 4)


# -----------------------------------------------------------------------------
#                        With target as string

def get_mc_string():
    ds = get_iris_ds()
    ds.type['Target'] = ds.CATEGORY
    ds['Target'] = ds['Target'].apply(lambda x: str(x))
    ds['Target'][ds['Target'] == '0'] = 'Iris-A'
    ds['Target'][ds['Target'] == '1'] = 'Iris-B'
    ds['Target'][ds['Target'] == '2'] = 'Iris-C'
    eq_(ds.metadata['dtype']['Target'], object)

    mc = copper.ModelComparison()
    mc.train_test_split(ds, random_state=0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    mc['LR'] = LogisticRegression()
    mc['SVM'] = SVC(probability=True)
    mc.fit()
    return mc


def test_repeat_tests_with_target_string():
    mc = get_mc_string()
    test_accuracy_score(mc)
    test_f1_score(mc)
    test_fbeta_score(mc)
    test_hinge_loss(mc)
    test_matthews_corrcoef(mc)
    test_precision_score(mc)
    test_recall_score(mc)
    test_zero_one_loss(mc)
    test_recall_score_average_none(mc)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
