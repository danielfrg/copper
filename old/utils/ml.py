from __future__ import division
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import grid_search
from sklearn import decomposition
from sklearn import cross_validation
from sklearn.base import clone

def bootstrap(base_clf, n_iter, ds, score=False):
    '''
    Use bootstrap cross validation to create classifiers

    Parameters
    ----------
        clf_class: scikit-learn classifier
        clf_name: str - prefix for the classifiers: clf_name + "_" + itertation
        n_iter: int - number of iterations
        X_train: np.array, inputs for the training, default is self.X_train
        y_train: np.array, targets for the training, default is self.y_train
        ds: copper.Dataset, dataset for the training, default is self.train
        **args: - arguments of the classifier

    Returns
    -------
        nothing, classifiers are added to the list
    '''
    X = copper.transform.inputs2ml(ds).values
    y = copper.transform.target2ml(ds).values

    clfs = []
    scores = []
    bs = cross_validation.Bootstrap(len(X), n_iter=n_iter)
    for train_index, test_index in bs:
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        clf_params = base_clf.get_params()
        clf = clone(base_clf)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        if score:
            scores.append(clf.score(X_test, y_test))
    if score:
        return clfs, scores
    else:
        return clfs

def cv_pca(ds, clf, range_=None, cv=None, n_iter=3, ascending=False):
    if cv is None:
        cv = cross_validation.ShuffleSplit(len(ds), n_iter=n_iter)
    if range_ is None:
        range_ = range(1, len(ds.inputs.columns))
    
    ans = pd.Series(index=range_)
    y = copper.transform.target2ml(ds).values
    for i in range_:
        pca = ds.PCA(n_components=i)
        X = pca.inputs
        scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
        ans[i] = np.mean(scores)
    return ans.order(ascending=ascending)

def grid(ds, base_clf, param, values, cv=None, verbose=False, **args):
    if cv is None:
        cv = cross_validation.ShuffleSplit(len(ds), **args)
    
    X = copper.transform.inputs2ml(ds).values
    y = copper.transform.target2ml(ds).values

    train_scores = np.zeros((len(values), cv.n_iter))
    test_scores = np.zeros((len(values), cv.n_iter))
    for i, value in enumerate(values):
        if verbose:
            print('%s= %s' % (param, str(value)))
        for j, (train, test) in enumerate(cv):
            clf_params = base_clf.get_params()
            clf_params[param] = value
            clf = clone(base_clf)
            clf.set_params(**clf_params)
            clf.fit(X[train], y[train])
            train_scores[i, j] = clf.score(X[train], y[train])
            test_scores[i, j] = clf.score(X[test], y[test])

    return train_scores, test_scores


def rmsle(y_test, y_pred):
    ans = np.log1p(y_pred) - np.log1p(y_test)
    ans = np.power(ans, 2)
    ans = ans.mean()
    return np.sqrt(ans)