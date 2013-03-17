# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn import cross_validation

class Ensemble(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def score(self, X_test, y_test):
        raise NotImplementedError("Should have implemented this")

    def predict(self, X_test):
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, X_test):
        raise NotImplementedError("Should have implemented this")

class Bagging(Ensemble):
    def __init__(self, clfs=None):
        if type(clfs) is pd.Series:
            # Comes from ml.clfs
            self.clfs = clfs.values
        elif type(clfs) is list:
            self.clfs = clfs
        else:
            self.clfs = []

    def add_clf(self, new):
        if type(new) is pd.Series:
            # Comes from ml.clfs
            self.add_clf(new.values.tolist())
        elif type(new) is list:
            for clf in new:
                self.add_clf(clf)
        else:            
            self.clfs.append(new)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def predict(self, X_test):
        # TODO: optimize this cuz is very low :P
        prediction = np.zeros(len(X_test))
        temp = np.zeros((len(X_test), len(self.clfs)))
        for i, clf in enumerate(self.clfs):
            temp[:, i] = clf.predict(X_test).T
        for i, row in enumerate(temp):
            row = row.tolist()
            prediction[i] = max(set(row), key=row.count)
        return prediction

    def predict_proba(self, X_test):
        temp = np.zeros((len(X_test), len(self.clfs)))
        for i, clf in enumerate(self.clfs):
            temp[:, i] = clf.predict_proba(X_test)[:, 0]
        probas = np.zeros((len(X_test), 2))
        probas[:,0] = np.mean(temp, axis=1)
        probas[:,1] = 1 - probas[:,0]
        return probas

def bootstrap(clf_class, n, ds, **args):
    '''
    Use bootstrap cross validation to create classifiers

    Parameters
    ----------
        clf_class: scikit-learn classifier
        clf_name: str - prefix for the classifiers: clf_name + "_" + itertation
        n: int - number of iterations
        X_train: np.array, inputs for the training, default is self.X_train
        y_train: np.array, targets for the training, default is self.y_train
        ds: copper.Dataset, dataset for the training, default is self.train
        **args: - arguments of the classifier

    Returns
    -------
        nothing, classifiers are added to the list
    '''
    if ds is not None:
        X_train = copper.transform.inputs2ml(ds).values
        y_train = copper.transform.target2ml(ds).values

    ans = []
    bs = cross_validation.Bootstrap(len(X_train), n_iter=n)
    for train_index, test_index in bs:
        _X_train = X_train[train_index]
        _y_train = y_train[train_index]
        clf = clf_class(**args)
        clf.fit(_X_train, _y_train)
        ans.append(clf)
    return ans
    