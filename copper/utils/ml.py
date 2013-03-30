# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import grid_search
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
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
        clf.set_params(**clf_params)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        if score:
            scores.append(clf.score(X_test, y_test))
    if score:
        return clfs, scores
    else:
        return clfs

def cv_pca(ds, clf, cv=None, **args):
    X, y = ds.PCA(2, ret_array=True)
    if cv is None:
        cv = cross_validation.ShuffleSplit(len(ds), **args)

    scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
    return np.mean(scores)

def grid(ds, base_clf, param, values, cv=None, **args):
    X = copper.transform.inputs2ml(ds).values
    y = copper.transform.target2ml(ds).values
    if cv is None:
        cv = cross_validation.ShuffleSplit(len(ds), **args)

    train_scores = np.zeros((len(values), cv.n_iter))
    test_scores = np.zeros((len(values), cv.n_iter))
    for i, value in enumerate(values):
        for j, (train, test) in enumerate(cv):
            clf_params = base_clf.get_params()
            clf_params[param] = value
            clf = clone(base_clf)
            clf.set_params(**clf_params)
            clf.fit(X[train], y[train])
            train_scores[i, j] = clf.score(X[train], y[train])
            test_scores[i, j] = clf.score(X[test], y[test])

    return train_scores, test_scores


if __name__ == '__main__':
    from sklearn import svm
    import pprint
    import random
    
    copper.project.path = '../../../data-mining/data-science-london/'
    train = copper.load('train')
    clf = svm.SVC()
    ans = grid(train, clf, 'C', [0.1, 1, 10, 100], plot=True, n_iter=3, random_state=123)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ans)
    # copper.plot.show()
    # print(ans.grid_scores_)
    # clf = svm.SVC()
    # print(train)


# --------------------------------------------------------------------------------------------
#                                        ENSEMBLE/BAGS
# --------------------------------------------------------------------------------------------

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

class AverageBag(Ensemble):
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
        return np.argmax(self.predict_proba(X_test), axis=1)

    def predict_proba(self, X_test):
        options = np.shape(self.clfs[0].predict_proba(X_test[:1]))[1]
        
        ans = np.zeros((len(X_test), options))
        for clf in self.clfs:
            probas = clf.predict_proba(X_test)
            for option in range(options):
                ans[:, option] = ans[:, option] + probas[:,option]

        ans = ans / len(self.clfs)
        return ans


class MaxProbaBag(Ensemble):
    pass


