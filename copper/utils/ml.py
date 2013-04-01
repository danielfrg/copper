# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import grid_search
from sklearn import decomposition
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.base import clone, BaseEstimator

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

def cv_pca(ds, clf, range_=None, cv=None, n_iter=3):
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
    return ans

def grid(ds, base_clf, param, values, cv=None, **args):
    if cv is None:
        cv = cross_validation.ShuffleSplit(len(ds), **args)
    
    X = copper.transform.inputs2ml(ds).values
    y = copper.transform.target2ml(ds).values

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


# --------------------------------------------------------------------------------------------
#                                        ENSEMBLE/BAGS
# --------------------------------------------------------------------------------------------

class Bag(BaseEstimator):
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

    def fit(self, X, y):
        pass

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class AverageBag(Bag):
    '''
    Implementation idea:
    for each classifier 
        for each option e.g.: (0,1,2)
            sum each predicted probability
    divide by the number of clfs
    '''

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if type(X) is copper.Dataset:
            X = copper.transform.inputs2ml(X).values
        options = len(self.clfs[0].predict_proba(X[:1])[0])
        ans = np.zeros((len(X), options))
        for clf in self.clfs:
            probas = clf.predict_proba(X)
            for option in range(options):
                ans[:, option] = ans[:, option] + probas[:,option]

        ans = ans / len(self.clfs)
        return ans

class MaxProbaBag(Bag):
    '''
    Create a big array with all the predicted probabilities for 
    each classifier.

    Calculate the max of each row and find the classifier that
    has that probability

    Iterate over the big array and slice each row to create the ans
    ''' 
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if type(X) is copper.Dataset:
            X = copper.transform.inputs2ml(X).values
        options = len(self.clfs[0].predict_proba(X[:1])[0])
        temp = np.zeros((len(X), options*len(self.clfs)))
        for i, clf in enumerate(self.clfs):
            probas = clf.predict_proba(X) 
            temp[:, i*options:i*options+2] = probas

        ans = np.zeros((len(X), options))
        max_pos = np.argmax(temp, axis=1)
        max_clf = np.floor(max_pos / options)
        for i, row in enumerate(temp):
            iclf = max_clf[i]
            ans[i, ] = temp[i, iclf * options:iclf * options + 2]
        return ans

class PCA_wrapper(BaseEstimator):

    def __init__(self, base_clf, n_components=None):
        self.base_clf = clone(base_clf)
        self.n_components = n_components
        self.pca_model = None

    def fit(self, X, y):
        self.pca_model = decomposition.PCA(n_components=self.n_components)
        self.pca_model.fit(X)
        _X = self.pca_model.transform(X)
        self.base_clf.fit(_X, y)
    
    def score(self, X, y):
        _X = self.pca_model.transform(X)
        y_pred = self.base_clf.predict(_X)
        return accuracy_score(y, y_pred)

    def predict(self, X):
        _X = self.pca_model.transform(X)
        return self.base_clf.predict(_X)

    def predict_proba(self, X):
        _X = self.pca_model.transform(X)
        return self.base_clf.predict_proba(_X)

if __name__ == '__main__':
    from sklearn import svm
    import pprint
    import random
    
    copper.project.path = '../../../data-mining/data-science-london/'
    train = copper.load('train')
    test = copper.load('test')
    clf = svm.SVC(kernel='rbf', gamma=0.02, C=10, probability=True)
    pca_clf = PCA_wrapper(clf, n_components=13)
    ml = copper.MachineLearning()
    ml.train = train
    ml.add_clf(clf, 'svm')
    ml.add_clf(pca_clf, 'pca')
    ml.fit()
    bag = MaxProbaBag()
    bag.add_clf(ml.clfs)
    # print(ml.predict_proba(test).head(3))
    print(bag.predict_proba(test))
    
