# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

class DivWrapper(BaseEstimator):
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

class PCAWrapper(BaseEstimator):

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