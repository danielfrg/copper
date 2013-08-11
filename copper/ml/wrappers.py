# encoding: utf-8
import numpy as np
from sklearn.base import BaseEstimator

class FeatureMixer(BaseEstimator):

    def __init__(self, clfs, ignorefit=False):
        self.clfs = clfs
        self.ignorefit = ignorefit

    def fit_transform(self, X, y=None):
        if not self.ignorefit:
            self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        if not self.ignorefit:
            for clf in self.clfs:
                new = clf.fit_transform(X, y)

    def transform(self, X):
        ans = None
        for clf in self.clfs:
            new = clf.transform(X)
            if ans is None:
                ans = new
            else:
                ans = np.hstack((ans, new))
        return ans


class TransformWrapper(BaseEstimator):

    def __init__(self, clf, transformation, fit_transform=True):
        self.clf = clf
        self.fit_transform = fit_transform
        self.transformation = transformation

    def fit(self, X, y):
        if self.fit_transform:
            self.transformation.fit(X)
        _X = self._pretransform(X)
        self.clf.fit(_X, y)

    def predict(self, X):
        _X = self._pretransform(X)
        return self.clf.predict(_X)

    def predict_proba(self, X):
        _X = self._pretransform(X)
        return self.clf.predict_proba(_X)

    def transform(self, X):
        _X = self._pretransform(X)
        return self.clf.transform(_X)

    def _pretransform(self, X):
        return self.transformation.transform(X)
