# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn import cross_validation

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
    
def grid(clf_cls, ds):
    pass


if __name__ == '__main__':
    copper.project.path = '../../../data-mining/data-science-london/'
    train = copper.load('train')
    from sklearn import svm.SVC
    clf = svm.SVC()
    print(train)
    

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


