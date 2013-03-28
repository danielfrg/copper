# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import grid_search
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

def bootstrap(clf_class, n_iter, ds, **args):
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

    ans = []
    bs = cross_validation.Bootstrap(len(X), n_iter=n_iter)
    for train_index, test_index in bs:
        _X = X[train_index]
        _y = y[train_index]
        clf = clf_class(**args)
        clf.fit(_X_train, _y_train)
        ans.append(clf)
    return ans
    
def grid(ds, clf, param, values, cv=None, plot=False, **args):
    random.seed(12345)
    X = copper.transform.inputs2ml(ds).values
    y = copper.transform.target2ml(ds).values
    if cv is None:
        cv = cross_validation.ShuffleSplit(len(X), **args)
    else:
        cv = cv(len(X), **args)
    parameters = {param: values}
    grid = grid_search.GridSearchCV(clf, parameters, cv=cv)
    grid.fit(X, y)

    # train_scores = np.zeros((len(values), cv.n_iter))
    # test_scores = np.zeros((len(values), cv.n_iter))
    # for i, value in enumerate(values):
    #     for j, (train, test) in enumerate(cv):
    #         parameters = clf.get_params()
    #         parameters[param] = value
    #         # print(clf.get_params)
    #         clf.set_params(param=value) # FIX!
    #         clf.fit(X[train], y[train])
    #         train_scores[i, j] = clf.score(X[train], y[train])
    #         test_scores[i, j] = clf.score(X[test], y[test])
    # return train_scores, test_scores

    if plot:
        test_scores = np.zeros((len(values), cv.n_iter))
        for i in range(cv.n_iter):
            test_scores[i, :] = grid.grid_scores_[i][2]
        for i in range(cv.n_iter):
            # plt.semilogx(gammas, train_scores[:, i], alpha=0.4, lw=2, c='b')
            plt.semilogx(values, test_scores[:, i], alpha=0.4, lw=2, c='g')
            plt.show()
            # print(test_scores)
    return grid.grid_scores_


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


