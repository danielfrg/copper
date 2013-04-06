# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd

from sklearn import decomposition
from sklearn.metrics import accuracy_score
from sklearn.base import clone, BaseEstimator

class Bag(BaseEstimator):
    def __init__(self, clfs=None):
        self.clfs = []
        if clfs is not None:
            self.add_clf(clfs)

    def add_clf(self, new):
        if type(new) is pd.Series:
            self.add_clf(new.values.tolist())
        elif type(new) is list:
            for clf in new:
                self.add_clf(clf)
        else:            
            self.clfs.append(new)

    def fit(self, X, y):
        for clf in self.clfs:
            clf.fit(X, y)

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
        num_options = len(self.clfs[0].predict_proba(X[:1])[0])
        ans = np.zeros((len(X), num_options))
        for clf in self.clfs:
            probas = clf.predict_proba(X)
            for i in range(num_options):
                ans[:, i] = ans[:, i] + probas[:,i]
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
        num_options = len(self.clfs[0].predict_proba(X[:1])[0])
       
        groups = []
        for i, clf in enumerate(self.clfs):
            groups.append((i*num_options, i*num_options+num_options))
        # print(groups)
       
        predictions = np.zeros((len(X), num_options*len(self.clfs)))
        predictions[:] = np.nan
        for i, clf in enumerate(self.clfs):
            probas = clf.predict_proba(X) 
            predictions[:, groups[i][0]:groups[i][1]] = probas
            # break
        # print(predictions)

        ans = np.zeros((len(X), num_options))
        max_pos = np.argmax(predictions, axis=1)
        max_clf = np.floor(max_pos / num_options)
        for i, row in enumerate(predictions):
            g = int(max_clf[i])
            ans[i, ] = predictions[i, groups[g][0]:groups[g][1]]
        # print(ans)
        return ans

class PriorityBag(Bag):
    def __init__(self, range=0.1, clfs=None):
        self.range = range
        super().__init__(clfs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        num_options = len(self.clfs[0].predict_proba(X[:1])[0])

        # create a list of the group columns
        groups = []
        for i, option in enumerate(self.clfs):
            groups.append((i*num_options, i*num_options+num_options))
        # print(groups)

        # create a huge matrix with all the predictions
        num_cols = num_options * len(self.clfs)
        predictions = np.zeros((len(X), num_cols))
        predictions[:] = np.nan
        for i, clf in enumerate(self.clfs):
            predictions[:, groups[i][0]:groups[i][1]] = clf.predict_proba(X)
            # break
        # print(predictions)

        ans = np.zeros((len(X), num_options))
        for i, row in enumerate(predictions):
            for group in groups:
                g = row[group[0]:group[1]]
                m = np.max(g)
                if m > 0.5 + self.range:
                    ans[i, :] = g
                    break
                elif group == groups[-1]:
                    g = row[group[0]:group[1]]
                    ans[i, :] = row[groups[0][0]:groups[0][1]]
                    # ans[i, :] = -1
        # print(ans)
        return ans

class SplitWrapper(BaseEstimator):
    def __init__(self, base_clf, ds_labels, variable, models={}):
        self.base_clf = base_clf
        self.variable = variable
        self.remove_index = []
        self.models = models

        if type(ds_labels) is copper.Dataset:
            ds_labels = copper.transform.ml_input_labels(ds_labels)
        self.ds_labels = ds_labels
        # print(self.ds_labels)
        self.var_options = [col.split('#')[1] for col in ds_labels 
                                            if col.startswith(variable+'#')]
        self.var_indexes = [i for i, col in enumerate(ds_labels) 
                                            if col.startswith(variable+'#')]
        # print(self.var_options)
        # print(self.var_indexes)

    def fit(self, X, y):
        self.models = {}
        # print(self.variable, X.shape)
        for option, index in zip(self.var_options, self.var_indexes):
            var_cols = X[:, self.var_indexes]
            col_with_1 = np.argmax(var_cols, axis=1)
            col_with_1 = col_with_1 + self.var_indexes[0]
            X_f = X[col_with_1 == index, :]
            X_f = np.delete(X_f, self.var_indexes, 1)
            y_f = y[col_with_1 == index, :]
            # print(self.variable, X_f.shape, y_f.shape)
            clf = clone(self.base_clf)
            clf.fit(X_f, y_f)
            # print(option, index)
            # print(self.variable, X_f.shape, clf) ##
            self.models[index] = clf
            # return

    def predict(self, X):
        X_f = np.delete(X, self.var_indexes, 1)

        first_index = self.var_indexes[0]
        predictions = np.zeros((len(X), len(self.var_options)))
        predictions[:] = np.nan
        for option, index in zip(self.var_options, self.var_indexes):
            clf = self.models[index]
            predictions[:, index - first_index] = clf.predict(X_f)
            # break
        # print(predictions)

        # print(X.shape)
        ans = np.zeros((len(X)))
        ans[:] = np.nan
        for option, index in zip(self.var_options, self.var_indexes):
            var_cols = X[:, self.var_indexes]
            col_with_1 = np.argmax(var_cols, axis=1)
            col_with_1 = col_with_1 + self.var_indexes[0]
            # print(ans[col_with_1 == index].shape)
            ans[col_with_1 == index] = predictions[col_with_1 == index, index - first_index]
            # break
        # print(ans)
        return ans
        # return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict_proba(self, X):
        X_f = np.delete(X, self.var_indexes, 1)

        # create a list of the group columns
        any_key = list(self.models.keys())[0]
        any_clf = self.models[any_key]
        num_options = len(any_clf.predict_proba(X_f[:1])[0])
        groups = []
        for i, option in enumerate(self.var_options):
            groups.append((i*num_options, i*num_options+num_options))
        # print(groups)

        # create a huge matrix with all the predictions
        num_cols = num_options * len(self.models)
        predictions = np.zeros((len(X), num_cols))
        predictions[:] = np.nan
        for i, index in enumerate(self.var_indexes):
            clf = self.models[index]
            # print(clf.predict_proba(X_f).shape)
            predictions[:, groups[i][0]:groups[i][1]] = clf.predict_proba(X_f)
            # break
        # print(predictions)

        num_cols = len(self.models)
        ans = np.zeros((len(X), num_options))
        ans[:] = np.nan
        for i, index in enumerate(self.var_indexes):
            var_cols = X[:, self.var_indexes]
            col_with_1 = np.argmax(var_cols, axis=1)
            col_with_1 = col_with_1 + self.var_indexes[0]
            # print(ans[col_with_1 == index].shape)
            ans[col_with_1 == index] = predictions[col_with_1 == index, groups[i][0]:groups[i][1]]
            # break
        # print(ans)
        return ans

class PCAWrapper(BaseEstimator):
    def __init__(self, base_clf, n_components=None, **args):
        self.base_clf = base_clf
        self.n_components = n_components
        self.pca_model = None

    def fit(self, X, y):
        self.pca_model = decomposition.PCA(n_components=self.n_components)
        self.pca_model.fit(X)
        self.estimator = clone(self.base_clf)
        _X = self.pca_model.transform(X)
        self.estimator.fit(_X, y)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict(self, X):
        _X = self.pca_model.transform(X)
        return self.estimator.predict(_X)

    def predict_proba(self, X):
        _X = self.pca_model.transform(X)
        return self.base_clf.predict_proba(_X)

if __name__ == '__main__':
    copper.project.path = '../../../data-mining/titanic/'
    train = copper.load('train')
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=1)
    clf2 = RandomForestClassifier(max_depth=2)
    sw1 = copper.SplitWrapper(clf, train, 'sex')
    sw2 = copper.SplitWrapper(clf2, train, 'embarked')
    
    mc = copper.ModelComparison()
    mc.train = train
    mc.add_clf(clf, 'rf')
    mc.add_clf(sw1, 'sw1')
    # mc.add_clf(sw2, 'sw2')
    # bag = AverageBag(clfs=mc.clfs)
    bag = PriorityBag(clfs=mc.clfs)
    mc.add_clf(bag, 'bag')
    mc.fit()
    # scores = mc.cv_accuracy(cv=20)
    print(scores)
    clone(bag)

    # from sklearn.cross_validation import check_cv
    # print(check_cv(3, mc.X_train))












