# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from copper.core.ensemble import *

class MachineLearning():
    '''
    Wrapper around scikit-learn and pandas to make machine learning faster and easier
    Utilities for model selection.
    '''

    def __init__(self):
        self.dataset = None
        self._clfs = {}
        self._ensembled = {}
        self.costs = [[1,0],[0,1]]
        self.X_train = None
        self.y_train = None
        self.X_test  = None
        self.y_test = None

    # --------------------------------------------------------------------------
    #                               PROPERTIES
    # --------------------------------------------------------------------------

    def set_train(self, ds):
        '''
        Uses a Dataset to set the values of inputs and targets for training
        In general sets self.X_train and self.y_train
        '''
        self.X_train = copper.transform.inputs2ml(ds).values
        self.y_train = copper.transform.target2ml(ds).values

    def set_test(self, ds):
        '''
        Uses a Dataset to set the values of inputs and targets for testing
        In general sets self.X_test and self.y_test
        '''
        self.X_test = copper.transform.inputs2ml(ds).values
        self.y_test = copper.transform.target2ml(ds).values

    train = property(None, set_train)
    test = property(None, set_test)

    def add_clf(self, clf, name):
        '''
        Adds a new classifier
        '''
        self._clfs[name] = clf

    def rm_clf(self, name):
        '''
        Removes a classifier
        '''
        try:
            del self._clfs[name]
        except:
            del self._ensembled[name]

    def clear_clfs(self):
        '''
        Removes all classifiers
        '''
        self._clfs = {}

    def list_clfs(self):
        '''
        Generates a Series with all the classifiers

        Returns
        -------
            pandas.Series
        '''
        clfs = list(self._clfs.keys())
        clfs = clfs + list(self._ensembled.keys())
        values = list(self._clfs.values())
        values = values + list(self._ensembled.values())
        return pd.Series(values, index=clfs)

    clfs = property(list_clfs, None)

    # --------------------------------------------------------------------------
    #                            Scikit-learn API
    # --------------------------------------------------------------------------

    def fit(self, clfs=None, X_train=None, y_train=None, ds=None):
        '''
        Fits the classifiers

        Parameters
        ----------
            clfs: list, of classifiers to fit, default all
            X_train: np.array, inputs for the training, default is self.X_train
            y_train: np.array, targets for the training, default is self.y_train
            ds: copper.Dataset, dataset fot the training, default is self.train

        Returns
        -------
            None
        '''
        if clfs is None:
            clfs = self._clfs.keys()
        if X_train is None and y_train is None:
            X_train = self.X_train
            y_train = self.y_train
        if ds is not None:
            X_train = ds.inputs.values
            y_train = ds.target.values

        for clf_name in clfs:
            self._clfs[clf_name].fit(X_train, y_train)

    def predict(self, clfs=None, X_test=None, ds=None):
        '''
        Make the classifiers predict the testing inputs

        Parameters
        ----------
            clfs: list, of classifiers to make prediction, default all
            X_test: np.array, inputs for the prediction, default is self.X_test
            ds: copper.Dataset, dataset fot the prediction, default is self.test

        Returns
        -------
            pandas.DataFrame with the predictions
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None:
            X_test = self.X_test
        if ds is not None:
            X_test = ds.inputs.values

        ans = pd.DataFrame(columns=clfs, index=range(len(X_test)))
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            scores = clf.predict(X_test)
            ans[clf_name][:] = pd.Series(scores)
        return ans

    def predict_proba(self, clfs=None, X_test=None, ds=None):
        '''
        Make the classifiers predict probabilities of inputs
        Parameters
        ----------
            clfs: list, of classifiers to make prediction, default all
            X_test: np.array, inputs for the prediction, default is self.X_test
            ds: copper.Dataset, dataset fot the prediction, default is self.test

        Returns
        -------
            pandas.DataFrame with the predicted probabilities
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None:
            X_test = self.X_test
        if ds is not None:
            X_test = ds.inputs.values

        ans = pd.DataFrame(columns=clfs, index=range(len(X_test)))
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            scores = clf.predict_proba(X_test)[:,0]
            ans[clf_name][:] = pd.Series(scores)
        return ans

    def accuracy(self, clfs=None, X_test=None, y_test=None, ds=None, ascending=False):
        '''
        Calculates the accuracy of inputs

        Parameters
        ----------
            clfs: list, of classifiers to calculate the accuracy, default all
            X_test: np.array, inputs for the prediction, default is self.X_test
            y_test: np.array, targets for the prediction, default is self.y_test
            ds: copper.Dataset, dataset for the prediction, default is self.test
            ascending: boolean, sort the Series on this direction

        Returns
        -------
            pandas.Series with the accuracy
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        if ds is not None:
            X_test = ds.inputs.values
            y_test = ds.target.values

        ans = pd.Series(index=clfs, name='Accuracy')
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            ans[clf_name] = clf.score(X_test, y_test)
        return ans.order(ascending=ascending)

    def auc(self, clfs=None, X_test=None, y_test=None, ds=None, ascending=False):
        '''
        Calculates the Area Under the ROC Curve

        Parameters
        ----------
            clfs: list, of classifiers to calculate the AUC, default all
            X_test: np.array, inputs for the prediction, default is self.X_test
            y_test: np.array, targets for the prediction, default is self.y_test
            ds: copper.Dataset, dataset for the prediction, default is self.test
            ascending: boolean, sort the Series on this direction

        Returns
        -------
            pandas.Series with the accuracy
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        if ds is not None:
            X_test = ds.inputs.values
            y_test = ds.target.values

        ans = pd.Series(index=clfs, name='Area Under the Curve')
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            probas_ = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            ans[clf_name] = auc(fpr, tpr)
        return ans.order(ascending=ascending)

    def roc(self, clfs=None,  X_test=None, y_test=None, ds=None,
                                 ascending=False, legend=True, retList=False):
        '''
        Plots the ROC chart

        Parameters
        ----------
            clfs: list, of classifiers to plot the ROC, default all
            X_test: np.array, inputs for the prediction, default is self.X_test
            y_test: np.array, targets for the prediction, default is self.y_test
            ds: copper.Dataset, dataset for the prediction, default is self.test
            legend: boolean, if want the legend on the chart
            ret_list: boolean, True if want the method to return a list with the
                            areas under the curve
            ascending: boolean, legend and list sorting direction

        Returns
        -------
            nothing, the plot is ready to be shown
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        if ds is not None:
            X_test = ds.inputs.values
            y_test = ds.target.values

        aucs = self.auc(clfs=clfs, X_test=X_test, y_test=y_test, ds=ds, ascending=ascending)
        ans = pd.Series(index=clfs)
        for clf_name in aucs.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            probas_ = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            _auc = aucs[clf_name]
            ans[clf_name] = _auc
            plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (clf_name, _auc))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC: Receiver operating characteristic')
        if legend:
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.legend(loc='best')
        if retList:
            return ans.order(ascending=ascending)


    def mse():
        # TODO
        from sklearn.metrics import mean_squared_error

    # --------------------------------------------------------------------------
    #                          Sampling/Crossvalidation
    # --------------------------------------------------------------------------

    def sample(self, ds=None, trainSize=0.5):
        '''
        Samples the dataset into training and testing

        Parameters
        ----------
            ds: copper.Dataset, to use to sample, default, self.dataset
            trainSize: int, percent of the dataset to be used to training,
                                        the remaining will be used to testing

        Returns
        -------
            nothing, self.X_train, self.y_train, self.X_test, self.y_test are set
        '''
        if ds is None:
            ds = self.dataset

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                        ds.inputs.values, ds.target.values,
                        test_size=(1-trainSize), random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def bootstrap(self, clf_class, clf_name, n_iter, X_train=None, y_train=None,
                                                            ds=None, **args):
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
        if X_train is None and y_train is None:
            X_train = self.X_train
            y_train = self.y_train
        if ds is not None:
            X_train = ds.inputs.values
            y_train = ds.target.values

        bs = cross_validation.Bootstrap(len(X_train), n_iter=n_iter)
        i = 0
        for train_index, test_index in bs:
            _X_train = X_train[train_index]
            _y_train = y_train[train_index]
            clf = clf_class(**args)
            clf.fit(_X_train, _y_train)
            self.add_clf(clf, "%s_%i" % (clf_name, i + 1))
            i += 1

    def bagging(self, name, clfs=None):
        '''
        Create a new bag with target classifiers

        Parameters
        ----------
            name: str, name of the new classifier
            clfs: list, of classifiers to include in the bag

        Returns
        -------
            nothing, new classifier is added to the list
        '''
        if clfs is None:
            clfs = self.clfs.index.tolist()

        new = copper.Bagging()
        _clfs = { key: self._clfs[key] for key in clfs }
        new.clfs = _clfs.values()
        self._ensembled[name] = new

    # --------------------------------------------------------------------------
    #                            CONFUSION MATRIX
    # --------------------------------------------------------------------------

    def cm(self, clfs=None, X_test=None, y_test=None, ds=None):
        '''
        Calculates the confusion matrixes of the classifiers

        Parameters
        ----------
            clfs: list, of classifiers to calculate the confusion matrix, default all
            X_test: np.array, inputs for the prediction, default is self.X_test
            y_test: np.array, targets for the prediction, default is self.y_test
            ds: copper.Dataset, dataset for the prediction, default is self.test

        Returns
        -------
            python dictionary
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        if ds is not None:
            X_test = self.test.values
            y_test = self.test.values

        ans = {}
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            y_pred = clf.predict(X_test)
            ans[clf_name] = confusion_matrix(y_test, y_pred)
        return ans

    def plot_cm(self, clf_name, X_test=None, y_test=None, ds=None):
        '''
        Plots the confusion matrixes of the classifier

        Parameters
        ----------
            clf_name: str, classifier identifier
            X_test: np.array, inputs for the prediction, default is self.X_test
            y_test: np.array, targets for the prediction, default is self.y_test
            ds: copper.Dataset, dataset for the prediction, default is self.test
        '''
        plt.matshow(self.cm(X_test=X_test, y_test=y_test, ds=ds)[clf_name])
        plt.title('%s Confusion matrix' % clf_name)
        plt.colorbar()

    def cm_table(self, values=None, clfs=None, X_test=None, y_test=None, ds=None, ascending=False):
        '''
        Calculates the confusion matrix of the classifiers and returns a DataFrame
        for easier visualization

        Parameters
        ----------
            value: int, target value of the target variable. For example if the
                        target variable is binary (0,1) value can be 0 or 1.
            clf_names: list, classifiers identifiers to generate data
            X_test: np.array, inputs for the prediction, default is self.X_test
            y_test: np.array, targets for the prediction, default is self.y_test
            ds: copper.Dataset, dataset for the prediction, default is self.test
            ascending: boolean, list sorting direction

        Returns
        -------
            pandas.DataFrame
        '''
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        if ds is not None:
            X_test = self.test.values
            y_test = self.test.values
        if values is None:
            values = set(y_test)
        elif type(values) is int:
            values = [values]

        cm_s = self.cm(clfs=clfs, X_test=X_test, y_test=y_test)
        ans = pd.DataFrame(index=cm_s.keys())
        for value in values:
            cols = ['Predicted %d\'s' % value, 'Correct %d\'s' % value,
                                    'Rate %d\'s' % value]
            n_ans = pd.DataFrame(index=cm_s.keys(), columns=cols)
            for clf_name in cm_s.keys():
                cm = cm_s[clf_name]
                n_ans['Predicted %d\'s' % value][clf_name] = cm[:,value].sum()
                n_ans['Correct %d\'s' % value][clf_name] = cm[value,value].sum()
                n_ans['Rate %d\'s' % value][clf_name] = cm[value,value].sum() / cm[:,value].sum()
            ans = ans.join(n_ans)
        # return ans
        return ans.sort_index(by='Rate %d\'s' % value, ascending=ascending)

    # --------------------------------------------------------------------------
    #                                 MONEY!
    # --------------------------------------------------------------------------

    def income(self, clfs=None, by='Income', ascending=False):
        '''
        Calculates the Revenue of using the classifiers, self.costs needs to be set.

        Parameters
        ----------
            clf_names: list, classifier identifiers to generate revenue, default all
            by: str, Sort the DataFrame by. Options are: [Loss from False Positive, Revenue, Income]
            ascending: boolean, Sort the DataFrame by direction

        Returns
        -------
            pandas.DataFrame
        '''
        cm_s = self.cm(clfs=clfs)
        cols = ['Loss from False Positive', 'Revenue', 'Income']
        ans = pd.DataFrame(index=cm_s.keys(), columns=cols)

        for clf in cm_s.keys():
            cm = cm_s[clf]
            ans['Loss from False Positive'][clf] = cm[0,1] * self.costs[0][1]
            ans['Revenue'][clf] = cm[1,1] * self.costs[1][1]
            ans['Income'][clf] = ans['Revenue'][clf] - \
                                        ans['Loss from False Positive'][clf]

        return ans.sort_index(by=by, ascending=ascending)

    def oportunity_cost(self, clfs=None, ascending=False):
        '''
        Calculates the Oportuniy Cost of the classifiers, self.costs needs to be set.

        Parameters
        ----------
            clf_names: list, classifier identifiers to generate revenue, default all.
            ascending: boolean, Sort the Series by direction

        Returns
        -------
            pandas.DataFrame
        '''
        cm_s = self.cm(clfs=clfs)
        ans = pd.Series(index=cm_s.keys(), name='Oportuniy cost')

        for clf in cm_s.keys():
            cm = cm_s[clf]
            ans[clf] = cm[1,0] * self.costs[1][0] + cm[0,1] * self.costs[0][1]
        return ans.order(ascending=ascending)

    def income_no_ml(self, ascending=False):
        '''
        Calculate the revenue of not using any classifier

        Parameters
        ----------
            ascending: boolean, Sort the DataFrame by direction

        Returns
        -------
            pandas.Series
        '''
        cols = ['Expense', 'Revenue', 'Net revenue']
        ans = pd.Series(index=cols, name='Revenue of not using ML')

        # TODO: replace for bincount
        # counts = np.bincount(self.y_test)
        counts = []
        counts.append(len(self.y_test[self.y_test == 0]))
        counts.append(len(self.y_test[self.y_test == 1]))
        ans['Expense'] = counts[0] * self.costs[1][0]
        ans['Revenue'] = counts[1] * self.costs[1][1]
        ans['Net revenue'] = ans['Revenue'] - ans['Expense']

        return ans.order(ascending=ascending)

    ## Is this useful?
    def important_features(self, clf_name):
        clf = self._clfs[clf_name]
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title("Feature importances")
        plt.bar(range(len(importances)), importances[indices],
                                color="r", align="center")
        plt.xticks(range(len(importances)), indices)
