# coding=utf-8
from __future__ import division
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copper.core.ensemble import *

from sklearn import cross_validation
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


class MachineLearning():
    '''
    Wrapper around scikit-learn and pandas to make machine learning faster and easier
    Utilities for model selection.
    '''

    def __init__(self):
        self.dataset = None
        self._clfs = {}
        self._ensembled = {}
        self.costs = [[1,-1],[-1,1]]
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
        '''
        self.X_train = copper.transform.inputs2ml(ds).values
        self.y_train = copper.transform.target2ml(ds).values

    def set_test(self, ds):
        '''
        Uses a Dataset to set the values of inputs and targets for testing
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

    def fit(self):
        '''
        Fit all the classifiers
        '''
        for clf_name in self.clfs.index:
            self._clfs[clf_name].fit(self.X_train, self.y_train)

    def predict(self, ds=None, clfs=None, ):
        '''
        Make the classifiers predict the testing inputs

        Parameters
        ----------
            ds: copper.Dataset, dataset fot the prediction, default is self.test
            clfs: list, of classifiers to make prediction, default all

        Returns
        -------
            pandas.DataFrame with the predictions
        '''
        if clfs is None:
            clfs = self.clfs.index
        if ds is not None:
            X_test = copper.transform.inputs2ml(ds).values
        else:
            X_test = self.X_test

        ans = pd.DataFrame(np.zeros((len(X_test), len(clfs))), columns=clfs, index=range(len(X_test)))
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            scores = clf.predict(X_test)
            ans[clf_name][:] = pd.Series(scores)
        return ans

    def predict_proba(self, ds=None, clfs=None, ):
        '''
        Make the classifiers predict probabilities of inputs
        Parameters
        ----------
            ds: copper.Dataset, dataset fot the prediction, default is self.test
            clfs: list, of classifiers to make prediction, default all

        Returns
        -------
            pandas.DataFrame with the predicted probabilities
        '''
        if clfs is None:
            clfs = self.clfs.index
        if ds is not None:
            X_test = copper.transform.inputs2ml(ds).values
        else:
            X_test = self.X_test

        ans = pd.DataFrame(np.zeros((len(X_test), len(clfs))), columns=clfs, index=range(len(X_test)))
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            scores = clf.predict_proba(X_test)[:,0]
            ans[clf_name][:] = pd.Series(scores)
        return ans

    # ----------------------------------------------------------------------------------------
    #                                            METRICS
    # ----------------------------------------------------------------------------------------

    def _metric_wrapper(self, fnc, name='', ascending=False):
        '''
        Wraper to not repeat code on all the possible metrics
        '''
        ans = pd.Series(index=self._clfs, name=name)
        for clf_name in self._clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            ans[clf_name] = fnc(clf, X_test=self.X_test, y_test=self.y_test)
        return ans.order(ascending=ascending)

    def accuracy(self, **args):
        '''
        Calculates the accuracy of inputs

        Parameters
        ----------
            ascending: boolean, sort the Series on this direction

        Returns
        -------
            pandas.Series with the accuracy
        '''
        def fnc (clf, X_test=None, y_test=None):
            return clf.score(X_test, y_test)

        return self._metric_wrapper(fnc, name='Accuracy', **args)

    def auc(self, **args):
        '''
        Calculates the Area Under the ROC Curve

        Parameters
        ----------
            ascending: boolean, sort the Series on this direction

        Returns
        -------
            pandas.Series with the Area under the Curve
        '''
        def fnc (clf, X_test=None, y_test=None):
            probas = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
            return auc(fpr, tpr)

        return self._metric_wrapper(fnc, name='Area Under the Curve', **args)

    def mse(self, **args):
        '''
        Calculates the Mean Squared Error

        Parameters
        ----------
            ascending: boolean, sort the Series on this direction

        Returns
        -------
            pandas.Series with the Mean Squared Error
        '''
        def fnc (clf, X_test=None, y_test=None):
            y_pred = clf.predict(X_test)
            return mean_squared_error(y_test, y_pred)

        return self._metric_wrapper(fnc, name='Mean Squared Error', **args)

    # --------------------------------------------------------------------------
    #                          Sampling / Crossvalidation
    # --------------------------------------------------------------------------

    def sample(self, ds, trainSize=0.5):
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
        inputs = copper.transform.inputs2ml(ds).values
        target = copper.transform.target2ml(ds).values

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                        inputs.values, target.values,
                        test_size=(1-trainSize), random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def bootstrap(self, clf_class, clf_name, n_iter, ds=None, **args):
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
        if ds is not None:
            X_train = copper.transform.inputs2ml(ds).values
            y_train = copper.transform.target2ml(ds).values

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

    def _cm(self, clfs=None):
        '''
        Calculates the confusion matrixes of the classifiers

        Parameters
        ----------
            clfs: list or str, of the classifiers to calculate the cm

        Returns
        -------
            python dictionary
        '''
        if clfs is None:
            clfs = self._clfs.keys()
        else :
            if type(clfs) is str:
                clfs = [clfs]

        ans = {}
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            y_pred = clf.predict(self.X_test)
            ans[clf_name] = confusion_matrix(self.y_test, y_pred)
        return ans

    def cm(self, clf):
        '''
        Return a pandas.DataFrame version of a confusion matrix

        Parameters
        ----------
            clf: str, classifier identifier
        '''
        cm = self._cm(clfs=clf)[clf]
        values = set(self.y_test)
        return pd.DataFrame(cm, index=values, columns=values)

    def cm_table(self, values=None, ascending=False):
        '''
        Returns a more information about the confusion matrix

        Parameters
        ----------
            value: int, target value of the target variable. For example if the
                        target variable is binary (0,1) value can be 0 or 1.
            ascending: boolean, list sorting direction

        Returns
        -------
            pandas.DataFrame
        '''
        if values is None:
            values = set(self.y_test)
        elif type(values) is int:
            values = [values]

        cm_s = self._cm()
        ans = pd.DataFrame(index=cm_s.keys())
        zeros = np.zeros((len(ans), 3))

        for value in values:
            cols = ['Predicted %d\'s' % value, 'Correct %d\'s' % value,
                                    'Rate %d\'s' % value]
            n_ans = pd.DataFrame(zeros ,index=cm_s.keys(), columns=cols)
            for clf_name in cm_s.keys():
                cm = cm_s[clf_name]
                n_ans['Predicted %d\'s' % value][clf_name] = cm[:,value].sum()
                n_ans['Correct %d\'s' % value][clf_name] = cm[value,value].sum()
                n_ans['Rate %d\'s' % value][clf_name] = cm[value,value].sum() / cm[:,value].sum()
            ans = ans.join(n_ans)
        return ans.sort_index(by='Rate %d\'s' % value, ascending=ascending)

    def cm_falses(self):
        # TODO: like above but for false negative, false positive
        pass

    # --------------------------------------------------------------------------
    #                                 COSTS
    # --------------------------------------------------------------------------

    def profit(self, by='Profit', ascending=False):
        '''
        Calculates the Revenue of using the classifiers.
        self.costs should be modified to get better information.

        Parameters
        ----------
            by: str, sort the DataFrame by. Options are: Loss from False Positive, Revenue, Profit
            ascending: boolean, Sort the DataFrame by direction

        Returns
        -------
            pandas.DataFrame
        '''
        cm_s = self._cm()
        cols = ['Loss from False Positive', 'Revenue', 'Profit']
        ans = pd.DataFrame(np.zeros((len(cm_s.keys()), 3)), index=cm_s.keys(), columns=cols)

        for clf in ans.index:
            cm = cm_s[clf]
            ans['Loss from False Positive'][clf] = cm[0,1] * self.costs[0][1]
            ans['Revenue'][clf] = cm[1,1] * self.costs[1][1]
            ans['Profit'][clf] = ans['Revenue'][clf] - \
                                        ans['Loss from False Positive'][clf]

        return ans.sort_index(by=by, ascending=ascending)

    def oportunity_cost(self, ascending=False):
        '''
        Calculates the Oportuniy Cost of the classifiers.
        self.costs should be modified to get better information.

        Parameters
        ----------
            ascending: boolean, Sort the Series by direction

        Returns
        -------
            pandas.DataFrame
        '''
        cm_s = self._cm()
        ans = pd.Series(index=cm_s.keys(), name='Oportuniy cost')

        for clf in ans.index:
            cm = cm_s[clf]
            ans[clf] = cm[1,0] * self.costs[1][0] + cm[0,1] * self.costs[0][1]
        return ans.order(ascending=ascending)

    def cost_no_ml(self, ascending=False):
        '''
        Calculate the revenue of not using any classifier.
        self.costs should be modified to get better information.

        Parameters
        ----------
            ascending: boolean, Sort the DataFrame by direction

        Returns
        -------
            pandas.Series
        '''
        cols = ['Expense', 'Revenue', 'Net revenue']
        ans = pd.Series(index=cols, name='Costs of not using ML')

        # TODO: replace for bincount
        # counts = np.bincount(self.y_test)
        counts = []
        counts.append(len(self.y_test[self.y_test == 0]))
        counts.append(len(self.y_test[self.y_test == 1]))
        ans['Expense'] = counts[0] * self.costs[1][0]
        ans['Revenue'] = counts[1] * self.costs[1][1]
        ans['Net revenue'] = ans['Revenue'] - ans['Expense']

        return ans.order(ascending=ascending)

    # --------------------------------------------------------------------------
    #                                 PLOTS
    # --------------------------------------------------------------------------

    def roc(self, ascending=False, legend=True, ret_list=False):
        '''
        Plots the ROC chart

        Parameters
        ----------
            legend: boolean, if want the legend on the chart
            ret_list: boolean, True if want the method to return a list with the
                            areas under the curve
            ascending: boolean, legend and list sorting direction

        Returns
        -------
            nothing, the plot is ready to be shown
        '''
        aucs = self.auc(ascending=ascending)
        for clf_name in aucs.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            try:
                probas_ = clf.predict_proba(self.X_test)
                fpr, tpr, thresholds = roc_curve(self.y_test, probas_[:, 1])
                plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (clf_name, aucs[clf_name]))
            except:
                pass # Is OK, some models do not have predict_proba

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC: Receiver operating characteristic')

        if legend:
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.legend(loc='best')
        if ret_list:
            return aucs

    def plot_cm(self, clf):
        '''
        Plots the confusion matrixes of the classifier

        Parameters
        ----------
            clf: str, classifier identifier
            X_test: np.array, inputs for the prediction, default is self.X_test
            y_test: np.array, targets for the prediction, default is self.y_test
            ds: copper.Dataset, dataset for the prediction, default is self.test
        '''
        plt.matshow(self.cm()[clf])
        plt.title('%s Confusion matrix' % clf)
        plt.colorbar()


if __name__ == '__main__':
    copper.project.path = '../tests/'

    train = copper.Dataset('ml/1/train.csv')
    train.role['CustomerID'] = train.ID
    train.role['Order'] = train.TARGET
    fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
    train['LASD'] = train['LASD'].apply(fnc)

    test = copper.Dataset('ml/1/test.csv')
    test.role['CustomerID'] = test.ID
    test.role['Order'] = test.TARGET
    test['LASD'] = test['LASD'].apply(fnc)

    ml = copper.MachineLearning()
    ml.set_train(train)
    ml.set_test(test)

    from sklearn import svm
    svm_clf = svm.SVC(probability=True)
    from sklearn import tree
    tree_clf = tree.DecisionTreeClassifier(max_depth=6)
    from sklearn.naive_bayes import GaussianNB
    gnb_clf = GaussianNB()
    from sklearn.ensemble import GradientBoostingClassifier
    gr_bst_clf = GradientBoostingClassifier()

    ml.add_clf(svm_clf, 'SVM')
    ml.add_clf(tree_clf, 'DT')
    ml.add_clf(gnb_clf, 'GNB')
    ml.add_clf(gr_bst_clf, 'GB')

    ml.fit()

    print(ml.predict().dtypes)
    # print(copper.save(ml.predict_proba(), 'predit_proba_test'))
    # print(copper.save(ml.predict_proba(ds=train), 'predict_proba_train'))

