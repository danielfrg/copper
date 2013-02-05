import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

class MachineLearning():
    '''
    Wrapper around scikit-learn and pandas to make machine learning faster and easier.
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
        Uses copper.DataSet to set the correct values of inputs and targets
        '''
        self.X_train = ds.inputs.values
        self.y_train = ds.target.values

    def set_test(self, ds):
        '''
        Uses copper.DataSet to set the correct values of inputs and targets
        '''
        self.X_test = ds.inputs.values
        self.y_test = ds.target.values

    train = property(None, set_train)
    test = property(None, set_test)

    def add_clf(self, clf, name):
        '''
        Add a new classifier
        '''
        self._clfs[name] = clf

    def rm_clf(self, name):
        '''
        Remove a classifier
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
        Generates a pandas.Series with all the classifiers
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

    def fit(self, clfs=None, X_train=None, y_train=None):
        '''
        Fits all the models
        Parameters
        ----------
            clfs: list, of classifiers to fit, default all
            X_train: np.array, inputs of the training, default is self.X_train
            y_train: np.array, targets of the training, default is self.y_train
        Returns
        -------
            None
        '''
        if clfs is None:
            clfs = self._clfs.keys()
        if X_train is None and y_train is None:
            X_train = self.X_train
            y_train = self.y_train

        for clf_name in clfs:
            self._clfs[clf_name].fit(X_train, y_train)

    def predict(self, clfs=None, X_test=None):
        '''
        Make the classifiers predict inputs
        Parameters
        ----------
            clfs: list, of classifiers to make prediction, default all
            X_test: np.array, inputs for the prediction, default is self.X_test
        Returns
        -------
            pandas.DataFrame with the predictions
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None:
            X_test = self.X_test

        ans = pd.DataFrame(columns=clfs, index=range(len(X_test)))
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            scores = clf.predict(X_test)
            ans[clf_name][:] = pd.Series(scores)
        return ans

    def predict_proba(self, clfs=None, X_test=None):
        '''
        Make the classifiers predict probabilities of inputs
        Parameters
        ----------
            clfs: list, of classifiers to make prediction, default all
            X_test: np.array, inputs of the prediction, default is self.X_test
        Returns
        -------
            pandas.DataFrame with the predictions
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None:
            X_test = self.X_test

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
            clfs: list, of classifiers to fit, default all
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
                                 ascending=False, legend=True, ret_list=False):
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
            plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
        if ret_list:
            return ans.order(ascending=ascending)

    # --------------------------------------------------------------------------
    #                          Sampling/Crossvalidation
    # --------------------------------------------------------------------------

    def sample(self, trainPercent=0.5):
        '''
        Samples the dataset into training and testing
        self.ds needs to be set

        Parameters
        ----------
            trainPercent: int, percent of the dataset to be set to training,
                                the remaining will be set to testing
        '''
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                        self.dataset.inputs.values, self.dataset.target.values,
                        test_size=(1-trainPercent), random_state=0)
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
            X_train: np.array - custom inputs train data, needs y_train
            y_train: np.array - custom target train data, needs X_train
            ds: copper.Dataset, dataset for the prediction, default is self.train
            **args: - arguments of the classifier
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

    def cm(self, clfs=None, X_test=None, y_test=None):
        '''
        Calculates the confusion matrixes of each model
        Parameters
        ----------
            clfs: list, of classifiers to calculate the confusion matrix, default all
            X_test: np.array, custom testing inputs
            y_test: np.array, custom testing target
        Returns
        -------
            python dictionary
        '''
        if clfs is None:
            clfs = self.clfs.index
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test

        ans = {}
        for clf_name in clfs:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            y_pred = clf.predict(X_test)
            ans[clf_name] = confusion_matrix(y_test, y_pred)
        return ans

    def plot_cm(self, clf_name, X_test=None, y_test=None):
        '''
        Plots the confusion matrixes of the classifier
        Parameters
        ----------
            clf_name: str, classifier identifier
            X_test: np.array, custom testing inputs
            y_test: np.array, custom testing target
        '''
        plt.matshow(self.cm(X_test, y_test)[clf_name])
        plt.title('%s Confusion matrix' % clf_name)
        plt.colorbar()

    def cm_table(self, value, clfs=None, X_test=None, y_test=None, ascending=False):
        '''
        Calculates the confusion matrix of the classifiers and returns a DataFrame
        for easier visualization
        Parameters
        ----------
            value: int, target value of the target variable. For example if the
                        target variable is binary (0,1) value can be 0 or 1.
            clf_names: list, classifiers identifiers to generate data
            X_test: np.array, custom testing inputs
            y_test: np.array, custom testing target
            ascending: boolean, list sorting direction
        Returns
        -------
            pandas.DataFrame
        '''
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test

        cm_s = self.cm(clfs=clfs, X_test=X_test, y_test=y_test)
        cols = ['Predicted %d\'s' % value, 'Correct %d\'s' % value,
                                    'Rate %d\'s' % value]
        ans = pd.DataFrame(index=cm_s.keys(), columns=cols)

        for clf_name in cm_s.keys():
            cm = cm_s[clf_name]
            ans['Predicted %d\'s' % value][clf_name] = cm[:,value].sum()
            ans['Correct %d\'s' % value][clf_name] = cm[value,value].sum()
            ans['Rate %d\'s' % value][clf_name] = cm[value,value].sum() / cm[:,value].sum()

        return ans.sort(ascending=ascending)

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
            ans[clf] = cm[1,0] * self.costs[1][0]
        return ans.order(ascending=ascending)

    def revenue_no_ml(self, ascending=False):
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
        ans = pd.Series(index=cols, name='Revenue Idiot')

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


if __name__ == '__main__':
    # ''' DONORS
    copper.config.path = '../tests'

    ds = copper.read_csv('donors/data.csv')
    ds.role['TARGET_D'] = ds.REJECTED
    ds.role['TARGET_B'] = ds.TARGET
    ds.type['ID'] = ds.CATEGORY

    ds.fillna('DemAge', 'mean')
    ds.fillna('GiftAvgCard36', 'mean')

    ml = copper.MachineLearning()
    ml.dataset = ds
    ml.sample(0.5)

    from sklearn import tree
    tree_clf = tree.DecisionTreeClassifier(compute_importances=True, max_depth=10)
    # ml.add_clf(tree_clf, 'DT')

    # from sklearn.naive_bayes import GaussianNB
    # gnb_clf = GaussianNB(compute_importances=True)
    # ml.add_clf(svc_clf, 'SVM')

    # ml.fit()

    ml.bootstrap(tree.DecisionTreeClassifier, "DT", 20, max_depth=4)
    # print(ml.clfs)

    ml.bagging("Bagging")
    # print(ml.accuracy())
    # print(ml.auc())
    ml.roc()
    # print(ml.roc(ret_list=True))
    plt.show()
    # '''

    ''' # CATALOG
    copper.config.path = '../tests'

    ds_train = copper.read_csv('catalog/training.csv')
    ds_train.type['RFA1'] = ds_train.NUMBER
    ds_train.type['RFA2'] = ds_train.NUMBER
    ds_train.type['Order'] = ds_train.NUMBER
    ds_train.role['CustomerID'] = ds_train.ID
    ds_train.role['Order'] = ds_train.TARGET

    ds_test = copper.read_csv('catalog/testing.csv')
    ds_test.type['RFA1'] = ds_test.NUMBER
    ds_test.type['RFA2'] = ds_test.NUMBER
    ds_test.type['Order'] = ds_test.NUMBER
    ds_test.role['CustomerID'] = ds_test.ID
    ds_test.role['Order'] = ds_test.TARGET

    ml = copper.MachineLearning()
    ml.train = ds_train
    ml.test = ds_test

    from sklearn import svm
    svm_clf = svm.SVC(probability=True)

    from sklearn import tree
    tree_clf = tree.DecisionTreeClassifier(max_depth=6)

    from sklearn.naive_bayes import GaussianNB
    gnb_clf = GaussianNB()

    from sklearn.ensemble import GradientBoostingClassifier
    gr_bst_clf = GradientBoostingClassifier()

    # ml.add_clf(svm_clf, 'SVM')
    ml.add_clf(tree_clf, 'Decision Tree')
    ml.add_clf(gnb_clf, 'GaussianNB')
    ml.add_clf(gr_bst_clf, 'Grad Boosting')
    ml.bagging("Bag 1")
    ml.fit()

    # print(ml.accuracy())
    # print(ml.auc())
    # print(ml.predict().head())
    # print(ml.predict_proba().head())
    # ml.roc()
    # plt.show()

    ml.costs = [[0, 4], [12, 16]]

    # print(ml.cm())
    # print(ml.cm_table(value=0))
    # print(ml.cm_table(clfs=['Decision Tree'], value=1))
    # print(ml.cm_table(clfs=['GaussianNB'], value=0))

    print(ml.income())
    # print(ml.oportunity_cost(clfs=['Bag 1']))
    # print(ml.revenue_no_ml())
    '''
