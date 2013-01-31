import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

class MachineLearning():

    def __init__(self):
        self.dataset = None
        self._clfs = {}
        self._ensembled = {}
        self.costs = [[1,0],[0,1]]

    # --------------------------------------------------------------------------
    #                               PROPERTIES
    # --------------------------------------------------------------------------
    # TODO: properties for X_train, X_test,...
    def set_train(self, ds):
        self.X_train = ds.inputs.values
        self.y_train = ds.target.values

    def set_test(self, ds):
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
        Generates a pandas.DataFrame to see all classifiers
        '''
        clfs = list(self._clfs.keys())
        clfs = clfs + list(self._ensembled.keys())
        values = list(self._clfs.values())
        values = values + list(self._ensembled.values())
        return pd.Series(values, index=clfs)

    clfs = property(list_clfs)

    #                               ENSAMBLING

    def sample(self, trainPercent=0.5):
        '''
        Samples the dataset into training and testing

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

    def bagging(self, name, clfs=None):
        '''
        Create a new bag with target classifiers
        '''
        # TODO: list of classifiers
        new = copper.Bagging()
        new.clfs = self._clfs.values()
        self._ensembled[name] = new

    # --------------------------------------------------------------------------
    #                               METHODS
    # --------------------------------------------------------------------------

    def fit(self):
        '''
        Fits all the models
        '''
        for clf_name in self._clfs:
            self._clfs[clf_name].fit(self.X_train, self.y_train)

    def predict(self, X_test=None):
        '''
        Predicts in all classifiers
        '''
        if X_test is None:
            X_test = self.X_test
        ans = pd.DataFrame(columns=self.clfs.index, index=range(len(X_test)))
        for clf_name in self.clfs.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            scores = clf.predict(X_test)
            ans[clf_name][:] = pd.Series(scores)
        return ans

    def predict_proba(self, X_test=None):
        '''
        Predicts with probability for all classifiers
        '''
        if X_test is None:
            X_test = self.X_test
        ans = pd.DataFrame(columns=self.clfs.index, index=range(len(X_test)))
        for clf_name in self.clfs.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            scores = clf.predict_proba(X_test)[:,0]
            ans[clf_name][:] = pd.Series(scores)
        return ans

    def accuracy(self, X_test=None, y_test=None, ascending=False):
        '''
        Calculates the accuracy for the testing set
        '''
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        ans = pd.Series(index=self.clfs.index, name='Accuracy')
        for clf_name in self.clfs.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            ans[clf_name] = clf.score(X_test, y_test)
        return ans.order(ascending=ascending)

    def auc(self, X_test=None, y_test=None, ascending=False):
        '''
        Calculates the Area Under the Curve
        '''
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        ans = pd.Series(index=self.clfs.index, name='Area Under the Curve')
        for clf_name in self.clfs.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            y_pred = clf.predict(X_test)
            ans[clf_name] = auc_score(y_test, y_pred)
        return ans.order(ascending=ascending)

    def roc(self, X_test=None, y_test=None, ascending=False):
        '''
        Plots the ROC chart
        '''
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        auc_s = self.auc(ascending=ascending)
        for clf_name in auc_s.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            probas_ = clf.predict_proba(self.X_test)
            fpr, tpr, thresholds = roc_curve(self.y_test, probas_[:, 1])
            auc = auc_s[clf_name]
            plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (clf_name, auc))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC: Receiver operating characteristic')
        plt.legend(loc="lower right")

    # --------------------------------------------------------------------------
    #                            CONFUSION MATRIX
    # --------------------------------------------------------------------------

    def cm(self, X_test=None):
        if X_test is None:
            X_test = self.X_test
        ans = {}
        for clf_name in self.clfs.index:
            if clf_name in self._clfs.keys():
                clf = self._clfs[clf_name]
            else:
                clf = self._ensembled[clf_name]
            y_pred = clf.predict(self.X_test)
            ans[clf_name] = confusion_matrix(self.y_test, y_pred)
        return ans

    def plot_cm(self, clf):
        import pylab as pl
        pl.matshow(self.cm()[clf])
        pl.title('%s Confusion matrix' % clf)
        pl.colorbar()

    def cm_table(self, value, X_test=None, ascending=False):
        # TODO: make value optional and default all
        if X_test is None:
            X_test = self.X_test
        cols = ['Predicted %d\'s' % value, 'Correct %d\'s' % value,
                                    'Rate %d\'s' % value]
        ans = pd.DataFrame(index=self.clfs.index, columns=cols)

        cm_s = self.cm(X_test=X_test)
        for clf_name in cm_s.keys():
            cm = cm_s[clf_name]
            ans['Predicted %d\'s' % value][clf_name] = cm[:,value].sum()
            ans['Correct %d\'s' % value][clf_name] = cm[value,value].sum()
            ans['Rate %d\'s' % value][clf_name] = cm[value,value].sum() / cm[:,value].sum()

        return ans.sort(ascending=ascending)

    # --------------------------------------------------------------------------
    #                                 MONEY!
    # --------------------------------------------------------------------------

    def revenue(self, by='Net revenue', ascending=False):
        cols = ['Loss from False Positive', 'Revenue', 'Net revenue']
        ans = pd.DataFrame(index=self.clfs.index, columns=cols)

        cm_s = self.cm()
        for clf in cm_s.keys():
            cm = cm_s[clf]
            ans['Loss from False Positive'][clf] = cm[0,1] * self.costs[0][1]
            ans['Revenue'][clf] = cm[1,1] * self.costs[1][1]
            ans['Net revenue'][clf] = ans['Revenue'][clf] - \
                                        ans['Loss from False Positive'][clf]

        return ans.sort_index(by=by, ascending=ascending)

    def oportunity_cost(self, ascending=False):
        ans = pd.Series(index=self.clfs.index, name='Oportuniy cost')

        cm_s = self.cm()
        for clf in cm_s.keys():
            cm = cm_s[clf]
            ans[clf] = cm[1,0] * self.costs[1][0]
        return ans.order(ascending=ascending)

    def revenue_idiot(self, ascending=False):
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
    tree_clf = tree.DecisionTreeClassifier(max_depth=10)

    from sklearn.ensemble import RandomForestClassifier
    ranfor_clf = RandomForestClassifier(n_estimators=10)

    # ml.add_clf(tree_clf, "DT")
    # ml.add_clf(ranfor_clf, "RF")

    # ml.fit()

    bs = cross_validation.Bootstrap(len(ds.inputs.values), n_iter=5)
    i = 0
    for train_index, test_index in bs:
        X_train = ds.inputs.values[test_index]
        y_train = ds.target.values[test_index]
        clf = tree.DecisionTreeClassifier(max_depth=10)
        clf.fit(X_train, y_train)
        ml.add_clf(clf, "DT" + str(i + 1))
        i += 1

    ml.bagging("Bagging")
    print(ml.accuracy())
    # print(ml.auc())
    ml.roc()
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
    # print(ml.cm_table(value=1))
    # print(ml.cm_table(value=0))

    print(ml.revenue())
    print(ml.oportunity_cost())
    print(ml.revenue_idiot())
    '''
