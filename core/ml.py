import copper
import numpy as np
import pandas as pd
from sklearn import cross_validation

class MachineLearning():

    def __init__(self):
        self.dataset = None
        self._models = {}
        self._ensembled = {}
        self.costs = [[1,0],[0,1]]

    # --------------------------------------------------------------------------
    #                               PROPERTIES
    # --------------------------------------------------------------------------

    def set_train(self, ds):
        self.X_train = ds.inputs.values
        self.y_train = ds.target.values

    def set_test(self, ds):
        self.X_test = ds.inputs.values
        self.y_test = ds.target.values

    train = property(None, set_train)
    test = property(None, set_test)

    def add_model(self, clf, _id):
        self._models[_id] = clf

    def remove_model(self, _id):
        del self._models[_id]

    def get_models(self):
        models = list(self._models.keys())
        models = models + list(self._ensembled.keys())
        values = list(self._models.values())
        values = values + list(self._ensembled.values())
        return pd.Series(values, index=models)

    models = property(get_models)

    # --------------------------------------------------------------------------
    #                               METHODS
    # --------------------------------------------------------------------------

    def sample(self, trainPercent=0.5):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                        self.dataset.inputs.values, self.dataset.target.values,
                        test_size=(1-trainPercent), random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        for model in self._models:
            clf = self._models[model]
            clf.fit(self.X_train, self.y_train)

    def predict(self):
        ans = pd.DataFrame(columns=self._models.keys(), index=range(len(self.X_test)))
        for model in self._models:
            clf = self._models[model]
            scores = clf.predict(self.X_test)
            ans[model][:] = pd.Series(scores)
        return ans

    def predict_proba(self):
        ans = pd.DataFrame(columns=self._models.keys(), index=range(len(self.X_test)))
        for model in self._models:
            clf = self._models[model]
            scores = clf.predict_proba(self.X_test)
            ans[model][:] = pd.Series(scores[:,0])
        return ans

    def accuracy(self, ascending=False):
        from sklearn.metrics import accuracy_score
        ans = pd.Series(index=self.models.index, name='Accuracy')

        for model in self._models.keys():
            clf = self._models[model]
            ans[model] = clf.score(self.X_test, self.y_test)
        for model in self._ensembled.keys():
            clf = self._ensembled[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = accuracy_score(self.y_test, y_pred)
        return ans.order(ascending=ascending)

    def auc(self, ascending=False):
        from sklearn.metrics import auc_score
        ans = pd.Series(index=self.models.index, name='Area Under the Curve')

        for model in self._models.keys():
            clf = self._models[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = auc_score(self.y_test, y_pred)
        for model in self._ensembled.keys():
            clf = self._ensembled[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = auc_score(self.y_test, y_pred)
        return ans.order(ascending=ascending)

    def roc(self, ascending=False):
        import pylab as pl
        from sklearn.metrics import roc_curve

        auc_s = self.auc(ascending=ascending)

        for model in auc_s.index:
            if model in self._ensembled.keys():
                clf = self._ensembled[model]
            else:
                clf = self._models[model]
            probas_ = clf.predict_proba(self.X_test)
            fpr, tpr, thresholds = roc_curve(self.y_test, probas_[:, 1])
            auc = auc_s[model]
            pl.plot(fpr, tpr, label='%s (area = %0.2f)' % (model, auc))

        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC: Receiver operating characteristic')
        pl.legend(loc="lower right")

    # --------------------------------------------------------------------------
    #                            CONFUSION MATRIX
    # --------------------------------------------------------------------------

    def cm(self):
        from sklearn.metrics import confusion_matrix
        ans = {}

        for model in self._models.keys():
            clf = self._models[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = confusion_matrix(self.y_test, y_pred)
        return ans

    def plot_cm(self, model):
        import pylab as pl
        pl.matshow(self.cm()[model])
        pl.title('%s Confusion matrix' % model)
        pl.colorbar()

    def cm_table(self, value):
        cols = ['Predicted %d\'s' % value, 'Correct %d\'s' % value, 'Rate']
        ans = pd.DataFrame(index=self._models.keys(), columns=cols)

        cm_s = self.cm()
        for model in cm_s.keys():
            cm = cm_s[model]
            ans['Predicted %d\'s' % value][model] = cm[:,value].sum()
            ans['Correct %d\'s' % value][model] = cm[value,value].sum()
            ans['Rate'][model] = cm[value,value].sum() / cm[:,value].sum()

        return ans

    # --------------------------------------------------------------------------
    #                               ENSAMBLING
    # --------------------------------------------------------------------------

    def bagging(self, name):
        new = copper.Bagging()
        new.models = self._models.values()
        new.X_test = self.X_test
        self._ensembled[name] = new

    # --------------------------------------------------------------------------
    #                                 MONEY!
    # --------------------------------------------------------------------------

    def revenue(self, by='Net revenue', ascending=False):
        cols = ['Loss from False Positive', 'Revenue', 'Net revenue']
        ans = pd.DataFrame(index=self._models.keys(), columns=cols)

        cm_s = self.cm()
        for model in cm_s.keys():
            cm = cm_s[model]
            ans['Loss from False Positive'][model] = cm[0,1] * self.costs[0][1]
            ans['Revenue'][model] = cm[1,1] * self.costs[1][1]
            ans['Net revenue'][model] = ans['Revenue'][model] - \
                                        ans['Loss from False Positive'][model]

        return ans.sort_index(by=by, ascending=ascending)

    def oportunity_cost(self, ascending=False):
        ans = pd.Series(index=self._models.keys(), name='Oportuniy cost')

        cm_s = self.cm()
        for model in cm_s.keys():
            cm = cm_s[model]
            ans[model] = cm[1,0] * self.costs[1][0]
        return ans.order(ascending=ascending)

    def revenue_idiot(self, ascending=False):
        cols = ['Expense', 'Revenue', 'Net revenue']
        ans = pd.Series(index=cols)

        # counts = np.bincount(self.y_test)
        counts = []
        counts.append(len(self.y_test[self.y_test == 0]))
        counts.append(len(self.y_test[self.y_test == 1]))
        ans['Expense'] = counts[0] * self.costs[1][0]
        ans['Revenue'] = counts[1] * self.costs[1][1]
        ans['Net revenue'] = ans['Revenue'] - ans['Expense']

        return ans.order(ascending=ascending)

if __name__ == '__main__':
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

    # ml.add_model(tree_clf, "DT")
    # ml.add_model(ranfor_clf, "RF")

    # ml.fit()

    bs = cross_validation.Bootstrap(len(ds.inputs.values), n_iter=5)
    i = 0
    for train_index, test_index in bs:
        X_train = ds.inputs.values[test_index]
        y_train = ds.target.values[test_index]
        clf = tree.DecisionTreeClassifier(max_depth=10)
        clf.fit(X_train, y_train)
        ml.add_model(clf, "DT" + str(i + 1))
        i += 1

    ml.bagging("Bagging")
    print(ml.accuracy())
    # print(ml.auc())
    import matplotlib.pyplot as plt
    # ml.roc()
    plt.show()


    # scores = cross_validation.cross_val_score(tree_clf, ds.inputs.values, ds.target.values, cv=5)
    # i = 0
    # for train_index, test_index in bs:
    #     X_train = ds.inputs.values[test_index]
    #     y_train = ds.target.values[test_index]
    #     clf = RandomForestClassifier(n_estimators=10)
    #     clf.fit(X_train, y_train)
    #     ml.add_model(clf, "RF" + str(i))
    #     i += 1


    ''' # CATALOG
    import matplotlib.pyplot as plt
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

    # ml.add_model(svm_clf, 'SVM')
    ml.add_model(tree_clf, 'Decision Tree')
    ml.add_model(gnb_clf, 'GaussianNB')
    ml.add_model(gr_bst_clf, 'Grad Boosting')

    ml.fit()

    # print(ml.auc())
    # print(ml.predict().head())
    # print(ml.predict_proba().head())
    # ml.roc()
    # plt.show()

    ml.costs = [[0, 4], [12, 16]]

    # print(ml.cm_table(value=1))
    # print(ml.cm_table(value=0))

    # print(ml.revenue())
    # print(ml.oportunity_cost())
    print(ml.revenue_idiot())
    '''


    # IRIS

    # copper.config.path = '../examples/iris'
    # ds = copper.read_csv('iris.csv')
    # ds.role['class'] = ds.TARGET

    # from sklearn import svm
    # clf = svm.SVC(gamma=0.001, C=100, probability=True)
    # clf2 = svm.SVC(gamma=1, C=100, probability=True)

    # ml = copper.MachineLearning()
    # ml.dataset = ds
    # ml.sample()
    # ml.add_model(clf, 'svm')
    # ml.add_model(clf2, 'svm 2')

    # ml.fit()

    # print(ml.accuracy())
    # print(ml.cf())
    # ml.plot_cf('svm')
    # plt.show()


