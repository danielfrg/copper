import copper
import numpy as np
import pandas as pd
from sklearn import cross_validation

class MachineLearning():

    def __init__(self):
        self.dataset = None
        self.models = {}
        self.costs = [[1,0],[0,1]]

    def set_train(self, ds):
        self.X_train = ds.inputs.values
        self.y_train = ds.target.values

    def set_test(self, ds):
        self.X_test = ds.inputs.values
        self.y_test = ds.target.values

    train = property(None, set_train)
    test = property(None, set_test)

    def add_model(self, clf, _id):
        self.models[_id] = clf

    def sample(self, trainPercent=0.5):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                        self.dataset.inputs, self.dataset.target,
                        test_size=(1-trainPercent), random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        for model in self.models:
            clf = self.models[model]
            clf.fit(self.X_train, self.y_train)

    def predict(self):
        ans = pd.DataFrame(columns=self.models.keys(), index=range(len(self.X_test)))
        for model in self.models:
            clf = self.models[model]
            scores = clf.predict(self.X_test)
            ans[model][:] = pd.Series(scores)
        return ans

    def predict_proba(self):
        ans = pd.DataFrame(columns=self.models.keys(), index=range(len(self.X_test)))
        for model in self.models:
            clf = self.models[model]
            scores = clf.predict_proba(self.X_test)
            ans[model][:] = pd.Series(scores[:,0])
        return ans

    def accuracy(self, ascending=False):
        ans = pd.Series(index=self.models.keys(), name='Accuracy')

        for model in self.models.keys():
            clf = self.models[model]
            ans[model] = clf.score(self.X_test, self.y_test)
        return ans.order(ascending=ascending)

    def auc(self, ascending=False):
        from sklearn.metrics import auc_score
        ans = pd.Series(index=self.models.keys(), name='Accuracy')

        for model in self.models.keys():
            clf = self.models[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = auc_score(self.y_test, y_pred)
        return ans.order(ascending=ascending)

    def roc(self, ascending=False):
        import pylab as pl
        from sklearn.metrics import roc_curve

        auc_s = self.auc(ascending=ascending)

        for model in auc_s.index:
            clf = self.models[model]
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

    def cm(self):
        from sklearn.metrics import confusion_matrix
        ans = {}

        for model in self.models.keys():
            clf = self.models[model]
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
        ans = pd.DataFrame(index=self.models.keys(), columns=cols)

        cm_s = self.cm()
        for model in cm_s.keys():
            cm = cm_s[model]
            ans['Predicted %d\'s' % value][model] = cm[:,value].sum()
            ans['Correct %d\'s' % value][model] = cm[value,value].sum()
            ans['Rate'][model] = cm[value,value].sum() / cm[:,value].sum()

        return ans

    def revenue(self, by='Net revenue', ascending=False):
        cols = ['Loss from False Positive', 'Revenue', 'Net revenue']
        ans = pd.DataFrame(index=self.models.keys(), columns=cols)

        cm_s = self.cm()
        for model in cm_s.keys():
            cm = cm_s[model]
            ans['Loss from False Positive'][model] = cm[0,1] * self.costs[0][1]
            ans['Revenue'][model] = cm[1,1] * self.costs[1][1]
            ans['Net revenue'][model] = ans['Revenue'][model] - \
                                        ans['Loss from False Positive'][model]

        return ans.sort_index(by=by, ascending=ascending)

    def oportunity_cost(self, ascending=False):
        ans = pd.Series(index=self.models.keys(), name='Oportuniy cost')

        cm_s = self.cm()
        for model in cm_s.keys():
            cm = cm_s[model]
            ans[model] = cm[1,0] * self.costs[1][0]
        return ans.order(ascending=ascending)

    def revenue_idiot(self, ascending=False):
        cols = ['Expense', 'Revenue', 'Net revenue']
        ans = pd.Series(index=cols)
        print(type(self.y_test))

        # counts = np.bincount(self.y_test)
        counts = []
        counts.append(len(self.y_test[self.y_test == 0]))
        counts.append(len(self.y_test[self.y_test == 1]))
        ans['Expense'] = counts[0] * self.costs[1][0]
        ans['Revenue'] = counts[1] * self.costs[1][1]
        ans['Net revenue'] = ans['Revenue'] - ans['Expense']

        return ans.order(ascending=ascending)

if __name__ == '__main__':
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


