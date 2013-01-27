import copper
import pandas as pd
from sklearn import cross_validation

class MachineLearning():

    def __init__(self):
        self.dataset = None
        self.models = {}

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

    def score(self):
        # Is this useful?
        ans = pd.Series(name='Score')
        for model in self.models:
            clf = self.models[model]
            scores = clf.score(X_test, y_test)
            print(scores)

    def accuracy(self):
        from sklearn.metrics import accuracy_score
        ans = pd.Series(index=self.models.keys(), name='Accuracy')

        for model in self.models.keys():
            clf = self.models[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = accuracy_score(self.y_test, y_pred)
        return ans

    def auc(self):
        from sklearn.metrics import auc_score
        ans = pd.Series(index=self.models.keys(), name='Accuracy')

        for model in self.models.keys():
            clf = self.models[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = auc_score(self.y_test, self.y_pred)
        return ans

    def cf(self):
        from sklearn.metrics import confusion_matrix
        ans = {}

        for model in self.models.keys():
            clf = self.models[model]
            y_pred = clf.predict(self.X_test)
            ans[model] = confusion_matrix(self.y_test, y_pred)
        return ans

    def roc(self):
        import pylab as pl
        from sklearn.metrics import roc_curve, auc

        for model in self.models.keys():
            clf = self.models[model]
            probas_ = clf.predict_proba(self.X_test)
            fpr, tpr, thresholds = roc_curve(self.y_test, probas_[:, 1])
            roc_auc = auc(fpr, tpr)
            pl.plot(fpr, tpr, label='%s (area = %0.2f)' % (model, roc_auc))

        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC: Receiver operating characteristic')
        pl.legend(loc="lower right")

    def plot_cf(self, model):
        import pylab as pl
        pl.matshow(self.cf()[model])
        pl.title('%s Confusion matrix' % model)
        pl.colorbar()

    def predict(self, inputs):
        pass # TODO

    def set_train(self, ds):
        self.X_train = ds.inputs
        self.y_train = ds.target

    def set_test(self, ds):
        self.X_test = ds.inputs
        self.y_test = ds.target

    train = property(None, set_train)
    test = property(None, set_test)

    # dataset = property(get_dataset, set_dataset) # In case of needed

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    copper.config.path = '../examples/catalog'

    ds_train = copper.read_csv('training.csv')
    ds_train.type['RFA1'] = ds_train.NUMBER
    ds_train.type['RFA2'] = ds_train.NUMBER
    ds_train.type['Order'] = ds_train.NUMBER
    ds_train.role['CustomerID'] = ds_train.ID
    ds_train.role['Order'] = ds_train.TARGET
    fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
    ds_train.transform('LASD', fnc)

    ds_test = copper.read_csv('testing.csv')
    ds_test.type['RFA1'] = ds_test.NUMBER
    ds_test.type['RFA2'] = ds_test.NUMBER
    ds_test.type['Order'] = ds_test.NUMBER
    ds_test.role['CustomerID'] = ds_test.ID
    ds_test.role['Order'] = ds_test.TARGET

    fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
    ds_test.transform('LASD', fnc)

    ml = copper.MachineLearning()
    ml.train = ds_train
    ml.test = ds_test

    from sklearn import svm
    svm_clf = svm.SVC(probability=True)

    from sklearn import tree
    tree_clf = tree.DecisionTreeClassifier(max_depth=6)

    from sklearn.naive_bayes import GaussianNB
    gnb_clf = GaussianNB()

    ml.add_model(svm_clf, 'SVM')
    ml.add_model(tree_clf, 'Decision Tree')
    ml.add_model(gnb_clf, 'GaussianNB')

    ml.fit()

    ml.roc()
    plt.show()


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


