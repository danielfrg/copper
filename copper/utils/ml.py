import copper
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn import cross_validation

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

class Bagging(Ensemble):
    def __init__(self, clfs=None):
        if type(clfs) is pd.Series:
            # Comes from ml.clfs
            self.clfs = clfs.values
        elif type(clfs) is list:
            self.clfs = clfs
        else:
            self.clfs = []

    def add_clf(self, new):
        if type(new) is list:
            pass # TODO
        else:            
            self.clfs.append(new)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def predict(self, X_test):
        # TODO: optimize this cuz is very low :P
        prediction = np.zeros(len(X_test))
        temp = np.zeros((len(X_test), len(self.clfs)))
        for i, clf in enumerate(self.clfs):
            temp[:, i] = clf.predict(X_test).T
        for i, row in enumerate(temp):
            row = row.tolist()
            prediction[i] = max(set(row), key=row.count)
        return prediction

    def predict_proba(self, X_test):
        temp = np.zeros((len(X_test), len(self.clfs)))
        for i, clf in enumerate(self.clfs):
            temp[:, i] = clf.predict_proba(X_test)[:, 0]
        probas = np.zeros((len(X_test), 2))
        probas[:,0] = np.mean(temp, axis=1)
        probas[:,1] = 1 - probas[:,0]
        return probas

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

    # ml.add_clf(svm_clf, 'SVM')
    # ml.add_clf(tree_clf, 'DT')
    # ml.add_clf(gnb_clf, 'GNB')
    # ml.add_clf(gr_bst_clf, 'GB')

    # bag = copper.utils.ml.Bagging(ml.clfs)
    bag = copper.utils.ml.Bagging()
    bag.add_clf(tree_clf)
    bag.add_clf(gnb_clf)
    bag.add_clf(gr_bst_clf)
    # ml.add_clf(bag, "bag")


    bootstraped = copper.utils.ml.bootstrap(tree.DecisionTreeClassifier, 5, train, max_depth=6)
    ml.add_clfs(bootstraped, 'tree')
    bootstraped = copper.utils.ml.bootstrap(GaussianNB, 5, train)
    ml.add_clfs(bootstraped, 'GNB')
    # print(bootstraped)


    print (ml.clfs)

    # ml.fit()
    print(ml.accuracy())
    # print(ml.auc())
    # print(ml.mse())