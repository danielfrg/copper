import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class Ensemble(object):

    def __init__(self):
        pass

    def score(self, X_test, y_test):
        raise NotImplementedError( "Should have implemented this")

    def predict(self, X_test):
        raise NotImplementedError( "Should have implemented this")

    def predict_proba(self, X_test):
        raise NotImplementedError( "Should have implemented this")


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

    def fit(self, X_train, y_train):
        pass # ;)

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

if __name__ == '__main__':
    import copper
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
    ml.add_clf(tree_clf, 'DT')
    ml.add_clf(gnb_clf, 'GNB')
    ml.add_clf(gr_bst_clf, 'GB')

    # bag = copper.utils.ml.Bagging(ml.clfs)
    bag = copper.utils.ml.Bagging()
    bag.add_clf(tree_clf)
    bag.add_clf(gnb_clf)
    bag.add_clf(gr_bst_clf)
    ml.add_clf(bag, "bag")

    ml.fit()

    print(ml.accuracy())
    print(ml.auc())
    print(ml.mse())