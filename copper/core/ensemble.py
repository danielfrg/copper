import numpy as np
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
    def __init__(self):
        self.clfs = []

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

