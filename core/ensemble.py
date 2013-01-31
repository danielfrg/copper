import numpy as np

class Ensemble(object):

    def __init__(self):
        pass

    def fit(self):
        pass


class Bagging(Ensemble):
    def __init__(self):
        self.models = []

    def fit(self):
        pass

    def predict(self, X_test):
        prediction = np.zeros(len(X_test))
        temp = np.zeros((len(X_test), len(self.models)))
        for i, clf in enumerate(self.models):
            temp[:, i] = clf.predict(X_test).T
        # TODO: optimize this cuz is very low :P
        for i, row in enumerate(temp):
            row = row.tolist()
            prediction[i] = max(set(row), key=row.count)
        return prediction

    def predict_proba(self, X_test):
        temp = np.zeros((len(X_test), len(self.models)))
        for i, clf in enumerate(self.models):
            temp[:, i] = clf.predict_proba(X_test)[:, 0]
        probas = np.zeros((len(X_test), 2))
        probas[:,0] = np.mean(temp, axis=1)
        probas[:,1] = 1 - probas[:,0]
        return probas

