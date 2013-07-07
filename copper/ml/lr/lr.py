import numpy as np
from scipy import optimize


class LogisticRegression(object):
    def __init__(self, reg_lambda=0):
        self.theta = None
        self.reg_lambda = reg_lambda

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def cost_func(self, theta, X, y):
        m = X.shape[0]
        h = self.hypothesis(X, theta)

        costPos = np.dot(-y.T, np.log(h))
        costNeg = np.dot((1 - y).T, np.log(1 - h))

        J = (costPos - costNeg) / m
        grad = np.dot(X.T, h - y) / m

        # Regularization
        theta_filtered = np.append([0], theta[1:])
        J = J + (self.reg_lambda / (2*m)) * np.dot(theta_filtered.T, theta_filtered)
        grad = grad + np.dot(self.reg_lambda / m, theta_filtered)

        return J, grad

    def fit(self, X, y):
        m, n = X.shape
        X = np.append(np.ones(m).reshape(m, 1), X, axis=1)
        theta0 = np.zeros(n + 1)
        self.theta, _, _ = optimize.fmin_tnc(self.cost_func, theta0,
                                             args=(X, y), disp=False)

    def predict_proba(self, X):
        m, n = X.shape
        X = np.append(np.ones(m).reshape(m, 1), X, axis=1)
        return self.hypothesis(X, self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
