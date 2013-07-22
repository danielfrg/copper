import numpy as np
from scipy import optimize


class LogisticRegression(object):

    def __init__(self, reg_lambda=0, opti_method='TNC', maxiter=1000, coef0=None):
        self.reg_lambda = reg_lambda
        self.method = opti_method
        self.maxiter = maxiter
        self.coef0 = coef0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def function(self, theta, X, y):
        m = X.shape[0]
        h = self.hypothesis(X, theta)

        costPos = np.dot(-y.T, np.log(h))
        costNeg = np.dot((1 - y).T, np.log(1 - h))
        J = (costPos - costNeg) / m

        if self.reg_lambda != 0:
            theta_filtered = np.append([0], theta[1:])
            J = J + (self.reg_lambda / (2 * m)) * np.dot(theta_filtered.T, theta_filtered)

        return J

    def function_prime(self, theta, X, y):
        m = X.shape[0]
        h = self.hypothesis(X, theta)
        grad = np.dot(X.T, h - y) / m

        if self.reg_lambda != 0:
            theta_filtered = np.append([0], theta[1:])
            grad = grad + np.dot(self.reg_lambda / m, theta_filtered)
        return grad

    def fit(self, X, y):
        m, n = X.shape
        X = np.append(np.ones(m).reshape(m, 1), X, axis=1)
        thetas0 = self.coef0 if self.coef0 is not None else np.zeros(n + 1)

        options = {'maxiter': self.maxiter}
        ans = optimize.minimize(self.function, thetas0, jac=self.function_prime, method=self.method,
                                args=(X, y), options=options)
        self.thetas = ans.x

    def predict_proba(self, X):
        m, n = X.shape
        X = np.append(np.ones(m).reshape(m, 1), X, axis=1)
        return self.hypothesis(X, self.thetas)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
