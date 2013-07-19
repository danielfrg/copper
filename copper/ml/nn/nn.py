from __future__ import division
import numpy as np
from scipy import optimize


def build_network(layers):
    return [(layers[i+1], layers[i] + 1) for i in range(len(layers) - 1)]


def pack_weigths(*weights):
    tuples = tuple([theta.reshape(-1) for theta in weights])
    return np.concatenate(tuples)


def unpack_weigths_gen(weights, weights_meta):
    start_pos = 0
    for layer in weights_meta:
        end_pos = start_pos + layer[0] * (layer[1])
        theta = weights[start_pos:end_pos].reshape((layer[0], layer[1]))
        yield theta
        start_pos = end_pos


def unpack_weigths_gen_inv(weights, weights_meta):
    end_pos = len(weights)
    for layer in reversed(weights_meta):
        start_pos = end_pos - layer[0] * (layer[1])
        theta = weights[start_pos:end_pos].reshape((layer[0], layer[1]))
        yield theta
        end_pos = start_pos


def rand_init(weights_meta, epsilon_init):
    s = 0
    for t in weights_meta:
        s += t[0] * t[1]
    return np.random.rand(s, ) * 2 * epsilon_init - epsilon_init


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


def hard_sigmoid(x):
    z = (x * 0.2) + 0.5
    z[z > 1] = 1
    z[z < 0] = 0
    return z


def hard_sigmoid_prime(z):
    sig = hard_sigmoid(z)
    return sig * (1 - sig)


def forward(weights, weights_meta, X, act_func):
    m = X.shape[0]
    ones = np.ones(m).reshape(m,1)

    a_prev = np.hstack((ones, X))  # Input layer
    for theta in unpack_weigths_gen(weights, weights_meta):
        # Hidden Layers
        z = np.dot(theta, a_prev.T)
        a = act_func(z)
        a_prev = np.hstack((ones, a.T))
    return a  # Output layer


def sumsqr(a):
    return np.sum(a ** 2)


def function(weights, weights_meta, X, y, reg_lambda, act_func, act_func_prime):
    m = X.shape[0]
    num_labels = len(np.unique(y))
    Y = np.eye(num_labels)[y]

    h = forward(weights, weights_meta, X, act_func)
    costPositive = -Y * np.log(h).T
    costNegative = (1 - Y) * np.log(1 - h).T
    cost = costPositive - costNegative
    J = np.sum(cost) / m

    if reg_lambda != 0:
        sums_qr = 0
        for theta in unpack_weigths_gen(weights, weights_meta):
            theta_filtered = theta[:, 1:]
            sums_qr += sumsqr(theta_filtered)
        reg = (reg_lambda / (2 * m)) * (sums_qr)
        J = J + reg
    return J


def function_prime(weights, weights_meta, X, y, reg_lambda, act_func, act_func_prime):
    m = X.shape[0]
    num_labels = len(np.unique(y))
    Y = np.eye(num_labels)[y]

    d_s = ()
    Deltas = [np.zeros(w_info) for w_info in weights_meta]
    for i, row in enumerate(X):
        # Forward
        ones = np.array(1).reshape(1,)
        a_prev = np.hstack((ones, row))  # Input layer
        a_s = (a_prev, ) ## a_s[0] == a1
        z_s = ()  # z_s[0] == z2
        for j, theta in enumerate(unpack_weigths_gen(weights, weights_meta)):
            # Hidden Layers
            z = np.dot(theta, a_prev.T)
            z_s = z_s + (z, )
            a = act_func(z)
            a_prev = np.hstack((ones, a.T))
            if j == len(weights_meta) - 1:
                a_s = a_s + (a, )
            else:
                a_s = a_s + (a_prev, )

        # Backprop
        # deltas:= error
        d_prev = a_s[-1] - Y[i, :].T  # last d
        d_s = (d_prev, )  # d_s[0] == d2
        for z_i, theta in zip(reversed(z_s[:-1]), unpack_weigths_gen_inv(weights, weights_meta)):
            d_new = np.dot(theta[:, 1:].T, d_prev) * act_func_prime(z_i)
            d_s = (d_new, ) + d_s
            d_prev = d_new
        for d_i, a_i, i in zip(reversed(d_s), reversed(a_s[:-1]), range(len(Deltas) - 1, -1, -1)):
            Deltas[i] = Deltas[i] + np.dot(d_i[np.newaxis].T, a_i[np.newaxis])

    thetas_gen = None
    if reg_lambda != 0:
        thetas_gen = unpack_weigths_gen(weights, weights_meta)
    for i in range(len(Deltas)):
        Deltas[i] = Deltas[i] / m
        if reg_lambda != 0:
            Deltas[i][:, 1:] = Deltas[i][:, 1:] + (reg_lambda / m) * thetas_gen.next()[:, 1:]
    return pack_weigths(*tuple(Deltas))


class NN(object):

    def __init__(self, hidden_layers=None, opti_method='TNC', maxiter=100,
                 act_func=sigmoid, act_func_prime=sigmoid_prime,
                 epsilon_init=0.12, reg_lambda=0,
                 random_state=0):
        assert hidden_layers is not None, "Please specify hidden_layers"
        self.hidden_layers = hidden_layers
        self.opti_method = opti_method
        self.maxiter = maxiter
        self.act_func = act_func
        self.act_func_prime = act_func_prime
        self.epsilon_init = epsilon_init
        self.reg_lambda = 0
        self.random_state = random_state
        self.coef_ = None

    def fit(self, X, y):
        layers = self.hidden_layers
        layers.insert(0, X.shape[1])
        layers.insert(len(layers), np.unique(y).shape[0])
        weight_metadata = build_network(layers)

        np.random.seed(self.random_state)
        thetas0 = rand_init(weight_metadata, self.epsilon_init)
        options = {'maxiter': self.maxiter}
        ans = optimize.minimize(function, thetas0, jac=function_prime, method=self.opti_method,
                                args=(weight_metadata, X, y, self.reg_lambda, self.act_func, self.act_func_prime),
                                options=options)
        self.coef_ = ans.x
        self.meta_ = weight_metadata

    def predict(self, X):
        return forward(self.coef_, self.meta_, X, self.act_func).argmax(0)
