from __future__ import division
import math
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
    return np.divide(1, (1 + np.exp(-z)))


def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


def forward(weights, weights_meta, X, act_func):
    m = X.shape[0]
    ones = np.ones(m).reshape(m, 1)

    a_prev = np.hstack((ones, X))  # Input layer
    for theta in unpack_weigths_gen(weights, weights_meta):
        # Hidden Layers
        z = np.dot(theta, a_prev.T)
        a = act_func(z)
        a_prev = np.hstack((ones, a.T))
    return a  # Output layer


def sumsqr(a):
    return np.sum(a ** 2)


def function(weights, X, y, weights_meta, num_labels, reg_lambda, act_func, act_func_prime):
    m = X.shape[0]
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


def function_prime(weights, X, y, weights_meta, num_labels, reg_lambda, act_func, act_func_prime):
    m = X.shape[0]
    Y = np.eye(num_labels)[y]
    ones = np.array(1).reshape(1,)

    d_s = ()
    Deltas = [np.zeros(w) for i, w in enumerate(weights_meta)]
    for i, row in enumerate(X):
        # Forward
        a_prev = np.hstack((ones, row))  # Input layer
        a_s = (a_prev, ) # a_s[0] == a1
        for j, theta in enumerate(unpack_weigths_gen(weights, weights_meta)):
            # Hidden Layers
            z = np.dot(theta, a_prev.T)
            a = act_func(z)
            a_prev = np.hstack((ones, a.T))
            a_s = a_s + (a_prev, )

        # Backprop
        d_prev = a_s[-1][1:] - Y[i, :].T  # last d
        d_s = (d_prev, )  # d_s[0] == d2
        for a_i, theta in zip(reversed(a_s[1:-1]), unpack_weigths_gen_inv(weights, weights_meta)):
            d_new = np.dot(theta[:, 1:].T, d_prev) * (a_i[1:] * (1 - a_i[1:]))
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


def minibatch(X, y, batch_size=1):
    m = X.shape[0]
    batch_size = batch_size if batch_size >= 1 else int(math.floor(m * batch_size))
    max_batchs = int(math.floor(m / batch_size))

    indices = np.random.choice(np.arange(m), m, replace=False)
    X, y = X[indices], y[indices]
    while True:
        for i in range(max_batchs):
            indices = np.arange(i * batch_size, (i + 1) * batch_size)
            yield X[indices], y[indices]


def mb_gd(func, func_prime, thetas0, X, y, options=None, args=()):
    batch_size = options['batch_size']
    maxiter = options['maxiter']
    learning_rate = options['learning_rate']
    tol = options['tol']
    disp = options['disp']
    thetas = thetas0
    diff = 1
    prevJ = 1000
    i = 0
    for _X, _y in minibatch(X, y, batch_size):
        thetas = thetas - learning_rate * func_prime(thetas, _X, _y, *args)
        newJ = float(function(thetas, X, y, *args))
        if disp:
            print(i, newJ)
        if not np.isnan(newJ) and newJ != float("inf"):
            diff = np.abs(newJ - prevJ)

            prevJ = newJ
            if diff < tol or i >= maxiter:
                break
        i += 1
    return thetas


def mb_rmsprop(func, func_prime, thetas0, X, y, options=None, args=()):
    print('RMSPROP')
    batch_size = options['batch_size']
    maxiter = options['maxiter']
    tol = options['tol']
    disp = options['disp']

    thetas = thetas0
    diff = 1
    prevJ = 1000
    i = 0
    rms = 1
    for _X, _y in minibatch(X, y, batch_size):
        grad = func_prime(thetas, _X, _y, *args)
        rms = 0.9 * rms + 0.1 * np.square(grad)
        thetas = thetas - np.divide(grad, np.sqrt(rms))
        newJ = float(function(thetas, X, y, *args))
        if disp:
            print(i, newJ)
        if not np.isnan(newJ) and newJ != float("inf"):
            diff = np.abs(newJ - prevJ)
            prevJ = newJ
            if diff < tol or i >= maxiter:
                break
        i += 1
    return thetas


def mb_scipy(func, func_prime, thetas0, X, y, options=None, args=()):
    print('scipy', options['mb_opti'])
    batch_size = options['batch_size']
    maxiter = options['maxiter']
    tol = options['tol']
    disp = options['disp']
    thetas = thetas0
    diff = 1
    prevJ = 1000
    i = 0
    thetas = thetas0
    for _X, _y in minibatch(X, y, batch_size):
        ans = optimize.minimize(func, thetas, jac=func_prime, method=options['mb_opti'],
                                args=(_X, _y) + args,
                                options={'maxiter': options['mb_opti_maxiter']})
        thetas = ans.x
        newJ = float(function(thetas, X, y, *args))
        if disp:
            print(i, newJ)
        if not np.isnan(newJ) and newJ != float("inf"):
            diff = np.abs(newJ - prevJ)

            prevJ = newJ
            if diff < tol or i >= maxiter:
                break
        i += 1
    return thetas


def mb_opti(func, func_prime, thetas0, X, y, options=None, args=()):
    if options['mb_opti'] == 'GD':
        return mb_gd(function, function_prime, thetas0, X, y, options=options, args=args)
    elif options['mb_opti'] == 'RMSPROP':
        return mb_rmsprop(function, function_prime, thetas0, X, y, options=options, args=args)
    else:
        return mb_scipy(function, function_prime, thetas0, X, y, options=options, args=args)


class NN(object):

    def __init__(self, hidden_layers,
                 opti_method='MB', maxiter=100, tol = 0.000001,
                 mb_opti='CG', batch_size=0.1, learning_rate=0.3, mb_opti_maxiter=10,
                 disp=False,
                 act_func=sigmoid, act_func_prime=sigmoid_prime,
                 epsilon_init=0.12, random_state=0,
                 reg_lambda=0, coef0=None):
        self.hidden_layers = hidden_layers
        self.opti_method = opti_method
        self.maxiter = maxiter
        self.tol = tol
        self.mb_opti = mb_opti
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mb_opti_maxiter = mb_opti_maxiter
        self.disp = disp
        self.act_func = act_func
        self.act_func_prime = act_func_prime
        self.epsilon_init = epsilon_init
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.coef0 = coef0
        self.coef_ = None

    def predict(self, X):
        return forward(self.coef_, self.meta_, X, self.act_func).argmax(0)

    def fit(self, X, y):
        layers = self.hidden_layers
        layers.insert(0, X.shape[1])
        layers.insert(len(layers), np.unique(y).shape[0])
        weight_metadata = build_network(layers)
        num_labels = len(np.unique(y))

        np.random.seed(self.random_state)

        thetas0 = self.coef0 if self.coef0 is not None else rand_init(weight_metadata, self.epsilon_init)
        options = {}

        options['maxiter'] = self.maxiter
        options['disp'] = self.disp

        if self.opti_method == 'MB':
            print 'fast'
            options['mb_opti'] = self.mb_opti
            options['batch_size'] = self.batch_size
            options['learning_rate'] = self.learning_rate
            options['tol'] = self.tol
            options['mb_opti_maxiter'] = self.mb_opti_maxiter
            ans = mb_opti(function, function_prime, thetas0, X, y, options=options,
                          args=(weight_metadata, num_labels, self.reg_lambda, self.act_func, self.act_func_prime))
        else:
            print 'slow', self.opti_method
            ans = optimize.minimize(function, thetas0, jac=function_prime, method=self.opti_method,
                                    args=(X, y, weight_metadata, num_labels, self.reg_lambda, self.act_func, self.act_func_prime),
                                    options=options)
            ans = ans.x
        self.coef_ = ans
        self.meta_ = weight_metadata
