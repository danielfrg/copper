# -*- coding: utf-8 -*-
from __future__ import division, print_function
import math
import numpy as np
from scipy import optimize


def minibatches(X, y=None, batch_size=50, batches_per_epoch=None, random_state=None):
    if random_state is None:
        rnd = np.random.RandomState()
    elif isinstance(random_state, int):
        rnd = np.random.RandomState(random_state)
    else:
        rnd = random_state

    m = X.shape[0]
    batch_size = batch_size if batch_size >= 1 else int(math.floor(m * batch_size))
    if batches_per_epoch is None:
        batches_per_epoch = int(math.floor(m / batch_size))

    random_indices = rnd.choice(np.arange(m), m, replace=False)
    for i in range(batches_per_epoch):
        batch_indices = np.arange(i * batch_size, (i + 1) * batch_size)
        indices = random_indices[batch_indices]
        if y is None:
            yield X[indices]
        else:
            yield X[indices], y[indices]


def GD(fun, weights, grad, X, y, options, args=()):
    weights -= options['learning_rate'] * grad(weights, X, y, *args)
    options['learning_rate'] = options['learning_rate'] * options['learning_rate_decay']


def GD_momentum(fun, weights, grad, X, y, options, args=()):
    bigjump = options['momentum'] * options['step']
    weights -= bigjump
    correction = options['learning_rate'] * grad(weights, X, y, *args)
    weights -= correction
    options['step'] = bigjump + correction
    options['learning_rate'] = options['learning_rate'] * options['learning_rate_decay']
    options['momentum'] = options['momemtum_decay'] * options['momentum']


def RMSPROP(fun, weights, grad, X, y, options, args=()):
    gradient = grad(weights, X, y, *args)
    options['moving_mean_squared'] = options['decay'] * options['moving_mean_squared'] + (1 - options['decay']) * gradient ** 2
    weights -= gradient / np.sqrt(options['moving_mean_squared'] + 1e-8)


def CG(fun, weights, grad, X, y, options, args=()):
    ans = optimize.minimize(fun, weights, jac=grad, method='CG', args=(X, y) + args, options={'maxiter': options['mb_maxiter']})
    weights[:] = ans.x


def LBFGSB(fun, weights, grad, X, y, options, args=()):
    ans = optimize.minimize(fun, weights, jac=grad, method='L-BFGS-B', args=(X, y) + args, options={'maxiter': options['mb_maxiter']})
    weights[:] = ans.x


def minimize(X, y, fun, grad, weights, method,
             epochs=10, batches_per_epoch=None, batch_size=50,
             random_state=None,
             options=None, args=None, callback=None):
    if method == 'GD':
        if 'learning_rate' not in options:
            options['learning_rate'] = 0.2
        if 'learning_rate_decay' not in options:
            options['learning_rate_decay'] = 1
        if 'momentum' in options:
            if 'momemtum_decay' not in options:
                options['momemtum_decay'] = 1
            options['step'] = 0
            update = GD_momentum
        else:
            update = GD
    elif method == 'RMSPROP':
        options['moving_mean_squared'] = 1
        update = RMSPROP
    elif method == 'CG':
        if 'mb_maxiter' not in options:
                options['mb_maxiter'] = 10
        update = CG
    elif method == 'L-BFGS-B':
        if 'mb_maxiter' not in options:
                options['mb_maxiter'] = 10
        update = LBFGSB
    else:
        raise Exception('Optimization method not found')

    if random_state is None:
        rnd = np.random.RandomState()
    elif isinstance(random_state, int):
        rnd = np.random.RandomState(random_state)
    else:
        rnd = random_state

    for epoch in range(epochs):
        batches = minibatches(X, y, batch_size=batch_size, batches_per_epoch=batches_per_epoch,
                              random_state=rnd)
        i = 0
        for _X, _y in batches:
            update(fun, weights, grad, _X, _y, options, args=args)
            if callback is not None:
                stop = callback(epoch + 1, i + 1)
                if stop:
                    break
                i += 1
