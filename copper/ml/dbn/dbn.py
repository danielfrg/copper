# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy as np
from sklearn.base import BaseEstimator

from rbm import RBM
from layers import SigmoidLayer
from copper.utils import opti as MBOpti
from utils import sigmoid
from copper.utils.progress import ProgressBar


def unpack_weigths(weights, weights_meta):
    start_pos = 0
    for layer in weights_meta:
        end_pos = start_pos + layer[0] * (layer[1])
        yield weights[start_pos:end_pos].reshape((layer[0], layer[1]))
        start_pos = end_pos


def cost(weights, X, y, weights_meta, num_labels):
    # Forward
    act_prev = np.insert(X, 0, 1, axis=1)
    for weight in unpack_weigths(weights, weights_meta):
        z = np.dot(act_prev, weight)
        activation = sigmoid(z)
        act_prev = np.insert(activation, 0, 1, axis=1)

    Y = np.eye(num_labels)[y]
    h = activation
    costPositive = -Y * np.log(h)
    costNegative = (1 - Y) * np.log(1 - h)
    J = np.sum(costPositive - costNegative) / X.shape[0]
    return J


def unpack_weigths_inv(weights, weights_meta):
    end_pos = len(weights)
    for layer in reversed(weights_meta):
        start_pos = end_pos - layer[0] * (layer[1])
        yield weights[start_pos:end_pos].reshape((layer[0], layer[1]))
        end_pos = start_pos


def cost_prime(weights, X, y, weights_meta, num_labels):
    Y = np.eye(num_labels)[y]
    Deltas = [np.zeros(shape) for shape in weights_meta]

    data = np.insert(X, 0, 1, axis=1)
    for i, row in enumerate(data):
        # Forward
        #row = np.array([row])
        act_prev = row
        activations = (act_prev, )
        for weight in unpack_weigths(weights, weights_meta):
            z = np.dot(act_prev, weight)
            activation = sigmoid(z)
            act_prev = np.append(1, activation)
            activations = activations + (act_prev, )

        # Backprop
        prev_delta = activations[-1][1:] - Y[i, :].T  # last delta
        deltas = (prev_delta, )  # deltas[0] == delta2
        for act, weight in zip(reversed(activations[1:-1]), unpack_weigths_inv(weights, weights_meta)):
            delta = np.dot(weight, prev_delta)[1:] * (act[1:] * (1 - act[1:])).T
            deltas = (delta, ) + deltas
            prev_delta = delta

        # Accumulate errors
        for delta, act, i in zip(deltas, activations[:-1], range(len(Deltas))):
            Deltas[i] = Deltas[i] + np.dot(delta[np.newaxis].T, act[np.newaxis]).T
    for i in range(len(Deltas)):
        Deltas[i] = Deltas[i] / X.shape[0]
    return np.concatenate(tuple([D.reshape(-1) for D in Deltas]))


class DBN(BaseEstimator):

    def __init__(self, hidden_layers, coef0=None, random_state=None,
                 progress_bars=False,
                 pretrain_batch_size=50,
                 pretrain_epochs=1, pretrain_batches_per_epoch=None,
                 pretrain_callback=None,
                 finetune_method='GD', finetune_batch_size=50,
                 finetune_epochs=1, finetune_batches_per_epoch=None,
                 finetune_options=None, finetune_callback=None):
        self.hidden_layers = hidden_layers
        self.coef_ = None if coef0 is None else np.copy(coef0)

        if random_state is None:
            self.rnd = np.random.RandomState()
        elif isinstance(random_state, int):
            self.rnd = np.random.RandomState(random_state)
        else:
            self.rnd = random_state

        self.progress_bars = progress_bars

        self.pretrain_batch_size = pretrain_batch_size
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_batches_per_epoch = pretrain_batches_per_epoch
        self.pretrain_callback = pretrain_callback
        self.finetune_method = finetune_method
        self.finetune_batch_size = finetune_batch_size
        self.finetune_epochs = finetune_epochs
        self.finetune_batches_per_epoch = finetune_batches_per_epoch

        self.finetune_options = {} if finetune_options is None else finetune_options
        self.finetune_callback = finetune_callback

    def rand_init(self, weights_info, random_state):
        w_sizes = []
        for layer_info in weights_info:
            w_sizes.append(layer_info[0] * layer_info[1])
        ans = np.zeros(sum(w_sizes))
        start_pos = 0
        for i, layer in enumerate(weights_info):
            end_pos = start_pos + layer[0] * (layer[1])
            gap = 4 * np.sqrt(6. / (layer[0] + layer[1]))
            ans[start_pos:end_pos] = random_state.uniform(low=-gap, high=gap, size=w_sizes[i])
            start_pos = end_pos
        return ans

    def predict_proba(self, X):
        output = self.layers[0].output(X)
        for layer in self.layers[1:]:
            output = layer.output(output)
        return output

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def fit(self, X, y):
        layers = list(self.hidden_layers)  # Copy
        layers.insert(0, X.shape[1])
        layers.insert(len(layers), len(np.unique(y)))
        weights_info = [(layers[i] + 1, layers[i + 1]) for i in range(len(layers) - 1)]

        if self.coef_ is None:
            self.coef_ = self.rand_init(weights_info, self.rnd)

        # Unpack the weights and assign them to the layers
        self.layers = []
        start_pos = 0
        for w_info in weights_info:
            end_pos = start_pos + w_info[0] * (w_info[1])
            weight = self.coef_[start_pos:end_pos].reshape((w_info[0], w_info[1]))
            self.layers.append(SigmoidLayer(W=weight))
            start_pos = end_pos

        # Create RBM layers using the same weights
        self.rbm_layers = []
        for layer in self.layers:
            self.rbm_layers.append(RBM(layer, random_state=self.rnd))

        # Pretrain
        if self.pretrain_epochs > 0:
            if self.progress_bars:
                max_batchs = self.pretrain_batches_per_epoch
                if max_batchs is None:
                    max_batchs = int(math.floor(X.shape[0] / self.pretrain_batch_size))
                maxiters = self.pretrain_epochs * max_batchs * len(self.layers)
                pt_bar = ProgressBar(max=maxiters, desc='Pretrain')

            for layer in range(len(self.rbm_layers)):
                if layer == 0:
                    input = X
                else:
                    #input = self.layers[layer - 1].output(input)
                    input = self.layers[layer - 1].sample_h_given_v(input)

                for epoch in range(self.pretrain_epochs):
                    batches = MBOpti.minibatches(input, batch_size=self.pretrain_batch_size,
                                                 batches_per_epoch=self.pretrain_batches_per_epoch,
                                                 random_state=self.rnd)
                    for i, _X in enumerate(batches):
                        self.rbm_layers[layer].contrastive_divergence(_X)
                        if self.progress_bars:
                            pt_bar.next()
                        if self.pretrain_callback is not None:
                            stop = self.pretrain_callback(self, layer, epoch + 1, i + 1)
                            if stop is True:
                                break
            if self.progress_bars:
                pt_bar.finish()

        # Finetune
        if self.finetune_epochs > 0:
            if self.progress_bars:
                max_batchs = self.finetune_batches_per_epoch
                if max_batchs is None:
                    max_batchs = int(math.floor(X.shape[0] / self.finetune_batch_size))
                maxiters = self.finetune_epochs * max_batchs
                ft_bar = ProgressBar(max=maxiters, desc='Finetune')

            def _callback(epoch, i):
                if self.progress_bars:
                    ft_bar.next()
                if self.finetune_callback is not None:
                    return self.finetune_callback(self, epoch, i)

            self.finetune_options = self.finetune_options.copy()
            args = (weights_info, len(np.unique(y)))
            MBOpti.minimize(X, y, fun=cost, grad=cost_prime, weights=self.coef_,
                            method=self.finetune_method, epochs=self.finetune_epochs,
                            batches_per_epoch=self.finetune_batches_per_epoch,
                            options=self.finetune_options, args=args, callback=_callback,
                            random_state=self.rnd)
