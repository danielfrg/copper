# -*- coding: utf-8 -*-
import numpy as np
from utils import softmax, sigmoid


class Layer(object):
    def __init__(self, n_in=None, n_out=None, W=None, random_state=None, activation=None):
        if random_state is None:
            self.rnd = np.random.RandomState()
        else:
            self.rnd = random_state

        if W is None:
            self.W = self.rnd.uniform(size=(n_in + 1, n_out))
        else:
            self.W = W

        self.activation = activation

    def output(self, input):
        data = np.insert(input, 0, 1, axis=1)
        linear_output = np.dot(data, self.W)
        return self.activation(linear_output)

    def sample_h_given_v(self, input):
        v_mean = self.output(input)
        h_sample = self.rnd.binomial(size=v_mean.shape, n=1, p=v_mean)
        return h_sample


class LogisticLayer(Layer):
    def __init__(self, n_in=None, n_out=None, W=None, random_state=None):
        Layer.__init__(self, n_in, n_out, W, random_state, activation=softmax)


class SigmoidLayer(Layer):
    def __init__(self, n_in=None, n_out=None, W=None, random_state=None):
        Layer.__init__(self, n_in, n_out, W, random_state, activation=sigmoid)
