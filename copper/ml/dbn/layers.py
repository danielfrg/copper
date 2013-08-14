# -*- coding: utf-8 -*-
import numpy as np
from utils import softmax, sigmoid


class Layer(object):
    def __init__(self, n_in=None, n_out=None, W=None, b=None, random_state=None, activation=None):
        if random_state is None:
            self.rnd = np.random.RandomState()
        elif isinstance(random_state, int):
            self.rnd = np.random.RandomState(random_state)
        else:
            self.rnd = random_state

        if W is None and b is None:
            if n_in is not None and n_out is not None:
                gap = 4 * np.sqrt(6. / (n_in + n_out))
                self.b = np.zeros(n_out)
                self.W = self.rnd.uniform(low=-gap, high=gap, size=(n_in, n_out))
                self.n_in = self.W.shape[0]
                self.n_out = self.W.shape[1]
        else:
            self.W = W
            self.b = b
            self.n_in = self.W.shape[0]
            self.n_out = self.W.shape[1]

        self.activation = activation

    def output(self, input):
        linear_output = np.dot(input, self.W) + self.b
        return self.activation(linear_output)

    def sample_h_given_v(self, input):
        v_mean = self.output(input)
        h_sample = self.rnd.binomial(size=v_mean.shape, n=1, p=v_mean)
        return h_sample


class LogisticLayer(Layer):
    def __init__(self, *args, **kwargs):
        Layer.__init__(self, *args, activation=softmax, **kwargs)


class SigmoidLayer(Layer):
    def __init__(self, *args, **kwargs):
        Layer.__init__(self, *args, activation=sigmoid, **kwargs)
