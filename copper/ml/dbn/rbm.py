# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np


class RBM(object):
    '''
    Note: this class was taken from yusugomori RBM
    https://github.com/yusugomori/DeepLearning/blob/master/python/RBM.py
    I modified the class to match my requirements but basicly is the same thing.
    '''
    def __init__(self, input_layer, learning_rate=0.1, random_state=None):
        self.input_layer = input_layer
        self.learning_rate = learning_rate
        self.activation = input_layer.activation

        if random_state is None:
            self.np_rng = np.random.RandomState()
        else:
            self.np_rng = random_state

        self.n_visible = input_layer.W.shape[0]
        self.n_hidden = input_layer.W.shape[1]
        self.W = input_layer.W

        self.hbias = input_layer.b
        self.vbias = np.zeros(self.n_visible)

    def contrastive_divergence(self, input, lr=0.3, k=1):
        ph_mean, ph_sample = self.sample_h_given_v(input)

        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(ph_sample)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)


        self.W += lr * (np.dot(input.T, ph_sample) -
                        np.dot(nv_samples.T, nh_means)) / input.shape[0]
        self.vbias += lr * np.mean(input - nv_samples, axis=0)
        self.hbias += lr * np.mean(ph_sample - nh_means, axis=0)


    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.np_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.np_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_activation = np.dot(v, self.W) + self.hbias
        return self.activation(pre_activation)

    def propdown(self, h):
        pre_sigmoid_activation = np.dot(h, self.W.T) + self.vbias
        return self.activation(pre_sigmoid_activation)

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return v1_mean, v1_sample, h1_mean, h1_sample


    def get_reconstruction_cross_entropy(self, input):
        pre_activation_h = np.dot(input, self.W) + self.hbias
        sigmoid_activation_h = self.activation(pre_activation_h)

        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = self.activation(pre_sigmoid_activation_v)

        return - np.mean(np.sum(input * np.log(sigmoid_activation_v) +
                         (1 - input) * np.log(1 - sigmoid_activation_v),
                                  axis=1))

    def reconstruct(self, v):
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)
        return reconstructed_v
