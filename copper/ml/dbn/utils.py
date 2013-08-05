# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


def sigmoid(x):
    return 1. / (1 + np.exp(-x))
