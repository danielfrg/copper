"""
 Copyright (c) 2011,2012 George Dahl

 Permission is hereby granted, free of charge, to any person  obtaining
 a copy of this software and associated documentation  files (the
 "Software"), to deal in the Software without  restriction, including
 without limitation the rights to use,  copy, modify, merge, publish,
 distribute, sublicense, and/or sell  copies of the Software, and to
 permit persons to whom the  Software is furnished to do so, subject
 to the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.  THE
 SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,  EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES  OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT  HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR  OTHER DEALINGS IN THE
 SOFTWARE.
"""

import numpy as num
from .. import gnumpy as gnp

class Binary(object):
    def activate(self, netInput):
        return netInput.sigmoid()
    def sampleStates(self, acts):
        return gnp.rand(*acts.shape) <= acts

class Gaussian(object):
    def activate(self, netInput):
        return netInput
    def sampleStates(self, acts): #probably shouldn't use this
        return acts + gnp.randn(*acts.shape)

class ReLU(object):
    def __init__(self, krizNoise = False):
        self.krizNoise = krizNoise
    def activate(self, netInput):
        return netInput*(netInput > 0)
    def sampleStates(self, acts):
        if self.krizNoise:
            return self.activate(acts + gnp.randn(*acts.shape))
        tiny = 1e-30
        stddev = gnp.sqrt(acts.sigmoid() + tiny)
        return self.activate( acts + stddev*gnp.randn(*acts.shape) )


def CD1(vis, visToHid, visBias, hidBias, visUnit = Binary(), hidUnit = Binary()):
    """
    Using Gaussian hidden units hasn't been tested. By assuming the
    visible units are Binary, ReLU, or Gaussian and the hidden units
    are Binary or ReLU this function becomes quite simple.
    """
    posHid = hidUnit.activate(gnp.dot(vis, visToHid) + hidBias)
    posHidStates = hidUnit.sampleStates(posHid)

    negVis = visUnit.activate(gnp.dot(posHidStates, visToHid.T) + visBias)
    negHid = hidUnit.activate(gnp.dot(negVis, visToHid) + hidBias)

    visHidStats = gnp.dot(vis.T, posHid) - gnp.dot(negVis.T, negHid)
    visBiasStats = vis.sum(axis=0).reshape(*visBias.shape) - negVis.sum(axis=0).reshape(*visBias.shape)
    hidBiasStats = posHid.sum(axis=0).reshape(*hidBias.shape) - negHid.sum(axis=0).reshape(*hidBias.shape)

    return visHidStats, hidBiasStats, visBiasStats, negVis
