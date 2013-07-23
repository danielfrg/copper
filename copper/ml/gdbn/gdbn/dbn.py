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
import itertools
from activationFunctions import *
from pretrain import CD1
from pretrain import Binary as RBMBinary
from pretrain import Gaussian as RBMGaussian
from pretrain import ReLU as RBMReLU
from counter import Progress

class DummyProgBar(object):
    def __init__(self, *args): pass
    def tick(self): pass
    def done(self): pass

def initWeightMatrix(shape, scale, maxNonZeroPerColumn = None, uniform = False):
    #number of nonzero incoming connections to a hidden unit
    fanIn = shape[0] if maxNonZeroPerColumn==None else min(maxNonZeroPerColumn, shape[0])
    if uniform:
        W = scale*(2*num.random.rand(*shape)-1)
    else:
        W = scale*num.random.randn(*shape)
    for j in range(shape[1]):
        perm = num.random.permutation(shape[0])
        W[perm[fanIn:],j] *= 0
    return W

def validShapes(weights, biases):
    if len(weights) + 1 == len(biases):
        t1 = all(b.shape[0] == 1 for b in biases)
        t2 = all(wA.shape[1] == wB.shape[0] for wA, wB in zip(weights[:-1], weights[1:]))
        t3 = all(w.shape[1] == hb.shape[1] for w, hb in zip(weights, biases[1:]))
        t4 = all(w.shape[0] == vb.shape[1] for w, vb in zip(weights, biases[:-1]))
        return t1 and t2 and t3 and t4
    return False

def garrayify(arrays):
    return [ar if isinstance(ar, gnp.garray) else gnp.garray(ar) for ar in arrays]

def numpyify(arrays):
    return [ar if isinstance(ar, num.ndarray) else ar.as_numpy_array(dtype=num.float32) for ar in arrays]

def loadDBN(path, outputActFunct, realValuedVis = False, useReLU = False):
    fd = open(path, 'rb')
    d = num.load(fd)
    weights = garrayify(d['weights'].flatten())
    biases = garrayify(d['biases'].flatten())
    genBiases = []
    if 'genBiases' in d:
        genBiases = garrayify(d['genBiases'].flatten())
    fd.close()
    return DBN(weights, biases, genBiases, outputActFunct, realValuedVis, useReLU)

def buildDBN(layerSizes, scales, fanOuts, outputActFunct, realValuedVis, useReLU = False, uniforms = None):
    shapes = [(layerSizes[i-1],layerSizes[i]) for i in range(1, len(layerSizes))]
    assert(len(scales) == len(shapes) == len(fanOuts))
    if uniforms == None:
        uniforms = [False for s in shapes]
    assert(len(scales) == len(uniforms))

    initialBiases = [gnp.garray(0*num.random.rand(1, layerSizes[i])) for i in range(1, len(layerSizes))]
    initialGenBiases = [gnp.garray(0*num.random.rand(1, layerSizes[i])) for i in range(len(layerSizes) - 1)]
    initialWeights = [gnp.garray(initWeightMatrix(shapes[i], scales[i], fanOuts[i], uniforms[i])) \
                      for i in range(len(shapes))]

    net = DBN(initialWeights, initialBiases, initialGenBiases, outputActFunct, realValuedVis, useReLU)
    return net

def columnRMS(W):
    return gnp.sqrt(gnp.mean(W*W,axis=0))

def limitColumnRMS(W, rmsLim):
    """
    All columns of W with rms entry above the limit are scaled to equal the limit.
    The limit can either be a row vector or a scalar.
    """
    rmsScale = rmsLim/columnRMS(W)
    return W*(1 + (rmsScale < 1)*(rmsScale-1))

class DBN(object):
    def __init__(self, initialWeights, initialBiases, initialGenBiases, outputActFunct, realValuedVis = False, useReLU = False):
        self.realValuedVis = realValuedVis
        self.learnRates = [0.05 for i in range(len(initialWeights))]
        self.momentum = 0.9
        self.L2Costs = [0.0001 for i in range(len(initialWeights))]
        self.dropouts = [0 for i in range(len(initialWeights))]
        self.nesterov = False
        self.nestCompare = False
        self.rmsLims = [None for i in range(len(initialWeights))]

        if self.realValuedVis:
            self.learnRates[0] = 0.005

        self.weights = initialWeights
        self.biases = initialBiases
        self.genBiases = initialGenBiases

        if useReLU:
            self.RBMHidUnitType = RBMReLU()
            self.hidActFuncts = [ReLU() for i in range(len(self.weights) - 1)]
        else:
            self.RBMHidUnitType = RBMBinary()
            self.hidActFuncts = [Sigmoid() for i in range(len(self.weights) - 1)]
        self.outputActFunct = outputActFunct

        #state variables modified in bprop
        self.WGrads = [gnp.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        self.biasGrads = [gnp.zeros(self.biases[i].shape) for i in range(len(self.biases))]

    def weightsDict(self):
        d = {}
        if len(self.weights) == 1:
            d['weights'] = num.empty((1,), dtype=num.object)
            d['weights'][0] = numpyify(self.weights)[0]
            d['biases'] = num.empty((1,), dtype=num.object)
            d['biases'][0] = numpyify(self.biases)[0]
        else:
            d['weights'] = num.array(numpyify(self.weights)).flatten()
            d['biases'] = num.array(numpyify(self.biases)).flatten()
            if len(self.genBiases) == 1:
                d['genBiases'] = num.empty((1,), dtype=num.object)
                d['genBiases'][0] = numpyify(self.genBiases)[0]
            else:
                d['genBiases'] = num.array(numpyify(self.genBiases)).flatten()
        return d

    def scaleDerivs(self, scale):
        for i in range(len(self.weights)):
            self.WGrads[i] *= scale
            self.biasGrads[i] *= scale

    def loadWeights(self, path, layersToLoad = None):
        fd = open(path, 'rb')
        d = num.load(fd)
        if layersToLoad != None:
            self.weights[:layersToLoad] = garrayify(d['weights'].flatten())[:layersToLoad]
            self.biases[:layersToLoad] = garrayify(d['biases'].flatten())[:layersToLoad]
            self.genBiases[:layersToLoad] = garrayify(d['genBiases'].flatten())[:layersToLoad] #this might not be quite right
        else:
            self.weights = garrayify(d['weights'].flatten())
            self.biases = garrayify(d['biases'].flatten())
            if 'genBiases' in d:
                self.genBiases = garrayify(d['genBiases'].flatten())
            else:
                self.genBiases = []
        fd.close()

    def saveWeights(self, path):
        num.savez(path, **self.weightsDict())

    def preTrainIth(self, i, minibatchStream, epochs, mbPerEpoch):
        #initialize CD gradient variables
        self.dW = gnp.zeros(self.weights[i].shape)
        self.dvb = gnp.zeros(self.genBiases[i].shape)
        self.dhb = gnp.zeros(self.biases[i].shape)

        for ep in range(epochs):
            recErr = 0
            totalCases = 0
            for j in range(mbPerEpoch):
                inpMB = minibatchStream.next()
                curRecErr = self.CDStep(inpMB, i, self.learnRates[i], self.momentum, self.L2Costs[i])
                recErr += curRecErr
                totalCases += inpMB.shape[0]
            yield recErr/float(totalCases)

    def fineTune(self, minibatchStream, epochs, mbPerEpoch, loss = None, progressBar = True, useDropout = False):
        for ep in range(epochs):
            totalCases = 0
            sumErr = 0
            sumLoss = 0
            if self.nesterov:
                step = self.stepNesterov
            else:
                step = self.step
            prog = Progress(mbPerEpoch) if progressBar else DummyProgBar()
            for i in range(mbPerEpoch):
                inpMB, targMB = minibatchStream.next()
                err, outMB = step(inpMB, targMB, self.learnRates, self.momentum, self.L2Costs, useDropout)
                sumErr += err
                if loss != None:
                    sumLoss += loss(targMB, outMB)
                totalCases += inpMB.shape[0]
                prog.tick()
            prog.done()
            yield sumErr/float(totalCases), sumLoss/float(totalCases)

    def totalLoss(self, minibatchStream, lossFuncts):
        totalCases = 0
        sumLosses = num.zeros((1+len(lossFuncts),))
        for inpMB, targMB in minibatchStream:
            inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
            targetBatch = targMB if isinstance(targMB, gnp.garray) else gnp.garray(targMB)

            outputActs = self.fprop(inputBatch)
            sumLosses[0] += self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
            for j,f in enumerate(lossFuncts):
                sumLosses[j+1] += f(targetBatch, outputActs)
            totalCases += inpMB.shape[0]
        return sumLosses / float(totalCases)

    def predictions(self, minibatchStream, asNumpy = False):
        for inpMB in minibatchStream:
            inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
            outputActs = self.fprop(inputBatch)
            yield outputActs.as_numpy_array() if asNumpy else outputActs

    def CDStep(self, inputBatch, layer, learnRate, momentum, L2Cost = 0):
        """
        layer=0 will train the first RBM directly on the input
        """
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        mbsz = inputBatch.shape[0]
        vis = self.fprop(inputBatch, layer)
        GRBMFlag = layer==0 and self.realValuedVis
        visType = RBMGaussian() if GRBMFlag else self.RBMHidUnitType
        visHidStats, hidBiasStats, visBiasStats, negVis = \
                     CD1(vis, self.weights[layer], self.genBiases[layer], self.biases[layer], visType, self.RBMHidUnitType)
        factor = 1-momentum if not self.nestCompare else 1
        self.dW = momentum*self.dW + factor*visHidStats
        self.dvb = momentum*self.dvb + factor*visBiasStats
        self.dhb = momentum*self.dhb + factor*hidBiasStats

        if L2Cost > 0:
            self.weights[layer] *= 1-L2Cost*learnRate*factor

        self.weights[layer] += (learnRate/mbsz) * self.dW
        self.genBiases[layer] += (learnRate/mbsz) * self.dvb
        self.biases[layer] += (learnRate/mbsz) * self.dhb

        #we compute squared error even for binary visible unit RBMs because who cares
        return gnp.sum((vis-negVis)**2)

    def fpropBprop(self, inputBatch, targetBatch, useDropout):
        if useDropout:
            outputActs = self.fpropDropout(inputBatch)
        else:
            outputActs = self.fprop(inputBatch)
        outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, self.state[-1], outputActs)
        error = self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
        errSignals = self.bprop(outputErrSignal)
        return errSignals, outputActs, error

    def constrainWeights(self):
        for i in range(len(self.rmsLims)):
            if self.rmsLims[i] != None:
                self.weights[i] = limitColumnRMS(self.weights[i], self.rmsLims[i])

    def step(self, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False):
        mbsz = inputBatch.shape[0]
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)

        errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout)

        factor = 1-momentum if not self.nestCompare else 1.0
        self.scaleDerivs(momentum)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.WGrads[i] += learnRates[i]*factor*(WGrad/mbsz - L2Costs[i]*self.weights[i])
            self.biasGrads[i] += (learnRates[i]*factor/mbsz)*biasGrad
        self.applyUpdates(self.weights, self.biases, self.weights, self.biases, self.WGrads, self.biasGrads)
        self.constrainWeights()
        return error, outputActs

    def applyUpdates(self, destWeights, destBiases, curWeights, curBiases, WGrads, biasGrads):
        for i in range(len(destWeights)):
            destWeights[i] = curWeights[i] + WGrads[i]
            destBiases[i] = curBiases[i] + biasGrads[i]

    def stepNesterov(self, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False):
        mbsz = inputBatch.shape[0]
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)

        curWeights = [w.copy() for w in self.weights]
        curBiases = [b.copy() for b in self.biases]
        self.scaleDerivs(momentum)
        self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)

        errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout)

        #self.scaleDerivs(momentum)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.WGrads[i] += learnRates[i]*(WGrad/mbsz - L2Costs[i]*self.weights[i])
            self.biasGrads[i] += (learnRates[i]/mbsz)*biasGrad

        self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)
        self.constrainWeights()
        return error, outputActs

    def gradDebug(self, inputBatch, targetBatch):
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)


        mbsz = inputBatch.shape[0]
        outputActs = self.fprop(inputBatch)
        outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, self.state[-1], outputActs)
        #error = self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
        errSignals = self.bprop(outputErrSignal)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            #update the weight increments
            self.WGrads[i] = WGrad
            self.biasGrads[i] = biasGrad
        allWeightGrads = itertools.chain(self.WGrads, self.biasGrads)
        return gnp.as_numpy_array(gnp.concatenate([dw.ravel() for dw in allWeightGrads]))

    def fprop(self, inputBatch, weightsToStopBefore = None ):
        """
        Perform a (possibly partial) forward pass through the
        network. Updates self.state which, on a full forward pass,
        holds the input followed by each hidden layer's activation and
        finally the net input incident on the output layer. For a full
        forward pass, we return the actual output unit activations. In
        a partial forward pass we return None.
        """
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        if weightsToStopBefore == None:
            weightsToStopBefore = len(self.weights)
        #self.state holds everything before the output nonlinearity, including the net input to the output units
        self.state = [inputBatch]
        for i in range(min(len(self.weights) - 1, weightsToStopBefore)):
            curActs = self.hidActFuncts[i].activation(gnp.dot(self.state[-1], self.weights[i]) + self.biases[i])
            self.state.append(curActs)
        if weightsToStopBefore >= len(self.weights):
            self.state.append(gnp.dot(self.state[-1], self.weights[-1]) + self.biases[-1])
            self.acts = self.outputActFunct.activation(self.state[-1])
            return self.acts
        #we didn't reach the output units
        # To return the first set of hidden activations, we would set
        # weightsToStopBefore to 1.
        return self.state[weightsToStopBefore]

    def fpropDropout(self, inputBatch, weightsToStopBefore = None ):
        """
        Perform a (possibly partial) forward pass through the
        network. Updates self.state which, on a full forward pass,
        holds the input followed by each hidden layer's activation and
        finally the net input incident on the output layer. For a full
        forward pass, we return the actual output unit activations. In
        a partial forward pass we return None.
        """
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        if weightsToStopBefore == None:
            weightsToStopBefore = len(self.weights)
        #self.state holds everything before the output nonlinearity, including the net input to the output units
        self.state = [inputBatch * (gnp.rand(*inputBatch.shape) > self.dropouts[0])]
        for i in range(min(len(self.weights) - 1, weightsToStopBefore)):
            dropoutMultiplier = 1.0/(1.0-self.dropouts[i])
            curActs = self.hidActFuncts[i].activation(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[i]) + self.biases[i])
            self.state.append(curActs * (gnp.rand(*curActs.shape) > self.dropouts[i+1]) )
        if weightsToStopBefore >= len(self.weights):
            dropoutMultiplier = 1.0/(1.0-self.dropouts[-1])
            self.state.append(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[-1]) + self.biases[-1])
            self.acts = self.outputActFunct.activation(self.state[-1])
            return self.acts
        #we didn't reach the output units
        # To return the first set of hidden activations, we would set
        # weightsToStopBefore to 1.
        return self.state[weightsToStopBefore]

    def bprop(self, outputErrSignal, fpropState = None):
        """
        Perform a backward pass through the network. fpropState
        defaults to self.state (set during fprop) and outputErrSignal
        should be self.outputActFunct.dErrordNetInput(...).
        """
        #if len(errSignals)==len(self.weights)==len(self.biases)==h+1 then
        # len(fpropState) == h+2 because it includes the input and the net input to the output layer and thus
        #fpropState[-2] is the activation of the penultimate hidden layer (or the input if there are no hidden layers)
        if fpropState == None:
            fpropState = self.state
        assert(len(fpropState) == len(self.weights) + 1)

        errSignals = [None for i in range(len(self.weights))]
        errSignals[-1] = outputErrSignal
        for i in reversed(range(len(self.weights) - 1)):
            errSignals[i] = gnp.dot(errSignals[i+1], self.weights[i+1].T)*self.hidActFuncts[i].dEdNetInput(fpropState[i+1])
        return errSignals

    def gradients(self, fpropState, errSignals):
        """
        Lazily generate (negative) gradients for the weights and biases given
        the result of fprop (fpropState) and the result of bprop
        (errSignals).
        """
        assert(len(fpropState) == len(self.weights)+1)
        assert(len(errSignals) == len(self.weights) == len(self.biases))
        for i in range(len(self.weights)):
            yield gnp.dot(fpropState[i].T, errSignals[i]), errSignals[i].sum(axis=0)



