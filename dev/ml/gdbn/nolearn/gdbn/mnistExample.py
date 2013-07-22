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
import itertools
from dbn import *

def numMistakes(targetsMB, outputs):
    if not isinstance(outputs, num.ndarray):
        outputs = outputs.as_numpy_array()
    if not isinstance(targetsMB, num.ndarray):
        targetsMB = targetsMB.as_numpy_array()
    return num.sum(outputs.argmax(1) != targetsMB.argmax(1))

def sampleMinibatch(mbsz, inps, targs):
    idx = num.random.randint(inps.shape[0], size=(mbsz,))
    return inps[idx], targs[idx]

def main():
    mbsz = 64
    layerSizes = [784, 512, 512, 10]
    scales = [0.05 for i in range(len(layerSizes)-1)]
    fanOuts = [None for i in range(len(layerSizes)-1)]
    learnRate = 0.1
    epochs = 10
    mbPerEpoch = int(num.ceil(60000./mbsz))
    
    f = num.load("mnist.npz")
    trainInps = f['trainInps']/255.
    testInps = f['testInps']/255.
    trainTargs = f['trainTargs']
    testTargs = f['testTargs']

    assert(trainInps.shape == (60000, 784))
    assert(trainTargs.shape == (60000, 10))
    assert(testInps.shape == (10000, 784))
    assert(testTargs.shape == (10000, 10))

    mbStream = (sampleMinibatch(mbsz, trainInps, trainTargs) for unused in itertools.repeat(None))
    
    net = buildDBN(layerSizes, scales, fanOuts, Softmax(), False)
    net.learnRates = [learnRate for x in net.learnRates]
    net.L2Costs = [0 for x in net.L2Costs]
    net.nestCompare = True #this flag existing is a design flaw that I might address later, for now always set it to True
    
    for ep, (trCE, trEr) in enumerate(net.fineTune(mbStream, epochs, mbPerEpoch, numMistakes, True)):
        print ep, trCE, trEr
    

    
    

    
    


if __name__ == "__main__":
    main()
