This package contains python code for pre-trained deep neural networks (sometimes called deep belief networks in an abuse of terminology that I too have been guilty of). RBM pre-training is supported and backpropagation. There are a handful of possible unit types. For training, minibatched stochastic gradient descent is implemented. This initial release has essentially no documentation other than what exists in the code itself (which is very little), but it is small enough so hopefully someone familiar with the algorithms and with python could use it. Of course such a person could probably write their own software.

This initial release only has the barest essentials of features. My internal version has more features, but some of them have not been published yet and thus have been stripped away from the release version. After all the features I have implelmented in my internal code have been made public, I plan on doing a slightly better release, perhaps even with some documentation.



Dependencies
gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html)
and one of
cudamat (http://code.google.com/p/cudamat/) or
npmat (http://www.cs.toronto.edu/~ilya/npmat.py), a non-gpu cudamat stand-in.


Running the Example (mnistExample.py)

Download the gzipped data from http://www.cs.toronto.edu/~gdahl/mnist.npz.gz and unzip it into the same folder as all of the code (or change the line 
f = num.load("mnist.npz")
in mnistExample.py. Then you should be able to run the example with
$ python mnistExample.py
assuming you have obtained all the dependencies for 