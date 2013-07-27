import os
import gevent
from gevent import Greenlet

class MyAlgorithm(object):

    def __init__(self):
        self.n = 0

    def fit(self):
        print "Fit", os.getpid()
        while self.n < 5:
            self.n += 1
            gevent.sleep(1)

g = MyAlgorithm()


def foo():
    g.fit()


def bar():
    print "Print", os.getpid()
    while g.n < 5:
        print g.n
        gevent.sleep(0.5)


gevent.joinall([
    gevent.spawn(foo),
    gevent.spawn(bar),
])

