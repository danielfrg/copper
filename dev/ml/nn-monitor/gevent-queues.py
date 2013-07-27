import os
import gevent
from gevent.queue import Queue

tasks = Queue()

class MyAlgorithm(object):

    def __init__(self):
        self.n = 0

    def fit(self):
        print "Fit", os.getpid()
        while self.n < 5:
            self.n += 1
            gevent.sleep(1)
            tasks.put_nowait(self.n)

g = MyAlgorithm()

def setter():
    g.fit()

def waiter():
    print "Print", os.getpid()
    while g.n < 5:
        print tasks.get()
        print os.getpid()

gevent.joinall([
    gevent.spawn(setter),
    gevent.spawn(waiter),
])
