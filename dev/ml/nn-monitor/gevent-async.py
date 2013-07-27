import gevent
from gevent.event import AsyncResult

a = AsyncResult()

class MyAlgorithm(object):

    def __init__(self):
        self.n = 0

    def fit(self):
        while self.n < 5:
            self.n += 1
            gevent.sleep(1)
            a.set()

g = MyAlgorithm()

def setter():
    g.fit()

def waiter():
    # while g.n < 5:
        a.get() # blocking
        print g.n
        # gevent.sleep(0.1)

gevent.joinall([
    gevent.spawn(setter),
    gevent.spawn(waiter),
])
