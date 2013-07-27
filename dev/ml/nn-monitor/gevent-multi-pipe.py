import os
import time
import gevent
from multiprocessing import Process, Pipe
from gevent.socket import wait_read, wait_write

a, b = Pipe()

class MyAlgorithm(object):

    def __init__(self):
        self.n = 0

    def fit(self):
        print "Fit", os.getpid()
        while self.n < 5:
            self.n += 1
            time.sleep(0.5)
            a.send(self.n)

def relay():
    g = MyAlgorithm()
    g.fit()

def get_msg():
    print "Print", os.getpid()
    while True:
        wait_read(b.fileno())
        print(b.recv())

if __name__ == '__main__':
    proc = Process(target=relay)
    proc.start()

    g1 = gevent.spawn(get_msg)
    gevent.joinall([g1], timeout=5)