import os
import gevent
from gevent.pool import Group

def talk(msg):
    for i in xrange(3):
        print(msg, os.getpid())

g1 = gevent.spawn(talk, 'bar')
g2 = gevent.spawn(talk, 'foo')
g3 = gevent.spawn(talk, 'fizz')

group = Group()
group.add(g1)
group.add(g2)
group.join()

group.add(g3)
group.join()