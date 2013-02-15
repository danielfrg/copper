import unittest

from copper.test.core.Dataset_pandas import Dataset_pandas
from copper.test.core.Dataset_1 import Dataset_1

def suite():
    suite = unittest.TestSuite()
    suite.addTest(Dataset_pandas().suite())
    suite.addTest(Dataset_1().suite())
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
