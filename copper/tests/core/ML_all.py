import unittest

from copper.test.core.ML_1 import ML_1

def suite():
    suite = unittest.TestSuite()
    suite.addTest(ML_1().suite())
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
