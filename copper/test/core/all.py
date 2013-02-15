import unittest

import copper.test.core.Dataset_all as Dataset

def suite():
    suite = unittest.TestSuite()
    suite.addTest(Dataset.suite())
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
