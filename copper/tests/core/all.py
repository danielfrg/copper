import unittest

from copper.test.CopperTest import CopperTest
import copper.test.core.Dataset_all as Dataset
import copper.test.core.ML_all as ML

class All(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(Dataset.suite())
        suite.addTest(ML.suite())
        return suite

if __name__ == '__main__':
    # unittest.main()
    unittest.TextTestRunner(verbosity=2).run(All().suite())
