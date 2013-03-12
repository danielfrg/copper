import os
import copper
import numpy as np
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class UtilsML(CopperTest):

    ml = None

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(UtilsML('test_bootstrap'))
        return suite

    def setup(self):
        if self.ml is None:
            self.setUpData()

            self.train = copper.Dataset('ml/1/train.csv')
            self.train.role['CustomerID'] = self.train.ID
            self.train.role['Order'] = self.train.TARGET
            fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
            self.train['LASD'] = self.train['LASD'].apply(fnc)

            self.test = copper.Dataset('ml/1/test.csv')
            self.test.role['CustomerID'] = self.test.ID
            self.test.role['Order'] = self.test.TARGET
            self.test['LASD'] = self.test['LASD'].apply(fnc)

            self.ml = copper.MachineLearning()
            self.ml.train = self.train
            self.ml.test = self.test

    def test_bootstrap(self):
        '''

        '''
        self.setup()

        from sklearn import tree
        bootstraped = copper.utils.ml.bootstrap(tree.DecisionTreeClassifier, 5, self.train, max_depth=6)
        self.ml.add_clfs(bootstraped, 'tree')
        self.assertEqual(len(self.ml.clfs), 5)

        from sklearn.naive_bayes import GaussianNB
        bootstraped = copper.utils.ml.bootstrap(GaussianNB, 5, self.train)
        self.ml.add_clfs(bootstraped, 'GNB')
        self.assertEqual(len(self.ml.clfs), 10)


if __name__ == '__main__':
    suite = UtilsML().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)





