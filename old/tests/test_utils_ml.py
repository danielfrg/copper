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

    def test_bootstrap(self):
        train = copper.Dataset(pd.DataFrame(np.random.rand(10,10)))
        train.role[0] = train.TARGET
        mc = copper.ModelComparison()
        mc.train = train

        from sklearn import tree
        dt = tree.DecisionTreeClassifier()
        trees = copper.utils.ml.bootstrap(dt, 5, train)
        mc.add_clfs(trees, 'tree')
        self.assertEqual(len(trees), 5)
        for t in trees:
            self.assertEqual(type(t), tree.DecisionTreeClassifier)

        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnbs = copper.utils.ml.bootstrap(gnb, 5, train)
        mc.add_clfs(gnbs, 'GNB')
        self.assertEqual(len(gnbs), 5)
        for gnb in gnbs:
            self.assertEqual(type(gnb), GaussianNB)

if __name__ == '__main__':
    suite = UtilsML().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)





