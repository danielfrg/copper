import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class Transforms_ML(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(Transforms_ML('test_1'))
        return suite

    def test_1(self):
        '''
        Tests Numberic and category inputs2ml
        Tets Numberic and categorical targets
        '''
        self.setUpData()
        ds = copper.Dataset('transforms/ml/data.csv')

        # Test: Inputs: Numeric and categorical
        ds.type['Num.as.Cat'] = ds.CATEGORY
        ds.role['Target.Num'] = ds.REJECTED
        ds.role['Target.Cat'] = ds.REJECTED
        sol = copper.read_csv('transforms/ml/ml.csv')
        del sol['Target.Num']
        del sol['Target.Cat']

        self.assertEqual(copper.transform.inputs2ml(ds), sol)

        # Tests: Numeric target
        sol = copper.read_csv('transforms/ml/ml.csv')['Target.Num']
        ds.role['Target.Num'] = ds.TARGET
        ds.role['Target.Cat'] = ds.REJECTED
        self.assertEqual(copper.transform.target2ml(ds), sol)

        # Tests: Categorical target
        sol = copper.read_csv('transforms/ml/ml.csv')['Target.Cat']
        ds.role['Target.Num'] = ds.REJECTED
        ds.role['Target.Cat'] = ds.TARGET
        self.assertEqual(copper.transform.target2ml(ds), sol)


if __name__ == '__main__':
    suite = Transforms_ML().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
