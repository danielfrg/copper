import os
import copper
import pandas as pd

import unittest
from copper.test.CopperTest import CopperTest

class DatasetTest(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(DatasetTest('test_pandas_1'))
        # suite.addTest(DatasetTest('test_1'))
        return suite

    def test_pandas_1(self):
        '''
        Test basic functionality of pandas
        '''
        self.setUpData()

        ds = copper.Dataset()
        ds.load('dataset/pandas1/data.csv')
        df = pd.read_csv(os.path.join(copper.project.data, 'dataset/pandas1/data.csv'))

        self.assertEqual(ds['Number'], df['Number'])
        self.assertEqual(ds['Date'], df['Date'])

        ds['Number'] = ds['Number'] - 10
        df['Number'] = df['Number'] - 10
        self.assertEqual(df, ds.frame)

        fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
        ds['Date'] = ds['Date'].apply(fnc)
        df['Date'] = df['Date'].apply(fnc)
        self.assertEqual(df, ds.frame)

    def test_1(self):
        self.setUpData()
        ds = copper.read_csv('dataset/pandas1/data.csv')



if __name__ == '__main__':
    unittest.main()
    # suite = DatasetTest().suite()
    # unittest.TextTestRunner(verbosity=2).run(suite)
