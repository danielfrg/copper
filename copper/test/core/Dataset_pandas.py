import os
import copper
import pandas as pd

import unittest
from copper.test.CopperTest import CopperTest

class Dataset_pandas(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(Dataset_pandas('test_pandas_1'))
        return suite

    def test_pandas_1(self):
        '''
        Test basic functionality of pandas
            1. Get/Set columns
            2. Values
        '''
        self.setUpData()

        ds = copper.Dataset()
        ds.load('dataset/pandas1/data.csv')
        df = pd.read_csv(os.path.join(copper.project.data, 'dataset/pandas1/data.csv'))

        # 1. Get columns
        self.assertEqual(ds['Number'], df['Number'])
        self.assertEqual(ds['Date'], df['Date'])
        # 1. Set columns - already existing only
        ds['Number'] = ds['Number'] - 10
        df['Number'] = df['Number'] - 10
        self.assertEqual(df, ds.frame)

        fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
        ds['Date'] = ds['Date'].apply(fnc)
        df['Date'] = df['Date'].apply(fnc)
        self.assertEqual(df, ds.frame)

        # 2. Values
        self.assertEqual(ds.values, df.values)

if __name__ == '__main__':
    # unittest.main()
    suite = Dataset_pandas().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
