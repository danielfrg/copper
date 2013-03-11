import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class Dataset_1(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(Dataset_1('test_cat2num'))
        suite.addTest(Dataset_1('test_fillna'))
        suite.addTest(Dataset_1('test_fillna_2'))
        suite.addTest(Dataset_1('test_filter'))
        return suite

    def test_cat2num(self):
        '''
        Tests the automatic transformation of a Category to Number.
        More tests can be found on the tranformation tests.
        '''
        self.setUpData()
        ds = copper.Dataset('dataset/1/data.csv')
        sol = copper.read_csv('dataset/1/transformed.csv')

        # Test the imported metadata
        self.assertEqual(ds.type['Number.1'], ds.NUMBER)
        self.assertEqual(ds.type['Number.2'], ds.NUMBER)
        self.assertEqual(ds.type['Cat.1'], ds.CATEGORY)
        self.assertEqual(ds.type['Cat.2'], ds.CATEGORY)
        self.assertEqual(ds.type['Num.as.Cat'], ds.CATEGORY)
        self.assertEqual(ds.type['Money'], ds.CATEGORY)

        # Change test 1
        ds.type['Num.as.Cat'] = ds.NUMBER
        self.assertEqual(ds.type['Num.as.Cat'], ds.NUMBER)
        ds.update()
        self.assertEqual(ds['Num.as.Cat'], sol['Num.as.Cat'])

        # Change test 2:
        ds.type['Money'] = ds.NUMBER
        self.assertEqual(ds.type['Money'], ds.NUMBER)
        ds.update()
        self.assertEqual(ds['Money'], sol['Money'])

    def test_fillna(self):
        '''
        Tests:
            * Fill na in type=Number
            * Fill na in type=Category
        '''
        self.setUpData()
        ds = copper.Dataset('dataset/1/data.csv')
        sol = copper.read_csv('dataset/1/transform_filled.csv')

        # Number.1 does not have missing values
        prev = ds['Number.1']
        ds.fillna(cols='Number.1', method='mean')
        self.assertEqual(ds['Number.1'], sol['Number.1'])

        # Number.2 does have missing values
        ds.fillna(cols='Number.2', method='mean')
        self.assertEqual(ds['Number.2'], sol['Number.2'])

        # Cat.1 does have missing values
        ds.fillna(cols='Cat.1', method='mode')
        self.assertEqual(ds['Cat.1'], sol['Cat.1'])

        # Cat.2 does NOT have missing values
        ds.fillna(cols='Cat.2', method='mode')
        self.assertEqual(ds['Cat.2'], sol['Cat.2'])

    def test_fillna_2(self):
        '''
        Tests the fill of all columns at once.
        '''

        self.setUpData()
        ds = copper.Dataset('dataset/1/data.csv')
        sol = copper.read_csv('dataset/1/transform_filled.csv')

        ds.type['Num.as.Cat'] = ds.NUMBER
        ds.type['Money'] = ds.NUMBER
        ds.update()

        ds.fillna(method='mean')
        self.assertEqual(ds.frame, sol)

    def test_filter(self):
        '''
        Tests: filter
        '''
        self.setUpData()
        ds = copper.Dataset('dataset/1/data.csv')
        df = copper.read_csv('dataset/1/data.csv')

        # 1. Initial frame
        self.assertEqual(ds.frame, df)

        # 2. No filters - Return everything
        self.assertEqual(ds.filter(), df)
        # 2.1 Reject a column but still no filters
        ds.role['Number.2'] = ds.REJECTED
        self.assertEqual(ds.filter(), df)

        # 3. Filter by inputs
        ds.role['Number.2'] = ds.REJECTED
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != 'Number.2'])
        # 3.1 Put the column back
        ds.role['Number.2'] = ds.INPUT
        self.assertEqual(ds.filter(role=ds.INPUT), df)

        # 4. Filter by Target - Inputs changed
        ds.role['Cat.1'] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.TARGET), df[['Cat.1']])
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != 'Cat.1'])

        # 5. Filter by type
        self.assertEqual(ds.filter(type=ds.NUMBER), df[['Number.1', 'Number.2']])
        self.assertEqual(ds.filter(type=ds.CATEGORY), df[['Cat.1', 'Cat.2', 'Num.as.Cat', 'Money']])

        # 6. Filter by role and type
        ds.role['Cat.1'] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df[['Number.1', 'Number.2']])
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.CATEGORY), df[['Cat.2', 'Num.as.Cat', 'Money']])
        self.assertEqual(ds.filter(role=ds.TARGET, type=ds.CATEGORY), df[['Cat.1']])

        # Multiple roles
        self.assertEqual(ds.filter(role=[ds.INPUT, ds.TARGET]), df)

        # Multiple types
        self.assertEqual(ds.filter(type=[ds.NUMBER, ds.CATEGORY]), df)

        # Multiple roles and types
        self.assertEqual(ds.filter(role=[ds.INPUT, ds.TARGET], type=[ds.NUMBER, ds.CATEGORY]), df)


if __name__ == '__main__':
    # unittest.main()
    suite = Dataset_1().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)





