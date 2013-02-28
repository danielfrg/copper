import os
import copper
import pandas as pd

import unittest
from copper.test.CopperTest import CopperTest

class Dataset_1(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(Dataset_1('test_1_cat_2_num'))
        suite.addTest(Dataset_1('test_1_fillna'))
        suite.addTest(Dataset_1('test_1_role_ml'))
        suite.addTest(Dataset_1('test_1_filter'))
        return suite

    def test_1_cat_2_num(self):
        '''
        Tests:
            * Initial metadata
            * Automatic category to number transformation
                * metadata
                * most of the values are converted to number
                * values that cannot be converted become nan
        '''
        self.setUpData()
        ds = copper.Dataset('dataset/1/data.csv')
        sol = copper.read_csv('dataset/1/transform_filled.csv')

        self.assertEqual(ds.type['Number.1'], ds.NUMBER)
        self.assertEqual(ds.type['Number.2'], ds.NUMBER)
        self.assertEqual(ds.type['Cat.1'], ds.CATEGORY)
        self.assertEqual(ds.type['Cat.2'], ds.CATEGORY)
        self.assertEqual(ds.type['Num.as.Cat'], ds.CATEGORY)

        ds.type['Num.as.Cat'] = ds.NUMBER
        ds.update()
        # Test the metadata
        self.assertEqual(ds.type['Num.as.Cat'], ds.NUMBER)

        # Test the values
        self.assertEqual(ds['Num.as.Cat'], sol['Num.as.Cat'])

    def test_1_fillna(self):
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
        self.assertEqual(ds['Number.1'], prev)

        # Number.2 does have missing values
        ds.fillna(cols='Number.2', method='mean')
        self.assertEqual(ds['Number.2'], sol['Number.2'])

        # Cat.1 does have missing values
        ds.fillna(cols='Cat.1', method='mode')
        self.assertEqual(ds['Cat.1'], sol['Cat.1'])

    def test_1_role_ml(self):
        '''
        Depends on: test_1_fillna and test_1_cat_2_num

        Tests:
            1. Initial roles are Input
            2. Modify roles and returned frames are correct
                * Inputs are correct for machine learning
            3. Target is correct
        '''
        self.setUpData()
        ds = copper.Dataset('dataset/1/data.csv')
        df = copper.read_csv('dataset/1/data.csv')
        ml_df = copper.read_csv('dataset/1/ml.csv')

        # 1. Initial role
        self.assertEqual(ds.role['Number.1'], ds.INPUT)
        self.assertEqual(ds.role['Number.2'], ds.INPUT)
        self.assertEqual(ds.role['Cat.1'], ds.INPUT)
        self.assertEqual(ds.role['Cat.2'], ds.INPUT)
        self.assertEqual(ds.role['Num.as.Cat'], ds.INPUT)
        self.assertEqual(ds.frame, df)
        # Correct data
        ds.type['Num.as.Cat'] = ds.NUMBER
        ds.update()
        ds.fillna(method='mean')
        # 2. Inputs values are correct
        self.assertEqual(ds.inputs, ml_df)

        # 2.1 Modify roles
        ds.role['Number.1'] = ds.REJECTED
        self.assertEqual(ds.role['Number.1'], ds.REJECTED)
        self.assertEqual(ds.inputs, ml_df[ml_df.columns[1:]])

        # 2.2 Modify roles
        ds.role['Number.2'] = ds.REJECTED
        self.assertEqual(ds.role['Number.2'], ds.REJECTED)
        self.assertEqual(ds.inputs, ml_df[ml_df.columns[2:]])

        # 2.3 Modify roles
        ds.role['Cat.1'] = ds.REJECTED
        self.assertEqual(ds.role['Cat.1'], ds.REJECTED)
        self.assertEqual(ds.inputs, ml_df[ml_df.columns[4:]])

        # 2.4 Modify roles
        ds.role['Cat.2'] = ds.REJECTED
        self.assertEqual(ds.role['Cat.2'], ds.REJECTED)
        self.assertEqual(ds.inputs, ml_df[ml_df.columns[6:]])
        self.assertEqual(ds.inputs, ml_df[['Num.as.Cat']])

        # 3. Target
        # Back to normal
        ds.role['Number.1'] = ds.INPUT
        ds.role['Number.2'] = ds.INPUT
        ds.role['Cat.1'] = ds.INPUT
        ds.role['Cat.2'] = ds.INPUT
        self.assertEqual(ds.inputs, ml_df)
        # Set Target
        ds.role['Num.as.Cat'] = ds.TARGET
        self.assertEqual(ds.inputs, ml_df[ml_df.columns[:-1]])
        target = ml_df[ml_df.columns[-1]]
        target.name = 'Target'
        self.assertEqual(ds.target, target)

    def test_1_filter(self):
        '''
        Tests: Change of role types combination on filter
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
        self.assertEqual(ds.filter(role=ds.INPUT), df[['Number.1', 'Cat.1', 'Cat.2', 'Num.as.Cat']])
        # 3.1 Put the column back
        ds.role['Number.2'] = ds.INPUT
        self.assertEqual(ds.filter(role=ds.INPUT), df)

        # 4. Filter by Target - Inputs changed
        ds.role['Cat.1'] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.TARGET), df[['Cat.1']])
        self.assertEqual(ds.filter(role=ds.INPUT), df[['Number.1', 'Number.2', 'Cat.2', 'Num.as.Cat']])

        # 5. Filter by type
        self.assertEqual(ds.filter(type=ds.NUMBER), df[['Number.1', 'Number.2']])
        self.assertEqual(ds.filter(type=ds.CATEGORY), df[['Cat.1', 'Cat.2', 'Num.as.Cat']])

        # 6. Filter by role and type
        ds.role['Cat.1'] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df[['Number.1', 'Number.2']])
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.CATEGORY), df[['Cat.2', 'Num.as.Cat']])
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





