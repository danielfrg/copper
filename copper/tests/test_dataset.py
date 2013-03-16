import os
import copper
import numpy as np
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class Dataset_1(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        # suite.addTest(Dataset_1('test_create'))
        # suite.addTest(Dataset_1('test_pandas'))
        # suite.addTest(Dataset_1('test_cat2num'))
        # suite.addTest(Dataset_1('test_fillna'))
        # suite.addTest(Dataset_1('test_fillna_2'))
        # suite.addTest(Dataset_1('test_join'))
        # suite.addTest(Dataset_1('test_filter'))
        suite.addTest(Dataset_1('test_match'))
        return suite

    def test_create(self):
        ''' 
        Different ways of creating a Dataset
        '''
        self.setUpData()
        ds1 = copper.Dataset('dataset/1/data.csv')
        
        df = copper.read_csv('dataset/1/data.csv')
        ds2 = copper.Dataset(df)

        ds3 = copper.Dataset()
        ds3.load('dataset/1/data.csv')

        self.assertEqual(ds1, ds2)
        self.assertEqual(ds2, ds3)

    def test_pandas(self):
        '''
        Test basic functionality of pandas
            1. Get/Set columns
            2. Head/Tail
            3. Correlation matrix
        '''
        self.setUpData()

        ds = copper.Dataset('dataset/1/data.csv')
        df = copper.read_csv('dataset/1/data.csv')

        # 1.1 Get columns
        for col in df.columns:
            self.assertEqual(ds[col], df[col])

        # 1.2 Set columns - already existing columns only
        ds['Number.1'] = ds['Number.1'] - 10
        df['Number.1'] = df['Number.1'] - 10
        self.assertEqual(df, ds.frame)

        fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
        ds['Date.Encoded'] = ds['Date.Encoded'].apply(fnc)
        df['Date.Encoded'] = df['Date.Encoded'].apply(fnc)
        self.assertEqual(df, ds.frame)

        # 2. Head/Tail
        self.assertEqual(ds.head(), df.head())
        self.assertEqual(ds.head(13), df.head(13))
        self.assertEqual(ds.tail(), df.tail())
        self.assertEqual(ds.tail(9), df.tail(9))

        # 3. Correlation matrix
        self.assertEqual(ds.corr(), df.corr())

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
        Tests the fill of all columns at once

        1. One column is REJECTED and therefore is not filled
        '''

        self.setUpData()
        ds = copper.Dataset('dataset/1/data.csv')
        sol = copper.read_csv('dataset/1/transform_filled.csv')

        ds.type['Num.as.Cat'] = ds.NUMBER
        ds.type['Money'] = ds.NUMBER
        ds.update()

        ds.fillna(method='mean')
        self.assertEqual(ds.frame, sol)

    def test_join(self):
        '''
        Tests join of different datasets
        '''
        self.setUpData()
        ds_all = copper.Dataset('dataset/1/data.csv')
        df = copper.read_csv('dataset/1/data.csv')

        l = len(df.columns)
        ds1 = copper.Dataset(df.ix[:, 0:int(l/4)])
        ds2 = copper.Dataset(df.ix[:, int(l/4):int(2*(l/4))])
        ds3 = copper.Dataset(df.ix[:, int(2*(l/4)):int(3*(l/4))])
        ds4 = copper.Dataset(df.ix[:, int(3*(l/4)):int(4*(l/4))])

        ds = copper.join(ds1, ds2, others=[ds3, ds4])
        self.assertEqual(ds, ds_all)

        # 2. Change value of a section, the change should be reflected on the joined
        # Note: increasing the data.csv file probably will have to change ds2 to ds1 below
        ds2.type['Cat.1'] = ds.NUMBER
        ds2.update()
        ds_all.type['Cat.1'] = ds.NUMBER
        ds_all.update()

        ds = copper.join(ds1, ds2, others=[ds3, ds4])
        self.assertEqual(ds, ds_all)

    def test_filter(self):
        '''
        Tests: filter
        '''
        df = pd.DataFrame({ '1': np.ones(10),
                            '2': np.ones(10),
                            '3': np.ones(10),
                            '4': np.ones(10),
                            '5': np.ones(10),
                            '6': np.ones(10),
                            '7': np.ones(10),
                            })
        ds = copper.Dataset(df)

        # 1. Initial frame
        self.assertEqual(ds.frame, df)

        # 2. No filters - Return everything
        self.assertEqual(ds.filter(), df)
        # 2.1 Reject a column but still no filters
        ds.role['2'] = ds.REJECTED
        self.assertEqual(ds.filter(), df)

        # 3. Filter by inputs
        # Reject one col
        ds.role['2'] = ds.REJECTED
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != '2'])
        # Reject another col
        ds.role['5'] = ds.REJECTED
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, (df.columns != '2') & (df.columns != '5')])
        # Put on col back
        ds.role['2'] = ds.INPUT
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != '5'])
        # Put the other col back
        ds.role['5'] = ds.INPUT
        self.assertEqual(ds.filter(role=ds.INPUT), df)

        # 4. Filter by Target
        # No targets
        self.assertEqual(ds.filter(role=ds.TARGET, ret_cols=True), [])
        # Add a target
        ds.role['3'] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.TARGET), df[['3']])
        self.assertEqual(ds.target, df['3']) # returns a series instead of a frame
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != '3'])

        # # 5. Filter by type
        self.assertEqual(ds.filter(type=ds.NUMBER), df)
        ds.type['1'] = ds.CATEGORY
        self.assertEqual(ds.filter(type=ds.CATEGORY), df[['1']])
        ds.type['3'] = ds.CATEGORY
        self.assertEqual(ds.filter(type=ds.CATEGORY), df[['1', '3']])
        self.assertEqual(ds.filter(type=ds.NUMBER), df.ix[:, (df.columns != '1') & (df.columns != '3')])

        # 6. Filter by role and type
        ds.type['1'] = ds.CATEGORY
        ds.type['3'] = ds.CATEGORY
        ds.role['3'] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.CATEGORY), df[['1']])
        self.assertEqual(ds.filter(role=ds.TARGET, type=ds.CATEGORY), df[['3']])
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df.ix[:, (df.columns != '1') & (df.columns != '3')])
        ds.type['1'] = ds.NUMBER
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df.ix[:, (df.columns != '3')])
        
        ds.type[:] = ds.CATEGORY
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.CATEGORY), df.ix[:, (df.columns != '3')])
        ds.type['4'] = ds.NUMBER
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df[['4']])
        ds.type['5'] = ds.NUMBER
        ds.type['7'] = ds.NUMBER
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df[['4', '5', '7']])

        # # Multiple roles
        self.assertEqual(ds.filter(role=[ds.INPUT, ds.TARGET]), df)

        # # Multiple types
        self.assertEqual(ds.filter(type=[ds.NUMBER, ds.CATEGORY]), df)

        # # Multiple roles and types
        self.assertEqual(ds.filter(role=[ds.INPUT, ds.TARGET], type=[ds.NUMBER, ds.CATEGORY]), df)
        
    def test_match(self):
        '''
        Test the match function of the dataset.
        Matches the metadata of other dataset.
        '''
        df = pd.DataFrame(np.random.randn(10, 10))
        train = copper.Dataset(df)
        test = copper.Dataset(df)

        train.role[1] = train.ID
        train.type[2] = train.CATEGORY
        train.role[4] = train.REJECT
        train.role[6] = train.REJECT
        train.type[7] = train.CATEGORY

        test.match(train)
        # Test indivitual changes
        self.assertEqual(test.role[1], test.ID)
        self.assertEqual(test.type[2], test.CATEGORY)
        self.assertEqual(test.role[4], test.REJECT)
        self.assertEqual(test.role[6], test.REJECT)
        self.assertEqual(test.type[7], test.CATEGORY)

        # Test whole metadata
        self.assertEqual(train.metadata, test.metadata)
        
if __name__ == '__main__':
    # unittest.main()
    suite = Dataset_1().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)





