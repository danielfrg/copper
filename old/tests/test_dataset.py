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
        # suite.addTest(Dataset_1('test_properties'))
        # suite.addTest(Dataset_1('test_pandas'))
        # suite.addTest(Dataset_1('test_update_cat2num'))
        # suite.addTest(Dataset_1('test_filter'))
        suite.addTest(Dataset_1('test_match'))
        # suite.addTest(Dataset_1('test_join'))
        # suite.addTest(Dataset_1('test_fillna'))
        # suite.addTest(Dataset_1('test_feature_wheight'))
        # suite.addTest(Dataset_1('test_rce_rank'))
        # suite.addTest(Dataset_1('test_pca'))
        return suite

    def test_create(self):
        '''  Tests  the different ways of creating a Dataset
        '''
        frame = pd.DataFrame(np.random.rand(5,5), index=np.arange(5))
        frame.index.name = 'index'
        frame.to_csv('/tmp/temp.csv')

        ds1 = copper.Dataset('/tmp/temp.csv')
        df = copper.read_csv('/tmp/temp.csv')
        ds2 = copper.Dataset(df)
        ds3 = copper.Dataset()
        ds3.load('/tmp/temp.csv')
        ds4 = copper.Dataset()
        ds4.frame = df

        self.assertEqual(ds1, ds2)
        self.assertEqual(ds2, ds3)
        self.assertEqual(ds3, ds4)

    def test_properties(self):
        ''' Tests the basic properties: ds.frame, ds.inputs, ds.target,
        ds.numbers, ds.category
        '''
        df = pd.DataFrame(np.random.rand(5,10), index=np.arange(5))
        ds = copper.Dataset(df)
        
        # Role
        inputs = [3,5,6,7]
        target = [1]
        ds.role[:] = ds.REJECTED

        self.assertEqual(ds.inputs, None)
        self.assertEqual(ds.target, None)

        ds.role[inputs] = ds.INPUT
        self.assertEqual(ds.inputs, ds[inputs])
        ds.role[target] = ds.TARGET
        self.assertEqual(ds.target, ds[target]) # Only return first target
        
        targets = [1,2]
        ds.role[targets] = ds.TARGET
        self.assertEqual(ds.target, ds[target])

        # Type
        self.assertEqual(ds.numerical, ds.frame)
        self.assertEqual(ds.categorical, None)
        ds.type[:] = ds.CATEGORY
        self.assertEqual(ds.numerical, None)

        numbers = [2,4,5,8,9]
        cats = [0,1,3,6,7]
        ds.type[numbers] = ds.NUMBER
        ds.type[cats] = ds.CATEGORY
        self.assertEqual(ds.numerical, ds[numbers])
        self.assertEqual(ds.categorical, ds[cats])

    def test_pandas(self):
        ''' Test basic functionality of pandas
            1. Get/Set columns
            2. Head/Tail
            3. Correlation matrix
        '''
        index = ['Column %s' % num for num in np.arange(10)]
        df = pd.DataFrame(np.random.rand(5,10), columns=index)
        ds = copper.Dataset(df)
        
        # 1.2 Set columns - already existing columns only
        ds['Column 1'] = ds['Column 1'] - 10
        df['Column 1'] = df['Column 1'] - 10
        self.assertEqual(df, ds.frame)

        fnc = lambda x: x ** 2 + 2*x + 12
        ds['Column 2'] = ds['Column 2'].apply(fnc)
        df['Column 2'] = df['Column 2'].apply(fnc)
        self.assertEqual(df, ds.frame)

        # 2. Head/Tail
        self.assertEqual(ds.head(), df.head())
        self.assertEqual(ds.head(13), df.head(13))
        self.assertEqual(ds.tail(), df.tail())
        self.assertEqual(ds.tail(9), df.tail(9))

        # 3. Correlation matrix
        self.assertEqual(ds.corr(), df.corr())

    def test_update_cat2num(self):
        ''' Tests the automatic transformation of a Category to Number.
        More tests can be found on the tranformation tests.
        '''
        rands = np.round(np.random.rand(5)) * 100
        money = ['$%d'%num for num in rands]
        dic = { 'Money': money,
                'Cat.1': ['A', 'B', 'A', 'A', 'B'],
                'Num.as.Cat': ['1', '0', '1', '0', '1']}
        df = pd.DataFrame(dic)
        ds = copper.Dataset(df)
        
        dic = { 'Money': rands,
                'Cat.1': [np.nan, np.nan, np.nan, np.nan, np.nan],
                'Num.as.Cat': [1, 0, 1, 0, 1]}
        sol = copper.Dataset(pd.DataFrame(dic))

        # Test the imported metadata
        self.assertEqual(ds.type['Cat.1'], ds.CATEGORY)
        self.assertEqual(ds.type['Num.as.Cat'], ds.CATEGORY)

        # Change
        ds.type[:] = ds.NUMBER
        self.assertEqual(ds.type['Money'], ds.NUMBER)
        self.assertEqual(ds.type['Cat.1'], ds.NUMBER)
        self.assertEqual(ds.type['Num.as.Cat'], ds.NUMBER)
        ds.update()
        self.assertEqual(ds['Cat.1'], sol['Cat.1'])
        self.assertEqual(ds['Num.as.Cat'], sol['Num.as.Cat'])
        self.assertEqual(ds['Money'], sol['Money'])

    def test_filter(self):
        ''' Tests different filters for a ds
        '''
        df = pd.DataFrame(np.random.rand(7,10))
        ds = copper.Dataset(df)

        # 1. Initial frame
        self.assertEqual(ds.frame, df)

        # 2. No filters - Return everything
        self.assertEqual(ds.filter(), df)
        # 2.1 Reject a column but still no filters
        ds.role[2] = ds.REJECTED
        self.assertEqual(ds.filter(), df)

        # 3. Filter by inputs
        # Reject one col
        ds.role[2] = ds.REJECTED
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != 2])
        # Reject another col
        ds.role[5] = ds.REJECTED
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, (df.columns != 2) & (df.columns != 5)])
        # Put one col back
        ds.role[2] = ds.INPUT
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != 5])
        # Put the other col back
        ds.role[5] = ds.INPUT
        self.assertEqual(ds.filter(role=ds.INPUT), df)

        # 4. Filter by Target
        # No targets
        self.assertEqual(ds.filter(role=ds.TARGET).empty, True)
        self.assertEqual(ds.target, None)
        # Add a target
        ds.role[3] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.TARGET), df[[3]])
        self.assertEqual(ds.target, df[[3]])
        self.assertEqual(ds.filter(role=ds.INPUT), df.ix[:, df.columns != 3])

        # 5. Filter by type
        self.assertEqual(ds.filter(type=ds.NUMBER), df)
        ds.type[1] = ds.CATEGORY
        self.assertEqual(ds.filter(type=ds.CATEGORY), df[[1]])
        ds.type[3] = ds.CATEGORY
        self.assertEqual(ds.filter(type=ds.CATEGORY), df[[1, 3]])
        self.assertEqual(ds.filter(type=ds.NUMBER), df.ix[:, (df.columns != 1) & (df.columns != 3)])

        # 6. Filter by role and type
        ds.type[1] = ds.CATEGORY
        ds.type[3] = ds.CATEGORY
        ds.role[3] = ds.TARGET
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.CATEGORY), df[[1]])
        self.assertEqual(ds.filter(role=ds.TARGET, type=ds.CATEGORY), df[[3]])
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df.ix[:, (df.columns != 1) & (df.columns != 3)])
        ds.type[1] = ds.NUMBER
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df.ix[:, (df.columns != 3)])
        
        ds.type[:] = ds.CATEGORY
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.CATEGORY), df.ix[:, (df.columns != 3)])
        ds.type[4] = ds.NUMBER
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df[[4]])
        ds.type[5] = ds.NUMBER
        ds.type[7] = ds.NUMBER
        self.assertEqual(ds.filter(role=ds.INPUT, type=ds.NUMBER), df[[4, 5, 7]])

        # Multiple roles
        self.assertEqual(ds.filter(role=[ds.INPUT, ds.TARGET]), df)

        # Multiple types
        self.assertEqual(ds.filter(type=[ds.NUMBER, ds.CATEGORY]), df)

        # Multiple roles and types
        self.assertEqual(ds.filter(role=[ds.INPUT, ds.TARGET], type=[ds.NUMBER, ds.CATEGORY]), df)
    
    def test_match(self):
        ''' Test the match function.
        Matches the metadata of another dataset.
        '''
        df = pd.DataFrame(np.random.randn(10, 10))
        df2 = pd.DataFrame(np.random.randn(10, 11))
        train = copper.Dataset(df)
        test = copper.Dataset(df2)

        train.role[1] = train.ID
        train.type[2] = train.CATEGORY
        train.role[4] = train.REJECT
        train.role[6] = train.REJECT
        train.type[7] = train.CATEGORY

        test.match(train)
        # Test roles
        self.assertEqual(test.role[0], train.role[0])
        self.assertEqual(test.role[1], train.role[1])
        self.assertEqual(test.role[2], train.role[2])
        self.assertEqual(test.role[3], train.role[3])
        self.assertEqual(test.role[4], train.role[4])
        self.assertEqual(test.role[5], train.role[5])
        self.assertEqual(test.role[6], train.role[6])
        self.assertEqual(test.role[7], train.role[7])
        self.assertEqual(test.role[8], train.role[8])
        self.assertEqual(test.role[9], train.role[9])
        self.assertEqual(test.role[10], test.REJECT)

        self.assertEqual(test.type[0], train.type[0])
        self.assertEqual(test.type[1], train.type[1])
        self.assertEqual(test.type[2], train.type[2])
        self.assertEqual(test.type[3], train.type[3])
        self.assertEqual(test.type[4], train.type[4])
        self.assertEqual(test.type[5], train.type[5])
        self.assertEqual(test.type[6], train.type[6])
        self.assertEqual(test.type[7], train.type[7])
        self.assertEqual(test.type[8], train.type[8])
        self.assertEqual(test.type[9], train.type[9])

    def test_join(self):
        ''' Tests join of different datasets
        '''
        array = np.random.randn(5, 8)
        full = pd.DataFrame(array)
        df1 = pd.DataFrame(array[:, 0:2], columns=[0,1])
        ds1 = copper.Dataset(df1)
        df2 = pd.DataFrame(array[:, 2:4], columns=[2,3])
        ds2 = copper.Dataset(df2)
        df3 = pd.DataFrame(array[:, 4:6], columns=[4,5])
        ds3 = copper.Dataset(df3)
        df4 = pd.DataFrame(array[:, 6:], columns=[6,7])
        ds4 = copper.Dataset(df4)
        
        # Simple Join
        ds = ds1.join(ds2).join(ds3).join(ds4)
        self.assertEqual(ds.frame, full)

        # Change value of a part, the change should be reflected on the joined
        ds1.type[1] = ds1.CATEGORY
        ds2.type[2] = ds2.CATEGORY
        ds = ds1.join(ds2)
        self.assertEqual(ds.type[1], ds.CATEGORY)
        self.assertEqual(ds.type[2], ds.CATEGORY)
        
        # Changes should mantain
        ds4.type[7] = ds4.CATEGORY
        ds = ds1.join(ds2).join(ds3).join(ds4)
        self.assertEqual(ds.type[1], ds.CATEGORY)
        self.assertEqual(ds.type[2], ds.CATEGORY)
        self.assertEqual(ds.type[7], ds.CATEGORY)

    def test_fillna(self):
        ''' Fill missing values of indivitual columns
        '''
        df = pd.DataFrame(np.random.randn(5, 5))
        ds = copper.Dataset(df)

        # Col 0 has no missing values
        ans_0 = df[0]
        ds.fillna(0, method='mean')
        self.assertEqual(ds[0], ans_0)
        
        # Col 1 has no missing values
        ds[1][0:3] = np.nan
        ans_1 = df[1].fillna(value=df[1].mean())
        ds.fillna(1, method='mean')
        self.assertEqual(ds[1], ans_1)

        # Fill the rest of the set
        ds[3][1:4] = np.nan
        ds[4][0:3] = np.nan
        ans_3 = df[3].fillna(value=df[3].mean())
        ans_4 = df[4].fillna(value=df[4].mean())
        ds.fillna(method='mean')
        self.assertEqual(ds[1], ans_1)
        self.assertEqual(ds[3], ans_3)

    # --------------------------------------------------------------------------
    #                             FRAME UTILITIES
    # --------------------------------------------------------------------------
        
    def test_feature_wheight(self):
        '''
        Only checks that the values matches the frame.utils values
        '''
        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series(np.round(np.random.rand(10, )), name='Target')
        
        ds = copper.Dataset(X.join(y))
        sol = copper.utils.frame.features_weight(X,y)
        self.assertEqual(ds.features_weight(), sol)

    def test_rce_rank(self):
        '''
        Only checks that the values matches the frame.utils values
        '''
        X = pd.DataFrame(np.random.randn(10, 2))
        y = pd.Series(np.round(np.random.rand(10, )), name='Target')

        ds = copper.Dataset(X.join(y))
        sol = copper.utils.frame.rce_rank(X,y)
        self.assertEqual(ds.rce_rank(), sol)

    def test_pca(self):
        '''
        Only checks that the values matches the frame.utils values
        '''
        X = pd.DataFrame(np.random.randn(100, 50))

        ds = copper.Dataset(X)
        ds.role[0]
        sol = copper.utils.frame.PCA(X, n_components=10)
        sol = copper.Dataset(sol)
        self.assertEqual(ds.PCA(n_components=10), sol)
        
if __name__ == '__main__':
    # unittest.main()
    suite = Dataset_1().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
