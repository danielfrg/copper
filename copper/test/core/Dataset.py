import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class DatasetTest(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(DatasetTest('test_pandas_1'))
        suite.addTest(DatasetTest('test_types_1'))
        suite.addTest(DatasetTest('test_roles_1'))
        suite.addTest(DatasetTest('test_inputs_1'))
        suite.addTest(DatasetTest('test_inputs_2'))
        suite.addTest(DatasetTest('test_inputs_3'))
        return suite

    def test_pandas_1(self):
        '''
        Test basic functionality of pandas
        '''
        self.setUpData()

        ds = copper.read_csv('dataset/pandas1/data.csv')
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

    def test_types_1(self):
        self.setUpData()
        ds = copper.read_csv('dataset/types1/data.csv')
        sol = pd.read_csv(os.path.join(copper.project.data, 'dataset/types1/solution.csv'))
        sol = sol.set_index('column')
        self.assertEqual(ds.type, sol['Type'])

    def test_roles_1(self):
        self.setUpData()
        ds = copper.read_csv('dataset/roles1/data.csv')
        self.assertEqual(ds.role.values, ds.frame.ix[0].values)

    def test_inputs_1(self):
        self.setUpData()

        ds = copper.Dataset()
        ds.load('dataset/inputs1/data.csv')
        metadata = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs1/metadata.csv'))
        metadata = metadata.set_index('column')
        self.assertEqual(ds.metadata, metadata)

        frame = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs1/frame.csv'))
        self.assertEqual(ds.frame, frame)

        inputs = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs1/inputs.csv'))
        self.assertEqual(ds.inputs, inputs)

        target = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs1/target.csv'))
        self.assertEqual(ds.target, target['Target'])

    def test_inputs_2(self):
        '''
        Tests
        -----
            1. Load Data: two columns
            2. Default Metadata
                2.1 Roles: Two inputs
                2.2 Types: Number and Category
            3. Default inputs: Category col is transformed into 4 columns
            4. Change the type of a column from Category to Number
                4.1 New Metadata
                4.2 New Inputs (fewer columns)
        '''
        self.setUpData()

        ds = copper.Dataset()
        ds.load('dataset/inputs2/data.csv')

        metadata = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs2/metadata1.csv'))
        metadata = metadata.set_index('column')
        self.assertEqual(ds.metadata, metadata)

        inputs = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs2/inputs1.csv'))
        self.assertEqual(ds.inputs, inputs)

        # -- Change the 'Category' col to 'Number'
        ds.type['Number as Category'] = ds.CATEGORY

        metadata = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs2/metadata2.csv'))
        metadata = metadata.set_index('column')
        self.assertEqual(ds.metadata, metadata)

        inputs = pd.read_csv(os.path.join(copper.project.data, 'dataset/inputs2/inputs2.csv'))
        self.assertEqual(ds.inputs, inputs)

    def test_inputs_3(self):
        '''
        This test compares the iris dataset from scikit-learn to the produced by
        copper
        '''
        self.setUpData()
        ds = copper.read_csv('dataset/iris/data.csv')
        ds.role['class'] = ds.TARGET

        from sklearn import datasets
        iris = datasets.load_iris()

        self.assertEqual(ds.inputs.values, iris.data)
        self.assertEqual(ds.target.values, iris.target)

    def test_save_load(self):
        '''
        Tests
        -----
            1. Load Data
            2. Save Data
            3. Load saved Data
            4. Test that loaded data is the same as the saved data
        '''
        self.setUpData()

        ds1 = copper.Dataset()
        ds1.load('dataset/test1/data.csv')

        ds1.save('data_saved')

        ds2 = copper.load('data_saved')
        self.assertEqual(ds1.metadata, ds2.metadata)
        self.assertEqual(ds1.frame, ds2.frame)
        self.assertEqual(ds1.inputs, ds2.inputs)

if __name__ == '__main__':
    suite = DatasetTest().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
