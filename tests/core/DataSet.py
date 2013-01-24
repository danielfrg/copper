import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class DataSetTest(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(DataSetTest('test_1'))
        suite.addTest(DataSetTest('test_2'))
        suite.addTest(DataSetTest('test_save_load'))
        return suite

    def test_1(self):
        self.setUpData()

        ds = copper.DataSet()
        ds.load('dataset/test1/data.csv')
        metadata = pd.read_csv(os.path.join(copper.config.data, 'dataset/test1/metadata.csv'))
        metadata = metadata.set_index('column')
        self.assertEqual(ds.metadata, metadata)

        frame = pd.read_csv(os.path.join(copper.config.data, 'dataset/test1/frame.csv'))
        self.assertEqual(ds.frame, frame)

        encoded = pd.read_csv(os.path.join(copper.config.data, 'dataset/test1/encoded.csv'))
        self.assertEqual(ds.gen_frame(encodeCategory=True), encoded)

        inputs = pd.read_csv(os.path.join(copper.config.data, 'dataset/test1/inputs.csv'))
        self.assertEqual(ds.inputs, inputs)

        target = pd.read_csv(os.path.join(copper.config.data, 'dataset/test1/target.csv'))
        self.assertEqual(ds.target, target)

    def test_2(self):
        self.setUpData()

        ds = copper.DataSet()
        ds.load('dataset/test2/data.csv')

        metadata = pd.read_csv(os.path.join(copper.config.data, 'dataset/test2/metadata1.csv'))
        metadata = metadata.set_index('column')
        self.assertEqual(ds.metadata, metadata)

        inputs = pd.read_csv(os.path.join(copper.config.data, 'dataset/test2/inputs1.csv'))
        self.assertEqual(ds.inputs, inputs)

        # -- Change the 'Category' col to 'Number'
        ds.type['Number as Category'] = ds.NUMBER

        metadata = pd.read_csv(os.path.join(copper.config.data, 'dataset/test2/metadata2.csv'))
        metadata = metadata.set_index('column')
        self.assertEqual(ds.metadata, metadata)

        inputs = pd.read_csv(os.path.join(copper.config.data, 'dataset/test2/inputs2.csv'))
        self.assertEqual(ds.inputs, inputs)

        encoded = ds.gen_frame(encodeCategory=True)
        self.assertEqual(encoded, inputs)

    def test_save_load(self):
        self.setUpData()

        ds1 = copper.DataSet()
        ds1.load('dataset/test1/data.csv')

        ds1.save('data_saved')

        ds2 = copper.load('data_saved')
        self.assertEqual(ds1.metadata, ds2.metadata)
        self.assertEqual(ds1.frame, ds2.frame)
        self.assertEqual(ds1.inputs, ds2.inputs)

if __name__ == '__main__':
    suite = DataSetTest().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
