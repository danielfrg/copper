import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class DataSetTest(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(DataSetTest('test_1'))
        return suite

    def test_1(self):
        self.setUpData()

        ds = copper.load('dataset/test1/data.csv')
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


if __name__ == '__main__':
    suite = DataSetTest().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
