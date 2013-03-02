import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class Dataset_2(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(Dataset_2('test_join'))
        return suite

    def test_join(self):
        '''
        Tests:
            * Join of different datasets
        '''
        self.setUpData()
        ds_all = copper.Dataset('dataset/2/data.csv')
        df = copper.read_csv('dataset/2/data.csv')

        ds1 = copper.Dataset(df.ix[:, 0:2])
        ds2 = copper.Dataset(df.ix[:, 2:5])
        ds3 = copper.Dataset(df.ix[:, 5:7])
        ds4 = copper.Dataset(df.ix[:, 7:11])

        ds = copper.join(ds1, ds2, others=[ds3, ds4])
        self.assertEqual(ds, ds_all)

        # 2. Change value of a section, the change should be reflected on the joined
        ds2.type['Money.1'] = ds.NUMBER
        ds2.update()
        ds_all.type['Money.1'] = ds.NUMBER
        ds_all.update()

        ds = copper.join(ds1, ds2, others=[ds3, ds4])
        self.assertEqual(ds, ds_all)


if __name__ == '__main__':
    # unittest.main()
    suite = Dataset_2().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)





