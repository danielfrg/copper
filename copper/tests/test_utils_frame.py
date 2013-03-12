import os
import copper
import numpy as np
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class UtilsFrame(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(UtilsFrame('test_percent_missing'))
        suite.addTest(UtilsFrame('test_unique_values'))
        return suite

    def test_percent_missing(self):
        frame = pd.DataFrame({ 'zeros': np.zeros(100), 'ones': np.ones(100)})
        ds = copper.Dataset(frame)

        frame['zeros'][0:10] = np.nan
        self.assertEqual(copper.utils.frame.percent_missing(frame)['ones'], 0, digits=8)
        self.assertEqual(copper.utils.frame.percent_missing(frame)['zeros'], 0.1, digits=8)
        self.assertEqual(ds.percent_missing()['ones'], 0, digits=8)
        self.assertEqual(ds.percent_missing()['zeros'], 0.1, digits=8)

        frame['ones'][0:23] = np.nan
        self.assertEqual(copper.utils.frame.percent_missing(frame)['ones'], 0.23, digits=8)
        self.assertEqual(copper.utils.frame.percent_missing(frame)['zeros'], 0.1, digits=8)
        self.assertEqual(ds.percent_missing()['ones'], 0.23, digits=8)
        self.assertEqual(ds.percent_missing()['zeros'], 0.1, digits=8)

        frame['zeros'][0:35] = np.nan
        self.assertEqual(copper.utils.frame.percent_missing(frame)['ones'], 0.23, digits=8)
        self.assertEqual(copper.utils.frame.percent_missing(frame)['zeros'], 0.35, digits=8)
        self.assertEqual(ds.percent_missing()['ones'], 0.23, digits=8)
        self.assertEqual(ds.percent_missing()['zeros'], 0.35, digits=8)
        
        frame['zeros'][:] = np.nan
        self.assertEqual(copper.utils.frame.percent_missing(frame)['ones'], 0.23, digits=8)
        self.assertEqual(copper.utils.frame.percent_missing(frame)['zeros'], 1, digits=8)
        self.assertEqual(ds.percent_missing()['ones'], 0.23, digits=8)
        self.assertEqual(ds.percent_missing()['zeros'], 1, digits=8)
        
    def test_unique_values(self):
        '''
        Uses pandas.value_counts so test are implicit ;)
        '''
        pass
        


if __name__ == '__main__':
    suite = UtilsFrame().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)





