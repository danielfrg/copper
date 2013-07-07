import os
import copper
import numpy as np
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class UtilsFrame(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        # suite.addTest(UtilsFrame('test_percent_missing'))
        # suite.addTest(UtilsFrame('test_unique_values'))
        # suite.addTest(UtilsFrame('test_outliers'))
        suite.addTest(UtilsFrame('test_pca'))
        # suite.addTest(UtilsFrame('test_feature_weight'))
        # suite.addTest(UtilsFrame('test_rce_rank'))
        return suite

    def test_percent_missing(self):
        frame = pd.DataFrame({  'Col.1': np.random.rand(100), 
                                'Col.2':  np.random.rand(100)})
        ds = copper.Dataset(frame)

        frame['Col.1'][0:10] = np.nan
        missing = copper.utils.frame.percent_missing(frame)
        self.assertEqual(missing['Col.2'], 0, digits=8)
        self.assertEqual(missing['Col.1'], 0.1, digits=8)
        self.assertEqual(ds.percent_missing()['Col.2'], 0, digits=8)
        self.assertEqual(ds.percent_missing()['Col.1'], 0.1, digits=8)

        frame['Col.2'][0:23] = np.nan
        missing = copper.utils.frame.percent_missing(frame)
        self.assertEqual(missing['Col.2'], 0.23, digits=8)
        self.assertEqual(missing['Col.1'], 0.1, digits=8)
        self.assertEqual(ds.percent_missing()['Col.2'], 0.23, digits=8)
        self.assertEqual(ds.percent_missing()['Col.1'], 0.1, digits=8)

        frame['Col.1'][0:35] = np.nan
        missing = copper.utils.frame.percent_missing(frame)
        self.assertEqual(missing['Col.2'], 0.23, digits=8)
        self.assertEqual(missing['Col.1'], 0.35, digits=8)
        self.assertEqual(ds.percent_missing()['Col.2'], 0.23, digits=8)
        self.assertEqual(ds.percent_missing()['Col.1'], 0.35, digits=8)
        
        frame['Col.1'][:] = np.nan
        missing = copper.utils.frame.percent_missing(frame)
        self.assertEqual(missing['Col.2'], 0.23, digits=8)
        self.assertEqual(missing['Col.1'], 1, digits=8)
        self.assertEqual(ds.percent_missing()['Col.2'], 0.23, digits=8)
        self.assertEqual(ds.percent_missing()['Col.1'], 1, digits=8)
        
    def test_unique_values(self):
        dic = { 0: np.ones(5),
                1: np.random.rand(5),
                2: [0,0,1,1,1],
                3: [0,0,2,1,1]}
        df = pd.DataFrame(dic)
        unique = copper.utils.frame.unique_values(df)
        self.assertEqual(unique[0], 1)
        self.assertEqual(unique[1], 5)
        self.assertEqual(unique[2], 2)
        self.assertEqual(unique[3], 3)

    def test_outliers(self):
        np.random.seed(123)
        dic = { 0: np.random.randn(20),
                1: np.random.randn(20),
                2: np.random.randn(20) }
        df = pd.DataFrame(dic)
        df[0][5] = 1000
        df[0][2] = 1000
        df[1][1] = 2000
        df[1][4] = 3000
        df[1][7] = 5000

        # Outlier rows
        outlier_rows = copper.utils.frame.outlier_rows(df)
        self.assertEqual(outlier_rows[0][2], True)
        self.assertEqual(outlier_rows[0][5], True)
        self.assertEqual(outlier_rows[1][1], True)
        self.assertEqual(outlier_rows[1][4], True)
        self.assertEqual(outlier_rows[1][7], True)

        # Outlier count: Series
        self.assertEqual(copper.utils.frame.outlier_count(df[0]), 2)
        self.assertEqual(copper.utils.frame.outlier_count(df[1]), 3)
        self.assertEqual(copper.utils.frame.outlier_count(df[2]), 0)

        # Outlier count: DataFrame
        outlier_count = copper.utils.frame.outlier_count(df)
        self.assertEqual(outlier_count[0], 2)
        self.assertEqual(outlier_count[1], 3)
        self.assertEqual(outlier_count[2], 0)

        # Outliers 
        outliers = copper.utils.frame.outliers(df[0])
        self.assertEqual(outliers.index[0], 2)
        self.assertEqual(outliers.index[1], 5)
        outliers = copper.utils.frame.outliers(df[1])
        self.assertEqual(outliers.index[0], 1)
        self.assertEqual(outliers.index[1], 4)
        self.assertEqual(outliers.index[2], 7)
        self.assertEqual(outliers[1], 2000)
        self.assertEqual(outliers[4], 3000)
        self.assertEqual(outliers[7], 5000)

    def test_pca(self):
        np.random.seed(123)
        index = np.arange(5,10)
        df = pd.DataFrame(np.random.randn(5, 5), index=index)

        pca = copper.utils.frame.PCA(df, n_components=2)
        self.assertEqual(len(pca.columns), 2)
        self.assertEqual(list(pca.index), list(df.index))
        self.assertEqual(pca[0][5], -1.308208, 5)
        self.assertEqual(pca[0][8], -2.130776, 5)
        self.assertEqual(pca[1][7], -1.377485, 5)

        pca = copper.utils.frame.PCA(df, n_components=4)
        self.assertEqual(len(pca.columns), 4)
        self.assertEqual(list(pca.index), list(df.index))
        self.assertEqual(pca[0][5], -1.308208, 5)
        self.assertEqual(pca[1][8], 1.077598, 5)
        self.assertEqual(pca[3][9], -0.112868, 5)



        # values

    def test_feature_weight(self):
        np.random.seed(123)
        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series(np.round(np.random.rand(10, )), name='Target')
        
        ds = copper.Dataset(X.join(y))
        weights = copper.utils.frame.features_weight(X,y)
        self.assertEqual(weights[0],0.181199, 5)
        self.assertEqual(weights[1],0.410150, 5)
        self.assertEqual(weights[2],0.452251, 5)
        self.assertEqual(weights[3],0.545589, 5)
        self.assertEqual(weights[4],1.000000, 5)

    def test_rce_rank(self):
        np.random.seed(123)
        X = pd.DataFrame(np.random.randn(10, 2))
        y = pd.Series(np.round(np.random.rand(10, )), name='Target')
        
        ds = copper.Dataset(X.join(y))
        weights = copper.utils.frame.rce_rank(X,y)
        
        self.assertEqual(weights[0],1)
        self.assertEqual(weights[1],2)

if __name__ == '__main__':
    suite = UtilsFrame().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)





