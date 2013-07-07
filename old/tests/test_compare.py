import os
import copper
import numpy as np
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class ModelComparison(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(ModelComparison('test_models_list'))
        suite.addTest(ModelComparison('test_transformations'))
        return suite
        
    def test_models_list(self):
        ''' Test add_clf and rm_clf
        '''
        # Test models
        from sklearn import svm
        from sklearn import tree
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import GradientBoostingClassifier
        svm_clf = svm.SVC(probability=True)
        tree_clf = tree.DecisionTreeClassifier(max_depth=6)
        gnb_clf = GaussianNB()
        gb_clf = GradientBoostingClassifier()

        # Add all models
        mc = copper.ModelComparison()
        mc.add_clf(svm_clf, 'SVM')
        mc.add_clf(tree_clf, 'DT')
        mc.add_clf(gnb_clf, 'GNB')
        mc.add_clf(gb_clf, 'GB')

        df = mc.clfs
        self.assertEqual(df['SVM'], svm_clf)
        self.assertEqual(df['DT'], tree_clf)
        self.assertEqual(df['GNB'], gnb_clf)
        self.assertEqual(df['GB'], gb_clf)

        # Remove one model
        mc.rm_clf('DT')
        df = mc.clfs
        try:
            self.assertEqual(df['DT'], dt_clf)
            self.fail("should generate error")
        except:
            pass
        self.assertEqual(df['SVM'], svm_clf)
        self.assertEqual(df['GNB'], gnb_clf)
        self.assertEqual(df['GB'], gb_clf)

        # Remove all models
        mc.clear_clfs()
        df = mc.clfs
        try:
            self.assertEqual(df['SVM'], svm_clf)
            self.fail("should generate error")
        except:
            pass
        try:
            self.assertEqual(df['DT'], tree_clf)
            self.fail("should generate error")
        except:
            pass
        try:
            self.assertEqual(df['GNB'], gnb_clf)
            self.fail("should generate error")
        except:
            pass
        try:
            self.assertEqual(df['GB'], gb_clf)
            self.fail("should generate error")
        except:
            pass

    def test_transformations(self):
        ''' Tests that the values used to train and test are correct
        Note: tests for the transformation values on test_transforms
        '''
        dic = { 'Cat.1': ['A','B','A','A','B'],
                'Cat.2' :['f','g','h','g','f'],
                'Num.1': np.random.rand(5),
                'Num.2': np.random.rand(5),
                'Target': [1,0,0,1,1]}
        train = copper.Dataset(pd.DataFrame(dic))
        dic = { 'Cat.1': ['B','B','B','A','A'],
                'Cat.2' :['g','h','f','g','g'],
                'Num.1': np.random.rand(5),
                'Num.2': np.random.rand(5),
                'Target': [0,1,1,0,1]}
        test = copper.Dataset(pd.DataFrame(dic))

        train.role['Target'] = train.TARGET
        test.role['Target'] = train.REJECT

        mc = copper.ModelComparison()
        mc.train = train
        mc.test = test
        self.assertEqual(mc.X_train, copper.transform.inputs2ml(train).values)
        self.assertEqual(mc.y_train, copper.transform.target2ml(train).values)
        self.assertEqual(mc.y_train, train['Target'].values)
        self.assertEqual(mc.X_train.shape, (5,7))
        self.assertEqual(mc.X_test, copper.transform.inputs2ml(test).values)
        self.assertEqual(mc.y_test, None)
        self.assertEqual(mc.X_test.shape, (5,7))

        test.role['Target'] = test.TARGET
        mc.test = test
        self.assertEqual(mc.y_test, copper.transform.target2ml(test).values)
        self.assertEqual(mc.y_test, test['Target'].values)
        
        train.role['Num.1'] = train.REJECT
        mc.train = train
        self.assertEqual(mc.X_train.shape, (5,6))
        train.role['Cat.1'] = train.REJECT
        mc.train = train
        self.assertEqual(mc.X_train.shape, (5,4))

if __name__ == '__main__':
    suite = ModelComparison().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
