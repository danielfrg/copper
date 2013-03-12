import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class ML_basic(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(ML_basic('test_models_list'))
        suite.addTest(ML_basic('test_transformations'))
        return suite
        
    def test_models_list(self):
        '''
        Test add_clf and rm_clf
        '''

        # Test models
        from sklearn import svm
        svm_clf = svm.SVC(probability=True)
        from sklearn import tree
        tree_clf = tree.DecisionTreeClassifier(max_depth=6)
        from sklearn.naive_bayes import GaussianNB
        gnb_clf = GaussianNB()
        from sklearn.ensemble import GradientBoostingClassifier
        gr_bst_clf = GradientBoostingClassifier()

        # Add all models
        ml = copper.MachineLearning()
        ml.add_clf(svm_clf, 'SVM')
        ml.add_clf(tree_clf, 'Decision Tree')
        ml.add_clf(gnb_clf, 'GaussianNB')
        ml.add_clf(gr_bst_clf, 'Grad Boosting')

        df = ml.clfs
        self.assertEqual(df['SVM'], svm_clf)
        self.assertEqual(df['Decision Tree'], tree_clf)
        self.assertEqual(df['GaussianNB'], gnb_clf)
        self.assertEqual(df['Grad Boosting'], gr_bst_clf)

        # Remove 1 model
        ml.rm_clf('SVM')
        df = ml.clfs
        try:
            self.assertEqual(df['SVM'], svm_clf)
            self.fail("should generate error")
        except:
            pass
        self.assertEqual(df['Decision Tree'], tree_clf)
        self.assertEqual(df['GaussianNB'], gnb_clf)
        self.assertEqual(df['Grad Boosting'], gr_bst_clf)

        # Remove all models
        ml.clear_clfs()
        df = ml.clfs
        try:
            self.assertEqual(df['SVM'], svm_clf)
            self.fail("should generate error")
        except:
            pass
        try:
            self.assertEqual(df['Decision Tree'], tree_clf)
            self.fail("should generate error")
        except:
            pass
        try:
            self.assertEqual(df['GaussianNB'], gnb_clf)
            self.fail("should generate error")
        except:
            pass
        try:
            self.assertEqual(df['Grad Boosting'], gr_bst_clf)
            self.fail("should generate error")
        except:
            pass

    def test_transformations(self):
        '''
        Tests that the values used to train and test are correct
        Note: tests for the transformation values on transforms
        '''
        self.setUpData()
        ds = copper.Dataset('transforms/ml/data.csv')
        ds.type['Num.as.Cat'] = ds.CATEGORY
        ds.role['Target.Num'] = ds.TARGET
        ds.role['Target.Cat'] = ds.REJECTED

        ml = copper.MachineLearning()
        ml.train = ds
        ml.test = ds
        self.assertEqual(ml.X_train, copper.transform.inputs2ml(ds).values)
        self.assertEqual(ml.y_train, copper.transform.target2ml(ds).values)
        self.assertEqual(ml.X_test, copper.transform.inputs2ml(ds).values)
        self.assertEqual(ml.y_test, copper.transform.target2ml(ds).values)
        self.assertEqual(ml.X_train.shape, (25,10))

        # Reject a few variables and test again
        ds.role['Num.1'] = ds.REJECT
        ds.role['Cat.1'] = ds.REJECT
        ml.train = ds
        ml.test = ds
        self.assertEqual(ml.X_train.shape, (25,4))
        self.assertEqual(ml.X_train, copper.transform.inputs2ml(ds).values)
        self.assertEqual(ml.y_train, copper.transform.target2ml(ds).values)
        self.assertEqual(ml.X_test, copper.transform.inputs2ml(ds).values)
        self.assertEqual(ml.y_test, copper.transform.target2ml(ds).values)

if __name__ == '__main__':
    suite = ML_basic().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
