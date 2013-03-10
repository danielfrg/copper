import os
import copper
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class ML_basic(CopperTest):

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(ML_basic('test_models_list'))
        suite.addTest(ML_basic('test_models_list_2'))
        return suite

    def test_models_list_2(self):
        '''
        Test the creation of bagged model and remove that model
        '''
        # Test models
        from sklearn import svm
        svm_clf = svm.SVC(probability=True)
        from sklearn import tree
        tree_clf = tree.DecisionTreeClassifier(max_depth=6)

        # Add models
        ml = copper.MachineLearning()
        ml.add_clf(svm_clf, 'SVM')
        ml.add_clf(tree_clf, 'Decision Tree')
        ml.bagging("Bag 1")

        df = ml.clfs
        self.assertEqual(df['SVM'], svm_clf)
        self.assertEqual(df['Decision Tree'], tree_clf)
        self.assertNotEqual(df['Bag 1'], None)

        # Remove the essembled
        ml.rm_clf('Bag 1')
        self.assertEqual(df['SVM'], svm_clf)
        self.assertEqual(df['Decision Tree'], tree_clf)
        try:
            self.assertNotEqual(df['Bag 1'], None)
            self.fail("should generate error")
        except:
            pass

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


if __name__ == '__main__':
    suite = ML_basic().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
