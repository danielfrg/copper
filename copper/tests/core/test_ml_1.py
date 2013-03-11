import os
import copper
import numpy as np
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class ML_1(CopperTest):

    ml = None

    def suite(self):
        suite = unittest.TestSuite()
        # suite.addTest(ML_1('test_models_list'))
        # suite.addTest(ML_1('test_metrics'))
        suite.addTest(ML_1('test_cm'))
        return suite

    def setup(self):
        if self.ml is None:
            self.setUpData()

            self.train = copper.Dataset('ml/1/train.csv')
            self.train.role['CustomerID'] = self.train.ID
            self.train.role['Order'] = self.train.TARGET
            fnc = lambda x: 12*(2007 - int(str(x)[0:4])) - int(str(x)[4:6]) + 2
            self.train['LASD'] = self.train['LASD'].apply(fnc)

            self.test = copper.Dataset('ml/1/test.csv')
            self.test.role['CustomerID'] = self.test.ID
            self.test.role['Order'] = self.test.TARGET
            self.test['LASD'] = self.test['LASD'].apply(fnc)

            self.ml = copper.MachineLearning()
            self.ml.train = self.train
            self.ml.test = self.test

            from sklearn import svm
            svm_clf = svm.SVC(probability=True)
            from sklearn import tree
            tree_clf = tree.DecisionTreeClassifier(max_depth=6)
            from sklearn.naive_bayes import GaussianNB
            gnb_clf = GaussianNB()
            from sklearn.ensemble import GradientBoostingClassifier
            gr_bst_clf = GradientBoostingClassifier()

            self.ml.add_clf(svm_clf, 'SVM')
            self.ml.add_clf(tree_clf, 'DT')
            self.ml.add_clf(gnb_clf, 'GNB')
            self.ml.add_clf(gr_bst_clf, 'GB')

            self.ml.fit()

    def test_metrics(self):
        self.setup()

        acc = self.ml.accuracy()
        self.assertEqual(acc['GB'], 0.72)
        self.assertEqual(acc['DT'], 0.7135)
        self.assertEqual(acc['SVM'], 0.6995)
        self.assertEqual(acc['GNB'], 0.6790)
        self.assertEqual(acc.name, 'Accuracy')

        auc = self.ml.auc()
        self.assertEqual(auc['GNB'], 0.589258, digits=3)
        self.assertEqual(auc['GB'], 0.577164, digits=3)
        self.assertEqual(auc['DT'], 0.547310, digits=3)
        self.assertEqual(auc['SVM'], 0.544189, digits=3)
        self.assertEqual(auc.name, 'Area Under the Curve')

        mse = self.ml.mse()
        self.assertEqual(mse['GNB'], 0.3210)
        self.assertEqual(mse['SVM'], 0.3005)
        self.assertEqual(mse['DT'], 0.2865)
        self.assertEqual(mse['GB'], 0.2800)
        self.assertEqual(mse.name, 'Mean Squared Error')

        # ROC
        self.assertEqual(self.ml.roc(), None)
        self.assertEqual(self.ml.roc(ret_list=True), auc)

    def test_cm(self):
        self.setup()

        sol = copper.read_csv('ml/1/cm.csv').set_index('Model')
        sol.index.name = None

        cms = self.ml._cm()
        self.assertEqual(cms['GNB'], np.array([[1196,  236], [ 406,  162]]))
        self.assertEqual(cms['DT'], np.array([[1365,   67], [ 506,   62]]))
        self.assertEqual(cms['SVM'], np.array([[1362,   70], [ 531,   37]]))
        self.assertEqual(cms['GB'], np.array([[1387,   45], [ 515,   53]]))

        self.assertEqual(self.ml.cm('GNB').values, np.array([[1196,  236], [ 406,  162]]))
        self.assertEqual(self.ml.cm('DT').values, np.array([[1365,   67], [ 506,   62]]))
        self.assertEqual(self.ml.cm('SVM').values, np.array([[1362,   70], [ 531,   37]]))
        self.assertEqual(self.ml.cm('GB').values, np.array([[1387,   45], [ 515,   53]]))

        self.assertEqual(self.ml.cm_table(), sol)
        cm_0 = sol.ix[:, sol.columns[0:3]].sort(['Rate 0\'s'], ascending=False)
        cm_1 = sol.ix[:, sol.columns[3:]].sort(['Rate 1\'s'], ascending=False)
        self.assertEqual(self.ml.cm_table(0), cm_0)
        self.assertEqual(self.ml.cm_table(1), cm_1)


if __name__ == '__main__':
    suite = ML_1().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
