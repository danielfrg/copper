import os
import copper
import numpy as np
import pandas as pd

import unittest
from copper.tests.CopperTest import CopperTest

class ModelComparion_1(CopperTest):
    '''
    This tests covers an example using the catalog dataset
    '''

    ml = None

    def suite(self):
        suite = unittest.TestSuite()
        # suite.addTest(ModelComparion_1('test_metrics'))
        # suite.addTest(ModelComparion_1('test_cm'))
        # suite.addTest(ModelComparion_1('test_costs'))
        # suite.addTest(ModelComparion_1('test_predict'))
        suite.addTest(ModelComparion_1('test_average_bag'))
        return suite

    def setup(self):
        try :
            self.mc
        except:
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

            self.mc = copper.ModelComparison()
            self.mc.train = self.train
            self.mc.test = self.test

            from sklearn import svm
            svm_clf = svm.SVC(probability=True)
            from sklearn import tree
            tree_clf = tree.DecisionTreeClassifier(max_depth=6)
            from sklearn.naive_bayes import GaussianNB
            gnb_clf = GaussianNB()
            from sklearn.ensemble import GradientBoostingClassifier
            gr_bst_clf = GradientBoostingClassifier()

            self.mc.add_clf(svm_clf, 'SVM')
            self.mc.add_clf(tree_clf, 'DT')
            self.mc.add_clf(gnb_clf, 'GNB')
            self.mc.add_clf(gr_bst_clf, 'GB')

            self.mc.fit()

    def test_metrics(self):
        ''' Tests the different metrics: accuracy, auc, mse
        '''
        self.setup()

        acc = self.mc.accuracy()
        self.assertEqual(len(acc), 4)
        self.assertEqual(acc['GB'], 0.72)
        self.assertEqual(acc['DT'], 0.7135)
        self.assertEqual(acc['SVM'], 0.6995)
        self.assertEqual(acc['GNB'], 0.6790)
        self.assertEqual(acc.name, 'Accuracy')

        auc = self.mc.auc()
        self.assertEqual(len(auc), 4)
        self.assertEqual(auc['GNB'], 0.589258, digits=3)
        self.assertEqual(auc['GB'], 0.577164, digits=3)
        self.assertEqual(auc['DT'], 0.547310, digits=3)
        self.assertEqual(auc['SVM'], 0.544189, digits=3)
        self.assertEqual(auc.name, 'Area Under the Curve')

        mse = self.mc.mse()
        self.assertEqual(len(mse), 4)
        self.assertEqual(mse['GNB'], 0.3210)
        self.assertEqual(mse['SVM'], 0.3005)
        self.assertEqual(mse['DT'], 0.2865)
        self.assertEqual(mse['GB'], 0.2800)
        self.assertEqual(mse.name, 'Mean Squared Error')

        # ROC
        self.assertEqual(self.mc.roc(), None)
        self.assertEqual(self.mc.roc(ret_list=True), auc)

    def test_cm(self):
        ''' Tests the values of the confusion matrixes
        '''
        self.setup()

        cms = self.mc._cm()
        self.assertEqual(cms['GNB'], np.array([[1196,  236], [ 406,  162]]))
        self.assertEqual(cms['DT'], np.array([[1365,   67], [ 506,   62]]))
        self.assertEqual(cms['SVM'], np.array([[1362,   70], [ 531,   37]]))
        self.assertEqual(cms['GB'], np.array([[1387,   45], [ 515,   53]]))

        self.assertEqual(self.mc.cm('GNB').values, np.array([[1196,  236], [ 406,  162]]))
        self.assertEqual(self.mc.cm('DT').values, np.array([[1365,   67], [ 506,   62]]))
        self.assertEqual(self.mc.cm('SVM').values, np.array([[1362,   70], [ 531,   37]]))
        self.assertEqual(self.mc.cm('GB').values, np.array([[1387,   45], [ 515,   53]]))

        cm_table = self.mc.cm_table()
        rates_0 = np.array([0.729232, 0.729556, 0.746567, 0.719493])
        rates_1 = np.array([0.540816, 0.480620, 0.407035, 0.345794])
        self.assertEqual(cm_table['Rate 0\'s'].values, rates_0, 5)
        self.assertEqual(cm_table['Rate 1\'s'].values, rates_1, 5)

    def test_costs(self):
        '''
        Tests the values of the costs functions
        '''
        self.setup()

        self.mc.costs = np.array([[0, 4], [12, 16]])

        profits = np.array([ [944.0,2592.0,1648.0],[268,992,724],[180,848,668],
                            [280,592,312]])
        self.assertEqual(self.mc.profit().values, profits)
        
        op_costs = np.array([6652.0,6360.0,6340.0,5816.0])
        self.assertEqual(self.mc.oportunity_cost().values, op_costs)
       
        no_ml_cost = np.array([17184.0,9088.0,-8096.0])
        self.assertEqual(self.mc.cost_no_ml().values, no_ml_cost)

    def test_predict(self):
        ''' Tests the prediction and prediction probabilities
        Tests that using the defaul option is the same as using the given test
        Tests that is possible to predict other datasets
        '''
        self.setup()

        predict_train = copper.read_csv('ml/1/predict_train.csv').set_index('Model')
        predict_test = copper.read_csv('ml/1/predict_test.csv').set_index('Model')
        predict_proba_train = copper.read_csv('ml/1/predict_proba_train.csv').set_index('index')
        predict_proba_test = copper.read_csv('ml/1/predict_proba_test.csv').set_index('index')

        self.assertEqual(self.mc.predict(), predict_test)
        self.assertEqual(self.mc.predict(ds=self.test), predict_test)
        self.assertEqual(self.mc.predict(self.train), predict_train)

        self.assertEqual(self.mc.predict_proba(), predict_proba_test, 1)
        self.assertEqual(self.mc.predict_proba(self.test), predict_proba_test, 1)
        self.assertEqual(self.mc.predict_proba(ds=self.train), predict_proba_train, 1)

    def test_average_bag(self):
        ''' Tests the creation of an Average Bag
        '''
        self.setup()

        bag = copper.AverageBag(self.mc.clfs)
        self.mc.add_clf(bag, "bag")

        acc = self.mc.accuracy()
        self.assertEqual(len(acc), 5)
        self.assertEqual(acc['bag'], 0.719, 2)

        auc = self.mc.auc()
        self.assertEqual(len(auc), 5)
        self.assertEqual(auc['bag'], 0.58, 2)
            
        mse = self.mc.mse()
        self.assertEqual(len(mse), 5)
        self.assertEqual(mse['bag'], 0.281, 2)

if __name__ == '__main__':
    suite = ModelComparion_1().suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
