import copper
import pandas as pd
import matplotlib.pyplot as plt


copper.project.path = '../'
train = copper.Dataset()
train.load('train.csv')
train.role['x32'] = train.INPUT
train.role['x38'] = train.INPUT
train.role['depend'] = train.TARGET
train.fillna(method='mean')

test = copper.Dataset()
test.load('test.csv')
test.role['x32'] = test.INPUT
test.role['x38'] = test.INPUT
test.role['depend'] = test.TARGET
test.fillna(method='mean')

print test

ml = copper.MachineLearning()
ml.set_train(train)
ml.set_test(test)

from PyWiseRF import WiseRF
rf = WiseRF(n_estimators=50, n_jobs=2)
ml.add_clf(rf, 'RF')

from sklearn import tree
tree_clf = tree.DecisionTreeClassifier(max_depth=5)
ml.add_clf(tree_clf, 'Tree')

from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()
ml.add_clf(gnb_clf, 'GNB')

# from sklearn import svm
# svm_clf = svm.SVC(kernel='linear', probability=True)
# ml.add_clf(svm_clf, 'SVM')

ml.fit()
# ml.bootstrap(tree.DecisionTreeClassifier, "DT", 7, max_depth=6)
# ml.bootstrap(GaussianNB, "GNB", 7)
ml.bagging("Bag 1")
# print ml.accuracy()

import matplotlib.pyplot as plt
ml.roc()
plt.show()




