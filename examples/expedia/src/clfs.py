import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'
# train = copper.load('train.dataset')
# train = copper.load('train_mean.dataset')
# train = copper.load('train_mean_log.dataset')
train = copper.load('train_imp.dataset')
# test = copper.load('test.dataset')
# test = copper.load('test_mean.dataset')
test = copper.load('test_imp.dataset')

# print test.inputs

ml = copper.MachineLearning()
ml.set_train(train)
ml.set_test(test)
ml.costs = [[0,1],[5,0]]

from PyWiseRF import WiseRF
rf = WiseRF(n_estimators=50, n_jobs=2)
ml.add_clf(rf, 'RF')

from sklearn import tree
tree_clf = tree.DecisionTreeClassifier(max_depth=5)
ml.add_clf(tree_clf, 'Tree')

ml.fit()
print ml.accuracy()

ml.roc()
plt.show()

print ml.cm_table()
# print ml.cm_table(0)
# print ml.cm_table(1)
print ml.oportunity_cost()

