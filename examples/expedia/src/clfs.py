import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'
train = copper.load('train_clean.dataset')
test = copper.load('test_clean.dataset')

print train.corr()

# ml = copper.MachineLearning()
# ml.set_train(train)
# ml.set_test(test)

# from PyWiseRF import WiseRF
# rf = WiseRF(n_estimators=50, n_jobs=2)
# ml.add_clf(rf, 'RF')

# ml.fit()
# print ml.accuracy()

# ml.roc()
# plt.show()
