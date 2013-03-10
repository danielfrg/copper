import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'
train = copper.load('train.dataset')
# train = copper.load('train_mean.dataset')
# train = copper.load('train_mean_log.dataset')
# train = copper.load('train_imp.dataset')
test = copper.load('test.dataset')

# print len(test), len(train)
# print train.corr('depend')

# train.histogram('depend')
train.scatter('x39', 'x40', s=100, alpha=0.2)
# train.frame[train.frame.columns[36:40]].boxplot()
plt.show()

# from pandas.tools.plotting import scatter_matrix
# plot1 = scatter_matrix(train.frame[train.frame.columns[-5:]], alpha=0.2, figsize=(8, 8))
# plt.show()
# plt.savefig('fig6.pdf')

# from pandas.tools.plotting import radviz
# radviz(train.frame[['depend', 'x1', 'x2', 'x3', 'x4']], 'depend')
# plt.show()
