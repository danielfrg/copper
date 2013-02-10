import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'
train = copper.Dataset()
train.load('raw/train.csv')
test = copper.Dataset()
test.load('raw/test.csv')

train.role['depend'] = train.TARGET
copper.save(train, 'train')

test.role['depend'] = test.TARGET
copper.save(test, 'test')

# Fill missing values using mean
train.fillna(method='mean')
copper.save(train, 'train_mean')

test.fillna(method='mean')
copper.save(test, 'test_mean')

# print train.corr()

# Histograms - Log transforms
# ans = train.histogram('x18', legend=False)
# print train['x26']
train['x18'] = train['x18'].map(np.log)
train['x19'] = train['x19'].map(np.log)
train['x16'] = train['x16'].map(np.log)
train['x14'] = train['x14'].map(np.log)
# train['x7'] = train['x7'].map(np.log)
train['x13'] = train['x13'].map(np.log)
train['x35'] = train['x35'].map(np.log)
train['x22'] = train['x22'].map(np.log)
train['x36'] = train['x36'].map(np.log)
train['x9'] = train['x9'].map(np.log)
train['x11'] = train['x11'].map(np.log)
# train['x12'] = train['x12'].map(np.log)
# ans = train.histogram('x11', legend=False)
# plt.show()

copper.save(train, 'train_mean_log')
