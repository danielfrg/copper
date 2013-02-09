import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'
test = copper.Dataset()
test.load('test.csv')
test.role['depend'] = test.TARGET
# print test.percent_missing()
test.fillna(method='mean')

# Histograms
# ans = test.histogram('x18', legend=False)
# print test['x26']
test['x18'] = test['x18'].map(np.log)
test['x19'] = test['x19'].map(np.log)
test['x16'] = test['x16'].map(np.log)
test['x14'] = test['x14'].map(np.log)
# test['x7'] = test['x7'].map(np.log)
test['x13'] = test['x13'].map(np.log)
test['x35'] = test['x35'].map(np.log)
test['x22'] = test['x22'].map(np.log)
test['x36'] = test['x36'].map(np.log)
test['x9'] = test['x9'].map(np.log)
test['x11'] = test['x11'].map(np.log)
# test['x12'] = test['x12'].map(np.log)
ans = test.histogram('x11', legend=False)
plt.show()

# copper.export(test, 'test_clean')
copper.save(test, 'test_clean')
