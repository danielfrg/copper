import copper
import pandas as pd
import matplotlib.pyplot as plt

# train = pd.read_csv('train.csv')
train = copper.read_csv('train.csv')
train.role['depend'] = train.TARGET

# print(train.percent_nas())
train.fillna(method='mean')
print(train.metadata)
# print(train.unique_values())

# Histograms
# ans = train.histogram('x2', legend=False)
# plt.draw()
# plt.figure()

# ans = train.histogram('x2')
# plt.show()
