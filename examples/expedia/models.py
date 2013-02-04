import copper
import pandas as pd
import matplotlib.pyplot as plt

train = copper.read_csv('train.csv')
train.role['depend'] = train.TARGET
train.fillna(method='mean')

test = copper.read_csv('test.csv')
test.role['depend'] = test.TARGET
test.fillna(method='mean')

ml = copper.MachineLearning()
ml.train = train
ml.test = test
