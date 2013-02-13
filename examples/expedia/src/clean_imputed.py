import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'
train = copper.Dataset()
train.load('exported/train_imp.csv')
test = copper.Dataset()
test.load('exported/test_imp.csv')

train.role['Unnamed: 0'] = train.REJECTED
train.role['Unnamed: 0.1'] = train.REJECTED
train.role['depend'] = train.TARGET
copper.save(train, 'train_imp')
print train

test.role['Unnamed: 0'] = test.REJECTED
test.role['Unnamed: 0.1'] = test.REJECTED
test.role['depend'] = test.TARGET
copper.save(test, 'test_imp')

