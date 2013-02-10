import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'
train = copper.Dataset()
train.load('exported/train_imp.csv')
test = copper.Dataset()
test.load('exported/test_imp.csv')

train.role['depend'] = train.TARGET
copper.save(train, 'train_imp')

test.role['depend'] = test.TARGET
copper.save(test, 'test_imp')

