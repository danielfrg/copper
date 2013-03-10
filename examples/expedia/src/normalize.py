import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'

train = copper.read_csv('raw/train.csv')

train_1 = train[train['depend'] == 1]
train_0 = train[train['depend'] == 0]

train_0_2 = train_0[0:len(train_1)]

train_normal = train_0_2.append(train_1)

copper.save(train_normal, 'train_normal')