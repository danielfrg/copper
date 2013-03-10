import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

copper.project.path = '../'

# Impute missing values via R
train_imp = copper.impute('raw/train.csv')
copper.export(train_imp, "train_imp")

test_imp = copper.impute('raw/test.csv')
copper.export(test_imp, "test_imp")

