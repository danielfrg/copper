import os
import pickle
import hashlib

import copper
import pandas as pd


def load(file_path):
    if len(file_path.split('.')) == 1:
        file_path = file_path + '.dataset'

    if file_path.endswith('dataset'):
        f = os.path.join(copper.config.data, file_path)
        pkl_file = open(f, 'rb')
        return pickle.load(pkl_file)

def save(data, name, format='csv'):
    if type(data) is pd.DataFrame and format == 'csv':
        if not (os.access(copper.config.export, os.F_OK)):
            os.makedirs(copper.config.export)

        fpath = os.path.join(copper.config.export, name + '.csv')
        data.to_csv(fpath, encoding='utf-8')
    elif type(data) is copper.DataSet:
        import pickle
        f = os.path.join(copper.config.data, name + format)
        output = open(f, 'wb')
        pickle.dump(data, output)
        output.close()

def read_csv(file_path):
    ds = copper.DataSet()
    ds.load(file_path)
    return ds
