import os
import pickle
import hashlib

import copper
import pandas as pd


def load(file_path):
    if file_path.endswith('dataset'):
        f = os.path.join(copper.config.data, file_path)
        pkl_file = open(f, 'rb')
        return pickle.load(pkl_file)
    elif file_path.endswith('csv'):
        f = os.path.join(copper.config.data, file_path)
        ds = copper.DataSet()
        ds.load(f)
        return ds

def export(data, name, format='csv'):
    if type(data) is pd.DataFrame and format == 'csv':
        if not (os.access(copper.config.export, os.F_OK)):
            os.makedirs(copper.config.export)

        fpath = os.path.join(copper.config.export, name + '.csv')
        data.to_csv(fpath, encoding='utf-8')

'''
def save_cache(data, name):
    cache_dir = os.path.join(copper.config.data_dir, 'cache')

    if not (os.access(cache_dir, os.F_OK)):
        os.makedirs(cache_dir)

    h = hashlib.md5()
    h.update(name.encode('utf8'))
    filename = h.hexdigest() + '.data'

    file_name = os.path.join(cache_dir, filename)
    data.save(file_name)


def load_cache(name):
    cache_dir = os.path.join(copper.data_path, 'cache')

    if not (os.access(cache_dir, os.F_OK)):
        os.makedirs(cache_dir)

    h = hashlib.md5()
    h.update(name.encode('utf8'))
    filename = h.hexdigest() + '.data'

    file_name = os.path.join(cache_dir, name)
    f = os.path.join(cache_dir, filename)
    if os.access(f, os.F_OK):
        return pd.load(f)
'''
