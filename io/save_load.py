import os
import hashlib

import copper
import pandas as pd


def load(file_path):
    ds = copper.DataSet()
    ds.load(file_path)
    return ds

def export_frame_csv(df, name):
    expo_dir = os.path.join(copper.config.data_dir, 'exported')
    if not (os.access(expo_dir, os.F_OK)):
        os.makedirs(expo_dir)

    fpath = os.path.join(expo_dir, name + '.csv')
    df.to_csv(fpath)


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

