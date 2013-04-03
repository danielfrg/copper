# coding=utf-8
from __future__ import division
import os
import io
import json
import pickle

import copper
import pandas as pd


def load(filepath):
    ''' Loads a pickled dataset

    Returns
    -------
        copper.Dataset
    '''
    if len(filepath.split('.')) == 1:
        filepath = filepath + '.ds'

    if filepath.endswith('.ds'):
        f = os.path.join(copper.project.data, filepath)
        pkl_file = open(f, 'rb')
        return pickle.load(pkl_file)

def save(data, filename, to='', **args):
    ''' Saves a picke Dataset or a csv file

    Parameters
    ----------
        to: str, folder to save the file
    '''
    fp = os.path.join(copper.project.data, to)
    if not (os.access(fp, os.F_OK)):
            os.makedirs(fp)

    format = filename.split('.')[-1]

    if format == 'csv':
        if type(data) is copper.Dataset:
            df = data.frame
        else:
            df = data
        fpath = os.path.join(fp, filename)
        df.to_csv(fpath, encoding='utf-8', **args)

    elif format == 'json':
        fpath = os.path.join(fp, filename + '.json')
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            json.dumps(df_to_json(df), outfile)

    elif format == 'ds':
        f = os.path.join(fp, filename)
        output = open(f, 'wb')
        pickle.dump(data, output)
        output.close()

def read_csv(file_path, **args):
    ''' Reads a csv file into a pandas DataFrame

    Parameters
    ----------
        same as pandas.read_csv

    Returns
    -------
        pandas.DataFrame
    '''
    file_path = os.path.join(copper.project.data, file_path)
    return pd.read_csv(file_path, **args)


