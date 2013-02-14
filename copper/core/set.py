# coding=utf-8
from __future__ import division
import os
import io
import json
import copper
import numpy as np
import pandas as pd

class Dataset(dict):
    '''
    Wrapper around pandas to define metadata to a pandas DataFrame.
    Also introduces a some utils for filling missing data, ploting.
    '''

    # Constants
    NUMBER = 'Number'
    CATEGORY = 'Category'

    ID = 'ID'
    INPUT = 'Input'
    TARGET = 'Target'
    REJECTED = 'Rejected'

    def __init__(self, data=None):
        self.frame = None
        self.role = None
        self.type = None

        self.unique_values_limit = 20
        self.percent_filter = 0.9
        self.money_percent_filter = 0.9
        self.money_symbols = []

        if data is not None:
            if type(data) is pd.DataFrame:
                self.set_frame(data)
            elif type(data) is str:
                self.load(data)

    # --------------------------------------------------------------------------
    #                                 LOAD
    # --------------------------------------------------------------------------

    def _id_identifier(self, col_name):
        '''
        Indentifier for Role=ID based on the name of the column

        Returns
        -------
            boolean
        '''
        return col_name.lower() in ['id']

    def _target_identifier(self, col_name):
        '''
        Indentifier for Role=Target based on the name of the column

        Returns
        -------
            boolean
        '''
        return col_name.lower() in ['target']

    def load(self, file_path, autoMetadata=True):
        '''
        Loads a csv file from the project/data directory.
        Then calls self.set_data to create the metadata

        Parameters
        ----------
            file_path: str
        '''
        filepath = os.path.join(copper.project.data, file_path)
        self.set_frame(pd.read_csv(filepath), autoMetadata=autoMetadata)

    def set_frame(self, dataframe, autoMetadata=True):
        self.frame = dataframe
        self.columns = self.frame.columns.values
        self.role = pd.Series(index=self.columns, name='Role', dtype=str)
        self.type = pd.Series(index=self.columns, name='Type', dtype=str)
        if autoMetadata:
            self.generate_metada()

    def generate_metada(self):
        '''
        Creates metadata for the data

        Parameters
        ----------
            df: pandas.DataFrame
        '''
        # Roles
        id_cols = [c for c in self.columns if self._id_identifier(c)]
        if len(id_cols) > 0:
            self.role[id_cols] = 'ID'

        target_cols = [c for c in self.columns if self._target_identifier(c)]
        if len(target_cols) > 0:
            # Set only one target by default
            self.role[target_cols[0]] = self.TARGET
            self.role[target_cols[1:]] = self.REJECTED

        rejected = self.percent_missing()[self.percent_missing() > 0.5].index
        self.role[rejected] = self.REJECTED

        self.role = self.role.fillna(value=self.INPUT) # Missing cols are Input

        # Types
        number_cols = [c for c in self.columns
                              if self.frame.dtypes[c] in (np.int64, np.float64)]
        self.type[number_cols] = self.NUMBER

        self.type = self.type.fillna(value=self.CATEGORY)

    # --------------------------------------------------------------------------
    #                                PROPERTIES
    # --------------------------------------------------------------------------

    def get_inputs(self):
        '''
        Generates and returns a DataFrame with the inputs ready for doing
        Machine Learning

        Returns
        -------
            df: pandas.DataFrame
        '''
        ans = pd.DataFrame(index=self.frame.index)
        for col in self.role[self.role == self.INPUT].index:
            if self.type[col] == self.NUMBER and \
                              self.frame[col].dtype in (np.int64, np.float64):
                ans = ans.join(self.frame[col])
            elif self.type[col] == self.NUMBER and \
                                        self.frame[col].dtype == object:
                ans = ans.join(copper.transform.to_number(self.frame[col]))
            elif col in self.type[self.type == self.CATEGORY] and \
                            self.frame[col].dtype in (np.int64, np.float64):
                new_cols = copper.transform.category2ml(self.frame[col])
                # new_cols = copper.transform.category2number(self.frame[col])
                ans = ans.join(new_cols)
            elif col in self.type[self.type == self.CATEGORY]:
                # new_cols = copper.transform.category2ml(self.frame[col])
                new_cols = copper.transform.category2number(self.frame[col])
                ans = ans.join(new_cols)
            else:
                # Crazy stuff TODO: generate error
                pass
        return ans

    inputs = property(get_inputs)

    def get_target(self):
        '''
        Generates and returns a DataFrame with the targets ready for doing
        Machine Learning

        Returns
        -------
            df: pandas.Series
        '''
        col = self.role[self.role == self.TARGET].index[0]
        ans = copper.transform.category2number(self.frame[col])
        ans.name = 'Target'
        return ans

    target = property(get_target)

    def get_metadata(self):
        '''
        Generates and return a DataFrame with a summary of the data:
            * Role
            * Missing values

        Returns
        -------
            pandas.DataFrame with the role and type of each column
        '''
        metadata = pd.DataFrame(index=self.columns)
        metadata['Role'] = self.role
        metadata['Type'] = self.type
        metadata['dtype'] = self.frame.dtypes
        # metadata['nas'] = len(self.frame) - self.frame.count()
        return metadata

    metadata = property(get_metadata)

    def update(self):
        for col in self.frame.columns:
            if self.type[col] == self.NUMBER and \
                                        self.frame[col].dtype == object:
                self.frame[col] = copper.transform.to_number(self.frame[col])
            elif col in self.type[self.type == self.CATEGORY] and \
                            self.frame[col].dtype in (np.int64, np.float64):
                self.frame[col] = copper.transform.category2number(self.frame[col])

    # --------------------------------------------------------------------------
    #                                    STATS
    # --------------------------------------------------------------------------

    def unique_values(self, ascending=False):
        '''
        Generetas a Series with the number of unique values of each column

        Parameters
        ----------
            ascending: boolean, sort the returned Series on this direction

        Returns
        -------
            pandas.Series
        '''
        ans = pd.Series(index=self.frame.columns)
        for col in self.frame.columns:
            ans[col] = len(self.frame[col].value_counts())
        return ans.order(ascending=ascending)

    def percent_missing(self, ascending=False):
        '''
        Generetas a Series with the percent of missing values of each column

        Parameters
        ----------
            ascending: boolean, sort the returned Series on this direction

        Returns
        -------
            pandas.Series
        '''
        return (1 - (self.frame.count() / len(self.frame))).order(ascending=ascending)

    def variance_explained(self, cols=None):
        # TODO: is this working?
        if cols is None:
            cols = self.inputs.columns
        U, s, V = np.linalg.svd(self.inputs[cols].values)
        var = np.square(s) / sum(np.square(s))
        xlocations = np.array(range(len(var)))+0.5
        width = 0.99
        plt.bar(xlocations, var, width=width)
        return var

    def corr(self, col=None, ascending=False):
        if col is None:
            col = self.role[self.role == self.TARGET].index[0]
        corrs = self.frame.corr()
        corrs = corrs[col]
        corrs = corrs[corrs.index != col]
        return corrs.order(ascending=ascending)

    def fillna(self, cols=None, method='mean'):
        '''
        Fill missing values using a method

        Parameters
        ----------
            cols: list, of columns to fill missing values
            method: str, method to use to fill missing values
                * mean(numerical,money)/mode(categorical): use the mean or most
                  repeted value of the column
                * knn
        '''
        if cols is None:
            cols = self.columns
        if cols is str:
            cols = [cols]

        if method == 'mean' or method == 'mode':
            for col in cols:
                if self.type[col] == self.NUMBER:
                    if method == 'mean' or method == 'mode':
                        value = self[col].mean()
                if self.type[col] == self.CATEGORY:
                    if method == 'mode' or method == 'mode':
                        pass # TODO
                self[col] = self[col].fillna(value=value)
        elif method == 'knn':
            for col in cols:
                imputed = copper.r.imputeKNN(self.frame)
                self.frame[col] = imputed[col]
    # --------------------------------------------------------------------------
    #                                    CHARTS
    # --------------------------------------------------------------------------

    def histogram(self, col, **args):
        '''
        Draws a histogram for the selected column on matplotlib

        Parameters
        ----------
            col:str, column name
            bins: int, number of bins of the histogram, default 20
            legend: boolean, True if want to display the legend of the ploting
            ret_list: boolean, True if want the method to return a list with the
                                distribution(information) of each bin

        Return
        ------
            nothing, figure is ready to be shown
        '''
        copper.plot.histogram(self.frame[col], **args)

    # --------------------------------------------------------------------------
    #                              SPECIAL METHODS
    # --------------------------------------------------------------------------

    def __unicode__(self):
        return self.metadata

    def __str__(self):
        return str(self.__unicode__())

    def __getitem__(self, name):
        return self.frame[name]

    def __setitem__(self, name, value):
        self.frame[name] = value

    def __len__(self):
        return len(self.frame)

    def head(self, n=5):
        return self.frame.head(n)

    def tail(self, n=5):
        return self.frame.tail(n)

if __name__ == "__main__":
    copper.project.path = '../../examples/coursera_data_analysis/assignment1'
    dataset = copper.Dataset()
    dataset.load('loansData.csv')

    dataset.type['Interest.Rate'] = dataset.NUMBER
    dataset.type['Loan.Length'] = dataset.NUMBER
    dataset.type['Debt.To.Income.Ratio'] = dataset.NUMBER
    dataset.type['Employment.Length'] = dataset.NUMBER
    # print dataset.metadata
    dataset.fillna(method='knn')
    print dataset.frame
    # print dataset.inputs
    # print dataset.inputs.head()

    # import matplotlib.pyplot as plt
    # dataset.histogram('FICO.Range', legend=False)
    # plt.show()
