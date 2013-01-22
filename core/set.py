import os
import copper
import numpy as np
import pandas as pd

from sklearn import preprocessing

class DataSet(object):
    '''
    Wrapper around a few pandas.DataFrames to include metadata.

    Usage
    -----
        data_set = copper.load(<csv_file>)
    '''

    def __init__(self):
        self._name = ''
        self._oframe = None
        self.money_symbol = '$'

        self._money_cols = None
        self._binary_cols = None
        self._number_cols = None

    def set_data(self, df):
        '''
        Uses a pandas.DataFrame to generate the metadata [Role, Type]

        Parameters
        ----------
            df: pandas.DataFrame
        '''
        self._oframe = df
        self._columns = self._oframe.columns.values
        self._name = ''.join(self._columns)

        # -- Roles
        self.role = pd.Series(index=self._oframe.columns, name='Role', dtype=str)
        id_cols = [c for c in self._columns if c.lower() in ['id', 'index']]
        target_cols = [c for c in self._columns if c.lower().startswith('target')]
        self.role[id_cols[0]] = 'ID'
        self.role[target_cols[0]] = 'Target'
        self.role[target_cols[1:]] = 'Rejected'
        self.role = self.role.fillna(value='Input') # Missing cols are Input

        # -- Types
        self.type = pd.Series(index=self.role.index, name='Type', dtype=str)

        number_cols = [c for c in self._columns
                            if self._oframe.dtypes[c] in [np.int64, np.float64]]
        self.type[number_cols] = 'Number'

        money_cols = []
        obj_cols = self._oframe.dtypes[self._oframe.dtypes == object].index
        for col in obj_cols:
            x = [x[:1] for x in self._oframe[col].dropna().values]
            y = [self.money_symbol for y in x]
            eq = np.array(x) == np.array(y)
            if len(eq[eq==True]) >= 0.1 * len(x):
                money_cols.append(col)
        self.type[money_cols] = 'Money'

        # -- Types: Finally
        self.type = self.type.fillna(value='Category')

        # Encode categories:
        self._categories_encoders = {}
        for col in self.type.index[self.type == 'Category']:
            le = preprocessing.LabelEncoder()
            le.fit(self._oframe[col].values)
            self._categories_encoders[col] = le

    def gen_frame(self, cols=None,
                  encodeCategory=False, mlCategory=False):
        '''
        Generates and returns a new pandas.DataFrame given the conditions

        Parameters
        ----------
            cols=None: list - filter for the columns of the DataFrame,
                              by default uses all the columns available
            encodeCategory=False: boolean - True if want to encode the categorical
                                     columns into number, useful for exploration
            mlCategory=False: boolean - True if want to convert the categorial
                  columns for machine learning, useful for use with scikit-learn

        Returns
        -------
            df: pandas.DataFrame
        '''
        if cols is None:
            cols = self._columns

        ans = pd.DataFrame(columns=cols, index=self._oframe.index)

        for col in cols:
            if col in self.type[self.type == 'Number']:
                ans[col] = self._oframe[col]
            elif col in self.type[self.type == 'Money']:
                # Removes the '$' and ','' if necessary
                ser = pd.Series(index=ans.index, dtype=float)
                for index, value in zip(self._oframe.index, self._oframe[col]):
                    if type(value) == str:
                        rm_sign = ''.join(value.split(self.money_symbol))
                        rm_coma = ''.join(rm_sign.split(','))
                        ser[index] = float(rm_coma)
                ans[col] = ser
            else: # Category column
                if encodeCategory:
                    le = self._categories_encoders[col] # LabelEncoder
                    ans[col] = le.transform(self._oframe[col].values)
                elif mlCategory:
                    # Creates and appends a few pd.Series for each category
                    cat_col = self._oframe[col]
                    categories = list(set(cat_col))
                    categories.sort()
                    for category in categories:
                        n_data = pd.Series(index=self._oframe.index)
                        n_data[cat_col == category] = 1
                        n_data.name = '%s [%s]' % (cat_col.name, category)
                        n_data = n_data.fillna(value=0)
                        ans = ans.join(n_data)
                    del ans[col] # Deletes the original column
                else:
                    ans[col] = self._oframe[col]

        return ans

    # --------------------------------------------------------------------------
    #                                METHODS
    # --------------------------------------------------------------------------

    def load(self, file_path):
        ''' Loads data and tries to figure out the best metadata '''
        self.set_data(pd.read_csv(os.path.join(copper.config.data, file_path)))

    def restore(self):
        ''' Restores the original version of the DataFrame '''
        self.set_data(self._oframe)

    # --------------------------------------------------------------------------
    #                               PROPERTIES
    # --------------------------------------------------------------------------

    def get_metadata(self):
        '''
        Return a pandas.DataFrame with a summary of the metadata [Role, Type]
        '''
        metadata = pd.DataFrame(index=self._columns, columns=['Role', 'Type'])
        metadata['Role'] = self.role
        metadata['Type'] = self.type
        return metadata

    def get_frame(self):
        '''
        Return a pandas.DataFrame with this catacteristics:
            1. Money columns are transformed into float
            2. Categorical columns are ignored (default is used)
        '''
        return self.gen_frame()

    def get_inputs(self):
        '''
        Return a pandas.DataFrame of the columns marked with Role='Input'
        with this catacteristics:
            1. Money columns transformed into float
            2. Categorical columns are transformed for Machine Learning
        '''
        return self.gen_frame(cols=self.role[self.role == 'Input'].index,
                                                                mlCategory=True)

    def get_target(self):
        '''
        Return a pandas.DataFrame of the column marked with Role='Target'
        with this catacteristics:
            1. Money columns transformed into float
        '''
        return self.gen_frame(cols=self.role[self.role == 'Target'].index)

    def get_cat_coder(self):
        return self._categories_encoders

    def histogram(self, bins=20):
        '''
        Return a pandas.DataFrame with the information necessary to build a
        histogram for each column, that is divisions and count of each bin

        Parameters
        ----------
            bins=20: int, number of bins of the histogram

        Return
        ------
            df: pandas.DataFrame
        '''
        data = self.gen_frame(cols=self.role[self.role != 'ID'].index,
                                                encodeCategory=True)
        index = range(-1, bins+1)
        ans = pd.DataFrame(index=index)
        for col in data.columns:
            count, division = np.histogram(data[col].dropna().values, bins=bins)
            dummy = np.zeros(bins+2)
            n_df = pd.DataFrame({'a': dummy, 'b': dummy}, index=index)
            n_df.columns = [[col, col],['division', 'count']]
            n_df[col]['division'].ix[-1] = np.nan
            n_df[col]['division'].ix[0:bins+1] = division
            n_df[col]['count'].ix[-1] = len(data[col]) - data[col].count()
            n_df[col]['count'].ix[0:bins-1] = count
            n_df[col]['count'].ix[bins] = np.nan
            ans = ans.join(n_df)
        return ans

    def __unicode__(self):
        return self.get_metadata()

    def __str__(self):
        return str(self.__unicode__())


    metadata = property(get_metadata)
    frame = property(get_frame)
    inputs = property(get_inputs)
    target = property(get_target)

    cat_coder = property(get_cat_coder)

if __name__ == "__main__":
    # copper.config.data_dir = '../tests/data'
    # ds = copper.load('dataset/test1/data.csv')
    copper.config.path = '../examples/donors'
    ds = copper.load('donors.csv')
    # print(ds)
    # print(ds.frame['DemGender'].value_counts())

    # print(ds.gen_frame(encodeCategory=True)['DemHomeOwner'].tail(10))
    # print(ds.frame['DemHomeOwner'].tail(10))

    print(ds.histogram().to_string())
    # copper.export(ds.histogram(), name='hist')
