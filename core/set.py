import os
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.columns = self._oframe.columns.values
        self._name = ''.join(self.columns)

        # -- Roles
        self.role = pd.Series(index=self._oframe.columns, name='Role', dtype=str)
        id_cols = [c for c in self.columns if c.lower() in ['id', 'index']]
        target_cols = [c for c in self.columns if c.lower().startswith('target')]
        self.role[id_cols[0]] = 'ID'
        self.role[target_cols[0]] = self.TARGET
        self.role[target_cols[1:]] = self.REJECTED
        self.role = self.role.fillna(value=self.INPUT) # Missing cols are Input

        # -- Types
        self.type = pd.Series(index=self.role.index, name='Type', dtype=str)

        number_cols = [c for c in self.columns
                            if self._oframe.dtypes[c] in [np.int64, np.float64]]
        self.type[number_cols] = self.NUMBER

        money_cols = []
        obj_cols = self._oframe.dtypes[self._oframe.dtypes == object].index
        for col in obj_cols:
            x = [x[:1] for x in self._oframe[col].dropna().values]
            y = [self.money_symbol for y in x]
            eq = np.array(x) == np.array(y)
            if len(eq[eq==True]) >= 0.1 * len(x):
                money_cols.append(col)
        self.type[money_cols] = self.MONEY

        # -- Types: Finally
        self.type = self.type.fillna(value=self.CATEGORY)

        # Encode categories:
        self._categories_encoders = {}
        for col in self.type.index[self.type == self.CATEGORY]:
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
            cols = self.columns

        ans = pd.DataFrame(columns=cols, index=self._oframe.index)

        for col in cols:
            if col in self.type[self.type == self.NUMBER]:
                ans[col] = self._oframe[col]
            elif col in self.type[self.type == self.MONEY]:
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
    #                          PROPERTIES & CONSTANTS
    # --------------------------------------------------------------------------

    MONEY = 'Money'
    NUMBER = 'Number'
    CATEGORY = 'Category'

    INPUT = 'Input'
    TARGET = 'Target'
    REJECTED = 'Rejected'

    def get_metadata(self):
        '''
        Return a pandas.DataFrame with a summary of the metadata [Role, Type]
        '''
        metadata = pd.DataFrame(index=self.columns, columns=['Role', 'Type'])
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
        Return a pandas.DataFrame of the columns marked with Role=self.INPUT
        with this catacteristics:
            1. Money columns transformed into float
            2. Categorical columns are transformed for Machine Learning
        '''
        return self.gen_frame(cols=self.role[self.role == self.INPUT].index,
                                                                mlCategory=True)

    def get_target(self):
        '''
        Return a pandas.DataFrame of the column marked with Role=self.TARGET
        with this catacteristics:
            1. Money columns transformed into float
        '''
        return self.gen_frame(cols=self.role[self.role == self.TARGET].index)

    def get_cat_coder(self):
        return self._categories_encoders

    metadata = property(get_metadata)
    frame = property(get_frame)
    inputs = property(get_inputs)
    target = property(get_target)

    cat_coder = property(get_cat_coder)

    # --------------------------------------------------------------------------
    #                               METHODS
    # --------------------------------------------------------------------------

    def save(self, name=None, format='.dataset'):
        import pickle
        f = os.path.join(copper.config.data, name + format)
        output = open(f, 'wb')
        pickle.dump(self, output)
        output.close()

    def load(self, file_path, *args):
        ''' Loads data and tries to figure out the best metadata '''
        self.set_data(pd.read_csv(os.path.join(copper.config.data, file_path)))

    def restore(self):
        ''' Restores the original version of the DataFrame '''
        self.set_data(self._oframe)

    def histogram(self, col, bins=None):
        '''
        Draws a histogram for the selected column on matplotlib

        Parameters
        ----------
            bins=20: int, number of bins of the histogram

        Return
        ------
            nothing, figure is ready to be shown
        '''
        data = self.gen_frame(cols=[col], encodeCategory=True)
        values = data.dropna().values[:,0]

        if self.type[col] == self.CATEGORY:
            bins = len(set(values))
        elif self.type[col] == self.NUMBER or self.type[col] == self.MONEY:
            if bins is None:
                bins=20
        count, divis = np.histogram(values, bins=bins)

        if self.type[col] == self.CATEGORY:
            types = self._categories_encoders[col].classes_
            tooltip = ['%d (%s)' % (cnt, typ) for cnt, typ in zip(count, types)]
        else:
            tooltip = ['%d (%d - %d)' % (c, i, f) for c, i, f in
                                    zip(count, divis[:-1], divis[1:])]

        width = 0.8 * (divis[1] - divis[0])
        center = (divis[:-1] + divis[1:]) / 2

        fig = plt.figure()
        # ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
        ax = fig.add_axes([0.075, 0.075, 0.7, 0.85])
        for c, h, t in zip(center, count, tooltip):
            ax.bar(c, h, align = 'center', width=width, label=t)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def __unicode__(self):
        return self.get_metadata()

    def __str__(self):
        return str(self.__unicode__())

if __name__ == "__main__":
    # copper.config.path = '../tests/'
    # ds = copper.load('dataset/test1/data.csv')
    copper.config.path = '../examples/donors'
    ds = copper.load('donors.csv')
    print(ds)
    # print(ds.frame['DemGender'].value_counts())

    # print(ds.gen_frame(encodeCategory=True)['DemHomeOwner'].tail(10))
    # print(ds.frame['DemHomeOwner'].tail(10))

    ds.save(name='donors')
    ds = copper.load('donors.dataset')

    ds.histogram('DemAge')
    # ds.histogram('DemGender', bins=20)
    plt.show()
