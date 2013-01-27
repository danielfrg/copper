import os
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

class DataSet(object):
    '''
    Wrapper around a few pandas.DataFrames to include metadata
    Provides easy transformation and generatios of DataFrames of the columns
    by defining roles and types of each column

    Usage
    -----
        data_set = copper.DataSet()
        data_set.load(<csv_file>)
    '''

    def __init__(self):
        self._name = ''
        self._oframe = None
        self.money_symbol = '$'
        self._transformations = []

        self._money_cols = None
        self._binary_cols = None
        self._number_cols = None

        # Configurations for import
        self.categoriesLimitFilter = 20
        self.idFilterCols = ['id', 'index']
        self.targetFilterCols = ['target']

        self.moneyPercentFilter = 0.1

    def _idFilter(self, col_name):
            return col_name.lower() in self.idFilterCols

    def _targetFilter(self, col_name):
        for _filter in self.targetFilterCols:
            if col_name.lower().startswith(_filter):
                return True
        return False

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

        # Roles
        self.role = pd.Series(index=self.columns, name='Role', dtype=str)
        # Roles: ID
        id_cols = [c for c in self.columns if self._idFilter(c)]
        if len(id_cols) > 0:
            self.role[id_cols] = 'ID'
        # Roles: Target
        target_cols = [c for c in self.columns if self._targetFilter(c)]
        if len(target_cols) > 0:
            self.role[target_cols[0]] = self.TARGET
            self.role[target_cols[1:]] = self.REJECTED
        # Roles: Input
        self.role = self.role.fillna(value=self.INPUT) # Missing cols are Input

        # Types
        self.type = pd.Series(index=self.role.index, name='Type', dtype=str)

        # Types: Number
        # -- dtype of the column is np.int or np.float AND (
        #            num of different values is greater than filter (default=20)
        #            OR role of the column is target, target are Number
        number_cols = [c for c in self.columns
                            if self._oframe.dtypes[c] in [np.int64, np.float64]
                                and (len(set(self._oframe[c].values)) > self.categoriesLimitFilter
                                    or self.role[c] == self.TARGET)]
        self.type[number_cols] = self.NUMBER

        # Types: Money
        money_cols = []
        obj_cols = self._oframe.dtypes[self._oframe.dtypes == object].index
        for col in obj_cols:
            x = [x[:1] for x in self._oframe[col].dropna().values]
            y = [self.money_symbol for y in x]
            eq = np.array(x) == np.array(y)
            if len(eq[eq==True]) >= self.moneyPercentFilter * len(x):
                money_cols.append(col)
        self.type[money_cols] = self.MONEY

        # Types: Category
        self.type = self.type.fillna(value=self.CATEGORY)

        # Create the categories encoders
        self._categories_encoders = {}
        for col in self.type.index[self.type == self.CATEGORY]:
            le = preprocessing.LabelEncoder()
            le.fit(self._oframe[col].values)
            self._categories_encoders[col] = le


        self._index_init = 0
        self._index_final = len(self._oframe)

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
                        n_data = pd.Series(np.zeros(len(self._oframe.index)),
                                            index=self._oframe.index, dtype=int)
                        n_data.name = '%s [%s]' % (cat_col.name, category)
                        n_data[cat_col == category] = 1
                        ans = ans.join(n_data)
                    del ans[col] # Deletes the original column
                else:
                    ans[col] = self._oframe[col]

        for trans in self._transformations:
            if trans['col'] in ans.columns:
                ans[trans['col']] = ans[trans['col']].apply(trans['fnc'])

        return ans[self._index_init:self._index_final]

    # --------------------------------------------------------------------------
    #                          PROPERTIES & CONSTANTS
    # --------------------------------------------------------------------------

    MONEY = 'Money'
    NUMBER = 'Number'
    CATEGORY = 'Category'

    ID = 'ID'
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
        df = self.gen_frame(encodeCategory=True,
                                cols=self.role[self.role == self.TARGET].index)
        if len(df.columns) == 1:
            return df[df.columns[0]]
        return df

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
        ''' Saves a pickle version of the DataSet '''
        copper.save(self, name=name, format=format)

    def load(self, file_path):
        ''' Loads data and tries to figure out the best metadata '''
        self.set_data(pd.read_csv(os.path.join(copper.config.data, file_path)))

    def restore(self):
        ''' Restores the original version of the DataFrame '''
        self.set_data(self._oframe)

    def index(self, initial, final):
        self._index_init = initial
        self._index_final = final

    def transform(self, col, fnc):
        self._transformations.append({'col': col, 'fnc':fnc})

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
        return ax

    def __unicode__(self):
        return self.get_metadata()

    def __str__(self):
        return str(self.__unicode__())

# if __name__ == "__main__":
    # copper.config.path = '../tests/'
    # ds = copper.DataSet()
    # ds.load('dataset/test1/data.csv')
    # ds.type['Number as Category'] = ds.NUMBER
    # print(ds)
    # print(ds.gen_frame(encodeCategory=True))


    # copper.config.path = '../examples/donors'
    # ds = copper.DataSet()
    # ds.load('donors.csv')
    # print(ds)
    # print(ds.inputs['DemGender']) # TODO: make it possible

    # print(ds.gen_frame(encodeCategory=True)['DemHomeOwner'].tail(10))
    # print(ds.frame['DemHomeOwner'].tail(10))

    # ds.save(name='donors')
    # ds = copper.load('donors.dataset')

    # ds.histogram('DemMedIncome')
    # ds.histogram('DemGender')
    # plt.show()

    # copper.config.path = '../examples/iris'
    # ds = copper.read_csv('iris.csv')
    # print(ds)
    # ds.role['class'] = 'as'


