import os
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

class DataSet(dict):
    '''
    Wrapper around a few pandas.DataFrames to include metadata
    Provides easy transformation and generatios of DataFrames of the columns
    by defining roles and types of each column

    Usage
    -----
        data_set = copper.DataSet()
        data_set.load(<csv_file>)
    '''

    MONEY = 'Money'
    NUMBER = 'Number'
    CATEGORY = 'Category'

    ID = 'ID'
    INPUT = 'Input'
    TARGET = 'Target'
    REJECTED = 'Rejected'

    MONEY_SYMBOLS = ['$','£','€']

    def __init__(self):
        self.frame = None
        self.role = None
        self.type = None

        self.categoriesLimitFilter = 20
        self.moneyPercentFilter = 0.1
        self.money_symbols = ['$']

    # --------------------------------------------------------------------------
    #                                 LOAD
    # --------------------------------------------------------------------------

    def _id_identifier(self, col_name):
        return col_name.lower() in ['id']

    def _target_identifier(self, col_name):
        return col_name.lower() in ['target']

    def load(self, file_path):
        '''
        Uses a pandas.DataFrame to generate the metadata [Role, Type]

        Parameters
        ----------
            df: pandas.DataFrame
        '''
        self.frame = pd.read_csv(os.path.join(copper.config.data, file_path))
        self.columns = self.frame.columns.values

        # Roles
        self.role = pd.Series(index=self.columns, name='Role', dtype=str)
        # Roles: ID
        id_cols = [c for c in self.columns if self._id_identifier(c)]
        if len(id_cols) > 0:
            self.role[id_cols] = 'ID'
        # Roles: Target
        target_cols = [c for c in self.columns if self._target_identifier(c)]
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
                            if self.frame.dtypes[c] in [np.int64, np.float64]
                                and (len(set(self.frame[c].values)) > self.categoriesLimitFilter
                                    or self.role[c] == self.TARGET)]
        self.type[number_cols] = self.NUMBER

        # Types: Money
        money_cols = []
        obj_cols = self.frame.dtypes[self.frame.dtypes == object].index
        for col in obj_cols:
            x = [x[:1] for x in self.frame[col].dropna().values]
            for money_symbol in self.money_symbols:
                y = [money_symbol for y in x]
                eq = np.array(x) == np.array(y)
                if len(eq[eq==True]) >= self.moneyPercentFilter * len(x):
                    money_cols.append(col)
        self.type[money_cols] = self.MONEY

        # Types: Category
        self.type = self.type.fillna(value=self.CATEGORY)

        # Transform money columns to numbers
        for col in self.type[self.type == self.MONEY].index:
            self.frame[col] = self._money2number(self.frame[col])

    # --------------------------------------------------------------------------
    #                              INPUTS / TARGET
    # --------------------------------------------------------------------------

    def get_inputs(self):
        '''
        Generates and returns a new pandas.DataFrame ready for machine learning
        Returns
        -------
            df: pandas.DataFrame
        '''
        cols = self.columns
        ans = pd.DataFrame(index=self.frame.index)
        for col in self.role[self.role == self.INPUT].index:
            if col in self.type[self.type == self.NUMBER]:
                ans = ans.join(self.frame[col])
            elif col in self.type[self.type == self.MONEY]:
                ans = ans.join(self.frame[col])
                # ans = ans.join(self._money2number(self.frame[col]))
            elif col in self.type[self.type == self.CATEGORY]:
                new_cols = self._category2ml(self.frame[col])
                ans = ans.join(new_cols)
            else:
                # Crazy stuff TODO: generate error
                pass
        return ans

    inputs = property(get_inputs)

    def get_target(self):
        col = self.role[self.role == self.TARGET].index[0]
        ans = self._category2number(self.frame[col])
        ans.name = 'Target'
        return ans

    target = property(get_target)

    # --------------------------------------------------------------------------
    #                                 TRANSORMS
    # --------------------------------------------------------------------------

    def _money2number(self, series):
        ans = pd.Series(index=series.index, name=series.name, dtype=float)
        splits = ''.join(self.money_symbols) + ','

        for index, value in zip(self.frame.index, series):
            if type(value) == str:
                # number = re.match(r"[0-9]{1,3}(?:\,[0-9]{3})+(?:\.[0-9]{1,10})", value)
                for split in splits:
                    value = ''.join(value.split(split))
                ans[index] = float(value)
        return ans

    def _category2ml(self, series):
        ans = pd.DataFrame(index=series.index)
        categories = list(set(series))
        categories.sort()
        for category in categories:
            n_col = pd.Series(np.zeros(len(self.frame.index)),
                                index=self.frame.index, dtype=int)
            n_col.name = '%s [%s]' % (series.name, category)
            n_col[series == category] = 1
            ans = ans.join(n_col)
        return ans

    def _category2number(self, series):
        le = preprocessing.LabelEncoder()
        le.fit(series.values)
        vals = le.transform(series.values)
        return pd.Series(vals, index=series.index, dtype=float)

    def _category_labels(self, series):
        le = preprocessing.LabelEncoder()
        le.fit(series.values)
        return le.classes_

    # --------------------------------------------------------------------------
    #                              METADATA
    # --------------------------------------------------------------------------

    def get_metadata(self):
        '''
        Return a pandas.DataFrame with a summary of the metadata [Role, Type]
        '''
        metadata = pd.DataFrame(index=self.columns, columns=['Role', 'Type'])
        metadata['Role'] = self.role
        metadata['Type'] = self.type
        return metadata

    metadata = property(get_metadata)

    # --------------------------------------------------------------------------
    #                                  PLOTS
    # --------------------------------------------------------------------------

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
        data = self.frame[col]
        if self.type[col] == self.CATEGORY:
            types = self._category_labels(data)
            data = self._category2number(data)
        values = data.dropna().values

        if self.type[col] == self.CATEGORY:
            bins = len(set(values))
        elif self.type[col] == self.NUMBER or self.type[col] == self.MONEY:
            if bins is None:
                bins=20
        count, divis = np.histogram(values, bins=bins)

        if self.type[col] == self.CATEGORY:
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

    # --------------------------------------------------------------------------
    #                              SPECIAL METHODS
    # --------------------------------------------------------------------------

    def __unicode__(self):
        return self.frame

    def __str__(self):
        return str(self.__unicode__())

    def __getitem__(self, name):
        return self.frame[name]

    def __setitem__(self, name, value):
        self.frame[name] = value

if __name__ == "__main__":
    copper.config.path = '../tests/'
    ds = copper.DataSet()
    ds.load('donors/data.csv')

    import matplotlib.pyplot as plt
    ds.histogram('DemGender')
    plt.show()

