import os
import io
import json
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

class Dataset(dict):

    # ------ Constants

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

        self.unique_values_limit = 20
        self.money_percent_filter = 0.1
        self.money_symbols = []

    # --------------------------------------------------------------------------
    #                                 LOAD
    # --------------------------------------------------------------------------

    def _id_identifier(self, col_name):
        '''
        Indentifier for Role=ID based on the name of the column
        '''
        return col_name.lower() in ['id']

    def _target_identifier(self, col_name):
        '''
        Indentifier for Role=Target based on the name of the column
        '''
        return col_name.lower() in ['target']

    def load(self, file_path):
        '''
        Uses a pandas.DataFrame to generate the metadata [Role, Type]

        Parameters
        ----------
            df: pandas.DataFrame
        '''
        # TODO: rewrite this method?
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
        # Check for a lot of missing values
        rejected = self.percent_nas()[self.percent_nas() > 0.5].index
        self.role[rejected] = self.REJECTED
        # Roles: Input
        self.role = self.role.fillna(value=self.INPUT) # Missing cols are Input

        # Types
        self.type = pd.Series(index=self.columns, name='Type', dtype=str)

        # Types: Number
        unique_vals = self.unique_values()
        number_cols = [c for c in self.columns
                            if self.frame.dtypes[c] in [np.int64, np.float64]]
        self.type[number_cols] = self.NUMBER

        # Types: Money
        money_cols = []
        obj_cols = self.frame.dtypes[self.frame.dtypes == object].index
        for col in obj_cols:
            x = [x[:1] for x in self.frame[col].dropna().values]
            for money_symbol in self.MONEY_SYMBOLS:
                y = [money_symbol for y in x]
                eq = np.array(x) == np.array(y)
                if len(eq[eq==True]) >= self.money_percent_filter * len(x):
                    self.money_symbols.append(money_symbol)
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
            elif col in self.type[self.type == self.CATEGORY]:
                # ans = ans.join(self._category2number(self.frame[col]))
                new_cols = self._category2ml(self.frame[col])
                ans = ans.join(new_cols)
            else:
                # Crazy stuff TODO: generate error
                pass
        return ans

    inputs = property(get_inputs)

    def get_target(self):
        '''
        Generates and returns a new pandas.Series ready for machine learning

        Returns
        -------
            df: pandas.Series
        '''
        col = self.role[self.role == self.TARGET].index[0]
        ans = self._category2number(self.frame[col])
        ans.name = 'Target'
        return ans

    target = property(get_target)

    # --------------------------------------------------------------------------
    #                                TRANSFORMS
    # --------------------------------------------------------------------------

    def _money2number(self, series):
        '''
        Converts a pd.Series with a money format to a simple number
        '''
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
        '''
        Converts a pd.Series with category format to a pd.DataFrame representing
        the same information on different columns of ones and zeros
        '''
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
        '''
        Convert a pd.Series with categorical information to a pd.Series of numbers
        using the scikit-learn LabelEncoder
        '''
        le = preprocessing.LabelEncoder()
        le.fit(series.values)
        vals = le.transform(series.values)
        return pd.Series(vals, index=series.index, name=series.name, dtype=float)

    def _category_labels(self, series):
        '''
        Return the labels for a categorical pd.Series
        '''
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
    #                           EXPLORE / PLOTS
    # --------------------------------------------------------------------------

    def histogram(self, col, bins=20, legend=True):
        '''
        Draws a histogram for the selected column on matplotlib

        Parameters
        ----------
            bins=20: int, number of bins of the histogram

        Return
        ------
            nothing, figure is ready to be shown
        '''
        plt.hold(True)
        data = self.frame[col]
        nas = len(data) - len(data.dropna())
        data = data.dropna()

        if self.type[col] == self.CATEGORY:
            types = self._category_labels(data)
            data = self._category2number(data)
            bins = len(set(data))

            count, divis = np.histogram(data.values, bins=bins)
            width = 0.97 * (divis[1] - divis[0])

            types = types.tolist()
            types.insert(0, 'NA')
            count = count.tolist()
            count.insert(0, nas)

            labels = ['%s: %d' % (typ, cnt) for cnt, typ in zip(count, types)]
            centers = np.array(range(len(types))) - 0.5

            plt.bar(-width, nas, width=width, color='r', label=labels[0])
            for c, h, t in zip(centers[1:], count[1:], labels[1:]):
                plt.bar(c, h, align = 'center', width=width, label=t)

            plt.xticks(centers, types)
        else:
            count, divis = np.histogram(data.values, bins=bins)
            width = 0.97 * (divis[1] - divis[0])
            centers = (divis[:-1] + divis[1:]) / 2
            labels = ['%.1f - %.2f: %s' % (i, f, c) for c, i, f in
                                            zip(count, divis[:-1], divis[1:])]
            plt.bar(min(divis) - width, nas, width=width, color='r', label="NA: %d" % nas)
            for c, h, t in zip(centers, count, labels):
                plt.bar(c, h, align = 'center', width=width, label=t)

        if legend:
            plt.legend(loc='best')

        return pd.Series(labels)

    def stats(self):
        '''
        Generates a pd.DataFrame with a summary of important statistics
        '''
        pass # TODO

    def unique_values(self, ascending=False):
        ans = pd.Series(index=self.frame.columns)
        for col in self.frame.columns:
            ans[col] = len(self.frame[col].value_counts())
        return ans.order(ascending=ascending)

    def percent_nas(self, ascending=False):
        return (1 - (self.frame.count() / len(self.frame))).order(ascending=ascending)

    # --------------------------------------------------------------------------
    #                        SPECIAL METHODS / Pandas API
    # --------------------------------------------------------------------------

    def __unicode__(self):
        return self.metadata

    def __str__(self):
        return str(self.__unicode__())

    def __getitem__(self, name):
        return self.frame[name]

    def __setitem__(self, name, value):
        self.frame[name] = value

    def fillna(self, cols=None, method='mean'):
        if cols is None:
            cols = self.columns
        if cols is str:
            cols = [cols]

        for col in self.columns:
            if self.type[col] == self.NUMBER or self.type[col] == self.MONEY:
                if method == 'mean':
                    value = self[col].mean()
            if self.type[col] == self.CATEGORY:
                if method == 'mode' or method == 'mode':
                    pass # TODO
            self[col] = self[col].fillna(value=value)

    def cov(self):
        return self.frame.cov()

    def corr(self):
        return self.frame.corr()

if __name__ == "__main__":
    copper.config.path = '../project/'
    train = copper.read_csv('train.csv')
    # copper.export(train, name='train', format='json')
    # print(train.frame)
    # train.histogram('x2')
    # plt.show()

    # print(train.cov().to_csv('cov.csv'))
    print(train.corr().to_csv('corr.csv'))


    ''' Donors
    copper.config.path = '../tests/'
    ds = copper.DataSet()
    ds.load('donors/data.csv')
    ds.role['TARGET_D'] = ds.REJECTED
    ds.role['TARGET_B'] = ds.TARGET
    ds.type['ID'] = ds.CATEGORY

    # print(ds.inputs)
    # ds['GiftAvgCard36'].fillna(method)
    ds.fillna('DemAge', 'mean')
    ds.fillna('GiftAvgCard36', 'mean')

    import matplotlib.pyplot as plt
    ds.histogram('DemGender')
    plt.show()
    '''

