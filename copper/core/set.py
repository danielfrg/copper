import os
import io
import json
import copper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

class Dataset(dict):
    '''
    Wrapper around pandas to define metadata of a DataFrame.
    Also introduces a some utils for filling missing data, ploting.
    '''

    # Constants

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
        self.frame = pd.read_csv(os.path.join(copper.project.data, file_path))
        if autoMetadata:
            self.create_metadata()

    def create_metadata(self):
        '''
        Creates metadata for the data

        Parameters
        ----------
            df: pandas.DataFrame
        '''
        # TODO: rewrite this method?
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
        Generates and returns a DataFrame with the targets ready for doing
        Machine Learning

        Returns
        -------
            df: pandas.Series
        '''
        col = self.role[self.role == self.TARGET].index[0]
        ans = self._category2number(self.frame[col])
        ans.name = 'Target'
        return ans

    target = property(get_target)

    def get_numbers(self):
        cols = self.type[self.type == self.NUMBER].index
        return self.frame[cols]

    number = property(get_numbers)

    def get_money(self):
        cols = self.type[self.type == self.MONEY].index
        return self.frame[cols]

    money = property(get_money)

    def get_categories(self):
        cols = self.type[self.type == self.CATEGORY].index
        return self.frame[cols]

    category = property(get_categories)

    # --------------------------------------------------------------------------
    #                                TRANSFORMS
    # --------------------------------------------------------------------------

    def _money2number(self, series):
        '''
        Converts a Series with money format to a numbers

        Parameters
        ----------
            series: pandas.Series, target to convert

        Returns
        -------
            pandas.Series with the converted data
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
        Converts a Series with category format to a format for machine learning
        Represents the same information on different columns of ones and zeros

        Parameters
        ----------
            series: pandas.Series, target to convert

        Returns
        -------
            pandas.DataFrame with the converted data
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
        Convert a Series with categorical information to a Series of numbers
        using the scikit-learn LabelEncoder

        Parameters
        ----------
            series: pandas.Series, target to convert

        Returns
        -------
            pandas.Series with the converted data
        '''
        le = preprocessing.LabelEncoder()
        le.fit(series.values)
        vals = le.transform(series.values)
        return pd.Series(vals, index=series.index, name=series.name, dtype=float)

    def _category_labels(self, series):
        '''
        Return the labels for a Series with categorical values

        Parameters
        ----------
            series: pandas.Series, target to convert

        Returns
        -------
            list, labels of the series
        '''
        le = preprocessing.LabelEncoder()
        le.fit(series.values)
        return le.classes_

    # --------------------------------------------------------------------------
    #                              METADATA
    # --------------------------------------------------------------------------

    def get_metadata(self):
        '''
        Generates and return a DataFrame with a summary of the metadata:
            * Role
            * Type

        Returns
        -------
            pandas.DataFrame with the role and type of each column
        '''
        metadata = pd.DataFrame(index=self.columns, columns=['Role', 'Type'])
        metadata['Role'] = self.role
        metadata['Type'] = self.type
        return metadata

    metadata = property(get_metadata)

    # --------------------------------------------------------------------------
    #                           EXPLORE / PLOTS
    # --------------------------------------------------------------------------

    def histogram(self, col, bins=20, legend=True, retList=False):
        '''
        Draws a histogram for the selected column on matplotlib

        Parameters
        ----------
            bins: int, number of bins of the histogram, default 20
            legend: boolean, True if want to display the legend of the ploting
            ret_list: boolean, True if want the method to return a list with the
                                distribution(information) of each bin

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

        if retList:
            return pd.Series(labels)

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

    def percent_nas(self, ascending=False):
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
        '''
        Fill missing values using a method

        Parameters
        ----------
            cols: list, of columns to fill missing values
            method: str, method to use to fill missing values
                * mean(numerical,money)/mode(categorical): use the mean or most
                  repeted value of the column
        '''
        if cols is None:
            cols = self.columns
        if cols is str:
            cols = [cols]

        for col in self.columns:
            if self.type[col] == self.NUMBER or self.type[col] == self.MONEY:
                if method == 'mean' or method == 'mode':
                    value = self[col].mean()
            if self.type[col] == self.CATEGORY:
                if method == 'mode' or method == 'mode':
                    pass # TODO
            self[col].fillna(value=value, inplace=True)

if __name__ == "__main__":
    copper.project.path = '../../examples/expedia'
    train = copper.read_csv('train.csv')
    # copper.export(train, name='train', format='json')
    # print(train.frame)
    # train.histogram('x2')
    # plt.show()

    # print(train.cov().to_csv('cov.csv'))
    # print(train.corr().to_csv('corr.csv'))

    print(train)


    ''' Donors
    copper.project.path = '../tests/'
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

