import os
import copper
import numpy as np
import pandas as pd

class DataSet(object):

    def __init__(self):
        self._name = ''
        self._oframe = None
        self.money_symbol = '$'

        self._money_cols = None
        self._binary_cols = None
        self._number_cols = None

    def set_data(self, data_frame):
        self._oframe = data_frame
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

    def generate_frame(self, cols=None, ignoreCategory=False):
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
            else: # Category/Category column
                if ignoreCategory:
                    ans[col] = self._oframe[col]
                else:
                    # Creates and appends a new pd.Series for each category
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

        return ans

    def get_metadata(self):
        metadata = pd.DataFrame(index=self._columns, columns=['Role', 'Type'])
        metadata['Role'] = self.role
        metadata['Type'] = self.type
        return metadata

    def get_frame(self):
        return self.generate_frame(ignoreCategory=True)

    def get_inputs(self):
        ans = self.generate_frame(cols=self.role[self.role == 'Input'].index)
        return ans

    def get_target(self):
        ans = self.generate_frame(cols=self.role[self.role == 'Target'].index)
        return ans

    def load(self, file_path):
        self.set_data(pd.read_csv(os.path.join(copper.config.data, file_path)))


    def restore(self):
        ''' Restores the original version of the DataFrame '''
        self.set_data(self._oframe)

    def __unicode__(self):
        return self.get_metadata()

    def __str__(self):
        return str(self.__unicode__())

    frame = property(get_frame)
    metadata = property(get_metadata)
    inputs = property(get_inputs)
    target = property(get_target)

if __name__ == "__main__":
    # copper.config.data_dir = '../tests/data'
    # ds = copper.load('dataset/test1/data.csv')
    copper.config.path = '../examples/donors'
    ds = copper.load('donors.csv')
    print(ds)
    # ds.export(name='frame', format='csv', w='frame')
    # ds.export(name='inputs', format='csv', w='inputs')
    # ds.export(name='target', format='csv', w='target')
