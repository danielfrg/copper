from __future__ import division
import pandas as pd
from copper.utils import transforms


class Dataset(dict):
    """ Wrapper around pandas.DataFrame introducing metadata to the different
    variables/columns making easier to interate in manual machine learning
    feature learning. Also provides access to basic data transformation such as
    string to numbers.
    Finally provides a convinient interface to the copper.ModelComparison
    utilities.

    Parameters
    ----------
    data : pandas.DataFrame

    Examples
    --------
    >>> df = pandas.read_cvs('a_csv_file.csv')
    >>> ds = copper.Dataset(df)
    """

    IGNORE = 'Ignore'
    INPUT = 'Input'
    TARGET = 'Target'
    NUMBER = 'Number'
    CATEGORY = 'Category'

    def __init__(self, data=None):
        self.role = pd.Series()
        self.type = pd.Series()
        self.frame = pd.DataFrame() if data is None else data

    def get_frame(self):
        return self._frame

    def set_frame(self, frame):
        assert type(frame) is pd.DataFrame, 'should be a pandas.DataFrame'
        self._frame = frame
        columns = self._frame.columns
        self.role = pd.Series(name='Role', index=columns, dtype=object)
        self.type = pd.Series(name='Type', index=columns, dtype=object)
        self.role[:] = self.INPUT
        self.type[:] = self.NUMBER

    def get_metadata(self):
        metadata = pd.DataFrame(index=self._frame.columns)
        metadata.index.name = 'Columns'
        metadata['Role'] = self.role
        metadata['Type'] = self.type
        metadata['dtype'] = [] if len(metadata) == 0 else self._frame.dtypes.values
        return metadata

    def set_metadata(self, metadata):
        assert type(metadata) is pd.DataFrame, 'should be a pandas.DataFrame'
        assert len(self.metadata) == len(metadata), 'length is not consistent'
        assert self.metadata.columns == metadata.columns, \
            'Columns do not match, try Dataset.match_metadata instead'
        self.role = metadata['Role']
        self.type = metadata['Type']

    frame = property(get_frame, set_frame, None, 'pandas.DataFrame')
    metadata = property(get_metadata, set_metadata, None, 'pandas.DataFrame')

    def update(self):
        """ Updates the DataFrame based on the metadata.
        Transforms strings to numbers using regular expression.
        """
        for col in self._frame.columns:
            if self.type[col] == self.NUMBER and self._frame[col].dtype == object:
                self._frame[col] = self._frame[col].apply(transforms.to_float)

    def filter_cols(self, role=None, type=None):
        """ Returns a list of the columns that matches the criterias.

        Parameters
        ----------
        role : list or string
        type : list or string

        Returns
        -------
        list with the columns names
        """
        def _type(obj):
            return obj.__class__

        if role is None:
            role = [self.INPUT, self.TARGET, self.IGNORE]
        elif _type(role) == str:
            role = [role]
        if type is None:
            type = [self.NUMBER, self.CATEGORY]
        elif _type(type) == str:
            type = [type]

        return [col for col in self._frame.columns.tolist()
                if self.role[col] in role and self.type[col] in type]

    def filter(self, role=None, type=None):
        """ Returns a pandas.DataFrame with the variables that match the
        criterias.

        Parameters
        ----------
        role : list or string
        type : list or string

        Returns
        -------
        pandas.DataFrame
        """
        return self._frame[self.filter_cols(role, type)]

    def __getitem__(self, name):
        return self._frame[name]

    def __setitem__(self, name, value):
        self._frame[name] = value

    def __len__(self):
        return len(self._frame)

#           ---------  TESTS

from copper.tests.utils import eq_

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
