from __future__ import division
import copper
import pandas as pd


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

    >>> df = pd.DataFrame(np.random.rand(3, 3))
    >>> ds = copper.Dataset(df)
    >>> ds.frame
             0    1    2
        0  0.9  0.1  0.6
        1  0.7  0.6  0.8
        2  0.4  0.4  0.6
    >>> ds.metadata
                 Role      Type    dtype
        Columns
        0        Input    Number  float64
        1        Input    Number  float64
        2        Input    Number  float64
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
        """ Return the pandas.DataFrame

        Examples
        --------
        >>> ds.frame
                 0    1    2    3    4    5
            0  0.9  0.1  0.6  0.9  0.4  0.8
            1  0.7  0.6  0.8  0.1  0.1  0.0
            2  0.4  0.4  0.6  0.8  0.2  0.7
            3  0.7  0.2  0.9  0.9  0.8  0.6
            4  0.5  0.7  0.6  0.0  0.2  0.0
        """
        return self._frame

    def set_frame(self, frame):
        """ Set the data of the dataset.

        When used recreates the metadata.

        Examples
        --------
        >>> ds.frame = pd.DataFrame(...)
        """
        assert type(frame) is pd.DataFrame, 'should be a pandas.DataFrame'
        self._frame = frame
        columns = self._frame.columns
        self.role = pd.Series(name='Role', index=columns, dtype=object)
        self.type = pd.Series(name='Type', index=columns, dtype=object)
        if not frame.empty:
            self.role[:] = self.INPUT
            self.type[:] = self.NUMBER
            self.type[self._frame.dtypes == object] = self.CATEGORY

    def get_metadata(self):
        """ Return the pandas.DataFrame

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> ds.metadata
                      Role      Type    dtype
            Columns
            0        Input  Category   object
            1        Input    Number  float64
            2        Input    Number  float64
            3        Input  Category   object
            4        Input    Number  float64
            5        Input    Number  float64
        """
        metadata = pd.DataFrame(index=self._frame.columns)
        metadata.index.name = 'Columns'
        metadata['Role'] = self.role
        metadata['Type'] = self.type
        metadata['dtype'] = [] if len(metadata) == 0 else self._frame.dtypes.values
        return metadata

    def set_metadata(self, metadata):
        """ Sets metadata

        Notes
        -----
        The new metadata index needs to match previous metadata index
        (columns of the DataFrame) in order to work

        See Also
        --------
        copper.Dataset.match
        """
        assert type(metadata) is pd.DataFrame, 'should be a pandas.DataFrame'
        assert len(self.metadata) == len(metadata), \
            'Length is not consistent, try Dataset.match_metadata instead'
        assert (self.metadata.index.values == metadata.index.values).all(), \
            'Index is not consistent, try Dataset.match_metadata instead'
        self.role = metadata['Role']
        self.type = metadata['Type']

    def get_columns(self):
        """ Returns the columns of the frame

        Examples
        --------
        >>> ds.columns == df.frame.columns
            True
        """
        return self._frame.columns

    def get_index(self):
        """ Returns the index of the frame

        Examples
        --------
        >>> ds.index == df.frame.index
            True
        """
        return self._frame.index

    frame = property(get_frame, set_frame, None, 'pandas.DataFrame')
    metadata = property(get_metadata, set_metadata, None, 'pandas.DataFrame')
    columns = property(get_columns, None, None)
    index = property(get_index, None, None)

    def update(self):
        """ Updates the DataFrame based on the metadata.
        Transforms strings to numbers using regular expression.
        """
        for col in self._frame.columns:
            if self.type[col] == self.NUMBER and self._frame[col].dtype == object:
                self._frame[col] = self._frame[col].apply(copper.t.to_float)

    def filter_cols(self, role=None, type=None):
        """ Returns a list of the columns that matches the criterias.

        Parameters
        ----------
        role : list or string
        type : list or string

        Returns
        -------
        list with the columns names

        Examples
        --------
        >>> ds.filter_cols(role=ds.INPUT)
            ... list ...
        >>> ds.filter_cols(role=ds.INPUT, type=ds.CATEGORY)
            ... list ...
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

        Examples
        --------
        >>> ds.filter() == ds.frame
            True
        >>> ds.filter(role=ds.INPUT)
            ... pd.DataFrame ...
        >>> ds.filter(role=ds.INPUT, type=ds.CATEGORY)
            ... pd.DataFrame ...
        """
        return self._frame[self.filter_cols(role, type)]

    def __getitem__(self, name):
        return self._frame[name]

    def __setitem__(self, name, value):
        self._frame[name] = value

    def __len__(self):
        return len(self._frame)

    def __str__(self):
        return str(self.metadata)

#           ---------  TESTS
import numpy as np
import math
import random
from copper.tests.utils import eq_
from nose.tools import raises





if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
