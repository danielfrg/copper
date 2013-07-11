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
        self._frame = pd.DataFrame()
        if data is not None:
            self.frame = data

# -----------------------------------------------------------------------------
#                               Properties

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

        recreate = True
        if len(self._frame.columns) > 0:
            if len(frame.columns) == len(self._frame.columns):
                if (frame.columns == self._frame.columns).all():
                    recreate = False
        if recreate:
            columns = frame.columns
            self.role = pd.Series(name='Role', index=columns, dtype=object)
            self.type = pd.Series(name='Type', index=columns, dtype=object)
            if not frame.empty:
                self.role[:] = self.INPUT
                self.type[:] = self.NUMBER
                self.type[frame.dtypes == object] = self.CATEGORY

        self._frame = frame

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
            'Length is not consistent, try Dataset.copy_metadata instead'
        assert (self.metadata.index.values == metadata.index.values).all(), \
            'Index is not consistent, try Dataset.copy_metadata instead'
        self.role = metadata['Role']
        self.type = metadata['Type']

    def copy_metadata(self, metadata, ignoreMissing=True):
        """ Copies the metadata from another dataset or dataframe

        Parameters
        ----------
        ignoreMissing: boolean
            If True (deafult) is going to ignore (do not modify)
            the variables that are not on the new metadata.
            if False is going to make role of variables not present on the
            new metadata "IGNORE"

        Returns
        -------

        """
        if isinstance(metadata, Dataset):
            metadata = metadata.metadata  # Brain damage

        if not ignoreMissing:
            self.role[:] = self.IGNORE
        for col in self.columns:
            if col in metadata.index:
                self.role[col] = metadata['Role'][col]
                self.type[col] = metadata['Type'][col]

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

    def __getitem__(self, name):
        return self._frame[name]

    def __setitem__(self, name, value):
        self._frame[name] = value

    def __len__(self):
        return len(self._frame)

    def __str__(self):
        return self.metadata.__str__()

    def __unicode__(self):
        return self.metadata.__unicode__()

# -----------------------------------------------------------------------------
#                            Functions

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

# -----------------------------------------------------------------------------
#                               Pandas API

    def head(self, *args, **kwargs):
        return self._frame.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self._frame.tail(*args, **kwargs)

'''
import math
import random
import copper
import numpy as np
import pandas as pd

from nose.tools import raises
from copper.tests.utils import eq_




if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
'''
