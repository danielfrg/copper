from __future__ import division
import pandas as pd
from copper.utils import transforms


class Dataset(dict):

    IGNORE = 'Ignore'
    INPUT = 'Input'
    TARGET = 'Target'
    NUMBER = 'Number'
    CATEGORY = 'Category'

    def __init__(self, frame=None):
        self.role = pd.Series()
        self.type = pd.Series()
        self.frame = pd.DataFrame() if frame is None else frame

    def get_frame(self):
        return self._frame

    def set_frame(self, frame):
        assert type(frame) is pd.DataFrame, "frame should be a pandas.DataFrame"
        self._frame = frame
        self.role = pd.Series(name='Role', index=self._frame.columns, dtype=object)
        self.type = pd.Series(name='Type', index=self._frame.columns, dtype=object)
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
        assert type(metadata) is pd.DataFrame, "metadata should be a pandas.DataFrame"
        assert len(self.metadata) == len(metadata), 'Should have the same length (rows)'
        self.role = metadata['Role']
        self.type = metadata['Type']

    frame = property(get_frame, set_frame, None, 'pandas.DataFrame')
    metadata = property(get_metadata, set_metadata, None, 'pandas.DataFrame')

    def update(self):
        ''' Updates the frame based on the metadata
        '''
        for col in self._frame.columns:
            if self.type[col] == self.NUMBER and self._frame[col].dtype == object:
                self._frame[col] = self._frame[col].apply(transforms.to_float)

    def filter_cols(self, role=None, type=None):
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
