import os

class Config(object):
    def __init__(self):
        self._data_dir = ''

    def get_data_dir(self):
        return self._data_dir

    def set_data_dir(self, value):
        self._data_dir = os.path.realpath(value)

    data_dir = property(get_data_dir, set_data_dir)
