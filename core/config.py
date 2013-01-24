import os

class Config(object):
    def __init__(self):
        self._path = ''
        self._data = ''
        self._explore = ''
        self._export = ''

    def set_path(self, value):
        self._path = os.path.realpath(value)
        self._data = os.path.join(self._path, 'data')
        self._export = os.path.join(self._path, 'exported')

    def get_path(self):
        return self._path

    def get_data(self):
        return self._data

    def get_export(self):
        return self._export

    path = property(get_path, set_path)
    data = property(get_data)
    export = property(get_export)
