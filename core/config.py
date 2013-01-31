import os

class Config(object):
    def __init__(self):
        self._path = ''
        self._data = ''
        self._cache = ''
        self._graphs = ''
        self._logs = ''

    def set_path(self, value):
        self._path = os.path.realpath(value)
        self._data = os.path.join(self._path, 'data')
        self._cache = os.path.join(self._path, 'cache')
        self._graphs = os.path.join(self._path, 'graphs')
        self._logs = os.path.join(self._path, 'logs')

    def get_path(self):
        return self._path

    def get_cache(self):
        return self._cache

    def get_data(self):
        return self._data

    def get_graphs(self):
        return self._export

    def get_logs(self):
        return self._logs

    path = property(get_path, set_path)
    cache = property(get_data)
    data = property(get_data)
    graphs = property(get_graphs)
    logs = property(get_logs)
