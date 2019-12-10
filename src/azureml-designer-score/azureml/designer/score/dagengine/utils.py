import os
import zipfile
from time import perf_counter

def set_global_setting(key, value):
    os.environ[key] = value

def get_global_setting(key):
    return os.environ.get(key, '')

def set_root_path(path):
    global root_path
    root_path = path

def get_root_path():
    return root_path

class ModelZip(object):
    def __init__(self, path, target_dir='studiomodelpackage', graph_file='modelpackage.json'):
        self.path = path
        self.target_dir = os.path.join(os.environ.get('DSPATH',''), target_dir)
        self.graph_file = graph_file
        self.extractall()

    def get_graph_json(self):
        return self.get_resource_data(self.graph_file)

    def get_resource_data(self, name):
        fn = self.get_fullname(name)
        with open(fn) as fp:
            data = fp.read()
        return data

    def get_fullname(self, name):
        return os.path.join(self.target_dir, name)

    def get_dir(self):
        return self.target_dir

    def extractall(self):
        with zipfile.ZipFile(self.path) as zf:
            zf.extractall(self.target_dir)

class PerformanceCounter(object):
    def __init__(self, logger, name):
        self.name = name
        self.logger = logger
        self.start = 0

    def __enter__(self):
        self.logger.info(f'Start {self.name}.')
        self.start = perf_counter()

    def __exit__(self, *args):
        duration = (perf_counter() - self.start) * 1000
        self.logger.info(f'End {self.name}. Duration: {duration} milliseconds.')