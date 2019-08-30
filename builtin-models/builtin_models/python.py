from __future__ import absolute_import, division, print_function, unicode_literals

import cloudpickle
import inspect
import os
import re
import shutil
import sys
import zipfile
import importlib
from urllib.request import urlopen
import builtin_models.utils as utils

FLAVOR_NAME = "python"
MODEL_FILE_NAME = "model.pkl"  # we cloud pickle the model to load/save the model by default


class PythonModel(object):
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized models.
    """
    def __init__(self, model_path):
        pass

    def predict(self, input1, input2):
        """
        Evaluates several inputs and produces output.

        :param input1, input2: variable number of input arguments.

        """
        return input1, input2

    def save(self, model_path):
        """
        save model to model_path.
        """
        pass

class DummyPythonModel(PythonModel):
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized models.
    """
    def __init__(self, model_path = None):
        if(model_path is None):
            self.a = 1
            self.b = 2
        else:
            m = _load_model(model_path)
            self.a = m.a
            self.b = m.b

    def predict(self, x, y):
        print("input:")
        print(x)
        print(y)
        result = self.a *x + self.b * y 
        return result
    
    def save(self, model_path):
        _save_model(self, model_path)

def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_pip_deps=[
            "cloudpickle=={}".format(cloudpickle.__version__),
        ])

def _save_model(python_model, model_path):
    fn = os.path.join(model_path, MODEL_FILE_NAME)
    print(f'Save model to file: {fn}')
    with open(fn, 'wb') as fp:
        cloudpickle.dump(python_model, fp)

def _load_model(model_path):
    fn = os.path.join(model_path, MODEL_FILE_NAME)
    print(f'Load model from file: {fn}')
    with open(fn, 'rb') as fp:
        model = cloudpickle.load(fp)
    return model


# constants
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
MASTER_BRANCH = 'master'
DEFAULT_CACHE_DIR = '~/.cache'
VAR_DEPENDENCY = 'dependencies'
READ_DATA_CHUNK = 8192
cache_dir = None


def _get_cache_dir():
    cache_dir = os.path.expanduser(
        os.path.join(DEFAULT_CACHE_DIR, 'builtin_models'))
    return cache_dir


def _setup_cache_dir():
    global cache_dir
    if cache_dir is None:
        cache_dir = _get_cache_dir()

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


def _parse_repo_info(github):
    branch = MASTER_BRANCH
    if ':' in github:
        repo_info, branch = github.split(':')
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split('/')
    return repo_owner, repo_name, branch


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo_owner, repo_name, branch):
    return 'https://github.com/{}/{}/archive/{}.zip'.format(repo_owner, repo_name, branch)


def _download_archive_zip(url, filename):
    sys.stderr.write('Downloading: \"{}\" to {}\n'.format(url, filename))
    response = urlopen(url)
    with open(filename, 'wb') as f:
        while True:
            data = response.read(READ_DATA_CHUNK)
            if len(data) == 0:
                break
            f.write(data)


def _get_cache_or_reload(github, force_reload):
    # Parse github repo information
    repo_owner, repo_name, branch = _parse_repo_info(github)

    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    repo_dir = os.path.join(cache_dir, '_'.join([repo_owner, repo_name, branch]))

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        sys.stderr.write('Using cache found in {}\n'.format(repo_dir))
    else:
        branchfilename = branch.replace('/', '_').replace('\\', '_')
        cached_file = os.path.join(cache_dir, branchfilename + '.zip')
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch)
        _download_archive_zip(url, cached_file)

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(cache_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder
            cached_zipfile.extractall(cache_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    return repo_dir


def import_module(name, path):
    if sys.version_info >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    elif sys.version_info >= (3, 0):
        from importlib.machinery import SourceFileLoader
        return SourceFileLoader(name, path).load_module()
    else:
        import imp
        return imp.load_source(name, path)


def _load_attr_from_module(module, func_name):
    # Check if callable is defined in the module
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)


def _load_entry_from_module(m, model):
    if not isinstance(model, str):
        raise ValueError('Invalid input: model should be a string of function name')

    func = _load_attr_from_module(m, model)

    if func is None or not callable(func):
        raise RuntimeError('Cannot find callable {} in module'.format(model))

    return func


def save_model(python_model, path='./model/', conda_env=None, dependencies=[], github = None, module_path = None, model_class= None):
    """
    Save a generic python model to a path on the local file system.

    :param python_model: Python model to be saved. 

    :param path: Path to a directory saving model data.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 

    :param dependencies: artifacts to be copied to model path. 
    """
    if (not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # call model save function
    python_model.save(path)

    if conda_env is None:
        conda_env = _get_default_conda_env()
    print(f'path={path}, conda_env={conda_env}')
    utils.save_conda_env(path, conda_env)

    for dependency in dependencies:
        shutil.copy(dependency, path)

    func = getattr(python_model, 'predict')
    if func is None:
        raise RuntimeError('Cannot find predict function in model')

    args = inspect.getargspec(func).args
    if 'self' in args:
        args.remove('self')

    spec = utils.generate_default_model_spec(FLAVOR_NAME, MODEL_FILE_NAME, input_args=args)
    pySpec = spec[FLAVOR_NAME]
    if github is not None: pySpec["github"] = github
    if module_path is not None: pySpec["module_path"] = module_path
    if model_class is not None: pySpec["model_class"] = model_class
    utils._save_model_spec(path, spec)
    utils.generate_ilearner_files(path)  # temp solution, to remove later

def load_model(model_path, github = None, module_path = None, model_class = None, *args, **kwargs):
    r"""
    Load a model from a github repo.
    Args:
        github: Required, a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'StudioCommunity/CustomModules[:branch]'
        modulepath: Required, a string of file: model.py
        model: Required, a string of entrypoint name defined in model.py
        *args: Optional, the corresponding args for callable `model`.
        force_reload: Optional, whether to force a fresh download of github repo unconditionally.
            Default is `False`.
        **kwargs: Optional, the corresponding kwargs for callable `model`.
    Returns:
        a single PythonModel with a predict function.
    Example:
        model = builtin_models.load('StudioCommunity/CustomModules:master', 'dstest/dstest/python/dummy.py', 'DummyPythonModel', pretrained=True)
    """
    if(model_path is None):
        raise RuntimeError("Invalid model_path")

    if github is not None:
        # Setup cache_dir to save downloaded files
        _setup_cache_dir()

        force_reload = kwargs.get('force_reload', False)
        kwargs.pop('force_reload', None)

        repo_dir = _get_cache_or_reload(github, force_reload)
        
        print(f'adding {repo_dir} to sys path')
        sys.path.insert(0, repo_dir)

        print(f'try load model from {repo_dir}, {module_path}, {model_class}')
        module = import_module(module_path, repo_dir + '/' + module_path)
        entry = _load_entry_from_module(module, model_class)
        print(f'initialize entry with: {args}, {kwargs}')
        model = entry(model_path, *args, **kwargs)
        print(f"removing {repo_dir} from sys path")
        sys.path.remove(repo_dir)

    elif module_path is None or model_class is None:
        print(f'try load cloudpickle model directly from {model_path}')
        model = _load_model(model_path)
    else:
        print(f'try load model from {module_path}, {model_class}')
        #module = import_module(module_path, module_path)
        module = importlib.import_module(module_path)
        entry = _load_entry_from_module(module, model_class)
        print(f'initialize entry with: {args}, {kwargs}')
        model = entry(model_path, *args, **kwargs)

    return model


def _test_dummy_model():
    import numpy as np
    import pandas as pd
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    d = {'x': x, 'y': y}
    model = DummyPythonModel()
    model.b = 1
    result = model.predict(x, y)
    print(f"result: {result}")

    model_path = "../dstest/model/python/dummy/"
    save_model(model, model_path)
    model1 = load_model(model_path, module_path="builtin_models.python", model_class="DummyPythonModel")
    result = model1.predict(x, y)
    print(f"result: {result}")
    
    # test_github_based_model
    github = 'StudioCommunity/CustomModules:master'
    module = 'builtin-models/builtin_models/python.py'
    model2 = load_model("../dstest/model/python/dummy/", github = github, module_path = module, model_class= 'DummyPythonModel', force_reload= True)
    result = model2.predict(x, y)
    print(f"result: {result}")

# python -m builtin_models.python
if __name__ == '__main__':
    _test_dummy_model()
    