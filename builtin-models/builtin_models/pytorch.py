import os
import shutil
import yaml
import cloudpickle
import inspect
import torch
import torchvision
import builtin_models.utils as utils

FLAVOR_NAME = "pytorch"
MODEL_FILE_NAME = "model.pkl"


def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_pip_deps=[
            "torch=={}".format(torch.__version__),
            "torchvision=={}".format(torchvision.__version__),
        ])


def _save_model(pytorch_model, path):
    with open(path, 'wb') as fp:
        cloudpickle.dump(pytorch_model, fp)


def save_model(pytorch_model, path='./model/', conda_env=None, dependencies=[]):
    """
    Save a PyTorch model to a path on the local file system.

    :param pytorch_model: PyTorch model to be saved. 

    :param path: Path to a directory containing model data.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # only save cpu version
    _save_model(pytorch_model.to('cpu'), os.path.join(path, MODEL_FILE_NAME))
    fn = os.path.join(path, MODEL_FILE_NAME)
    print(f'MODEL_FILE: {fn}')
    
    if conda_env is None:
        conda_env = _get_default_conda_env()
    print(f'path={path}, conda_env={conda_env}')
    utils.save_conda_env(path, conda_env)

    for dependency in dependencies:
        shutil.copy(dependency, path)
    forward_func = getattr(pytorch_model, 'forward')
    args = inspect.getargspec(forward_func).args
    if 'self' in args:
        args.remove('self')

    utils.save_model_spec(path, FLAVOR_NAME, MODEL_FILE_NAME, input_args = args)
    utils.generate_ilearner_files(path) # temp solution, to remove later


    