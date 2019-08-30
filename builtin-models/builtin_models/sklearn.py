import os
import yaml
import pickle
import sklearn
import builtin_models.utils as utils

FLAVOR_NAME = "sklearn"
model_file_name = "model.pkl"

def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_pip_deps=[
            "scikit-learn=={}".format(sklearn.__version__)
        ])


def _save_model(sklearn_model, path):
    with open(os.path.join(path, model_file_name), "wb") as fb:
        pickle.dump(sklearn_model, fb)


def save_model(sklearn_model, path='./model/', conda_env=None):
    """
    Save a Sklearn model to a path on the local file system.

    :param sklearn_model: Sklearn model to be saved. 

    :param path: Path to a directory containing model data.
    
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    _save_model(sklearn_model, path)

    if conda_env is None:
        conda_env = _get_default_conda_env()
    utils.save_conda_env(path, conda_env)

    utils.save_model_spec(path, FLAVOR_NAME, model_file_name)
    utils.generate_ilearner_files(path) # temp solution, to remove later

    