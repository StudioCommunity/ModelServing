import inspect
import logging 
import os
import shutil
import ast
import imp
import sys
import importlib

import cloudpickle
import torch
import torchvision
import yaml
import pandas as pd

from .generic import GenericModel
from . import utils

logger = logging.getLogger(__name__)

FLAVOR_NAME = "pytorch"
MODEL_FILE_NAME = "model.pkl"
CODE_FOLDER_NAME = "code"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_conda_deps=[
            "pytorch={}".format(torch.__version__),
            "torchvision={}".format(torchvision.__version__),
        ],
        additional_pip_deps=[
            "cloudpickle=={}".format(cloudpickle.__version__)
        ],
        additional_conda_channels=[
            "pytorch"
        ])


# TODO: Investigate is it possible to substitute with "torch.save" since we copy all the code now
def _save(pytorch_model, path):
    with open(path, 'wb') as fp:
        cloudpickle.dump(pytorch_model, fp)


def save(pytorch_model, path="./AzureMLModel", conda_env=None, code_path=None, exist_ok=False):
    os.makedirs(path, exist_ok=exist_ok)
    _save(pytorch_model, os.path.join(path, MODEL_FILE_NAME))

    if conda_env is None:
        conda_env = _get_default_conda_env()
    utils.save_conda_env(path, conda_env)

    if code_path is not None:
        dst_code_path = os.path.join(path, CODE_FOLDER_NAME)
        utils._copytree_include(code_path, dst_code_path, include_extensions=(".py"))

    forward_func = getattr(pytorch_model, 'forward')
    args = inspect.getargspec(forward_func).args
    if 'self' in args:
        args.remove('self')

    utils.save_model_spec(path, FLAVOR_NAME, MODEL_FILE_NAME, input_args=args, code_path=CODE_FOLDER_NAME)
    utils.generate_ilearner_files(path) # temp solution, to remove later


def _load_from_cloudpickle(model_path, pytorch_conf):
    with open(model_path, "rb") as fp:
        model = cloudpickle.load(fp)
    return model


def _load_from_savedmodel(model_path, pytorch_conf):
    model_file_path = os.path.join(model_path, pytorch_conf['model_file_path'])
    model = torch.load(model_file_path, map_location=device)
    return model


def _load_from_saveddict(model_path, pytorch_conf):
    model_class_package = pytorch_conf['model_class_package']
    model_class_name = pytorch_conf['model_class_name']
    model_class_init_args = os.path.join(model_path, pytorch_conf['model_class_init_args'])
    model_file_path = os.path.join(model_path, pytorch_conf['model_file_path'])
    module = importlib.import_module(model_class_package)
    model_class = getattr(module, model_class_name)
    logger.info(f'model_class_init_args = {model_class_init_args}')
    with open(model_class_init_args, 'rb') as fp:
        import pickle
        args = pickle.load(fp)
    model = model_class(*args)
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    return model
    

def load(artifact_path="./AzureMLModel") -> torch.nn.Module:
    model_conf = utils._get_configuration(artifact_path)
    utils.add_code_path_to_syspath(artifact_path, model_conf)
    pytorch_conf = model_conf['pytorch']
    model_path = os.path.join(artifact_path, pytorch_conf['model_file_path'])
    serializer = pytorch_conf.get('serialization_format', 'cloudpickle')
    if serializer == 'cloudpickle':
        model = _load_from_cloudpickle(model_path, pytorch_conf)
    elif serializer == 'savedmodel':
        model = _load_from_savedmodel(model_path, pytorch_conf)
    elif serializer == 'saveddict':
        model = _load_from_saveddict(model_path, pytorch_conf)
    else:
        raise Exception(f"Unrecognized serializtion format {serializer}")
    logger.info('Load model success.')
    return model


class _PytorchWrapper(GenericModel):
    def __init__(self, model, is_gpu, input_args):
        self.model = model
        self.device = 'cuda' if is_gpu else 'cpu'
        self.input_args = input_args

    def _to_tensor(self, entry):
        if type(entry) == str:
            entry = ast.literal_eval(entry)
        return torch.Tensor(list(entry)).to(self.device)
    
    def predict(self, df):
        logger.info(f"Device type = {self.device}, input_args = {self.input_args}")
        output = []

        with torch.no_grad():
            logger.info(f"input_df =\n {df}")
            # TODO: Consolidate serveral rows together for efficiency
            for _, row in df.iterrows():
                input_params = []
                input_params = list(map(self._to_tensor, row[self.input_args]))
                predicted = self.model(*input_params)
                output.append(predicted.tolist())

        output_df = pd.DataFrame(output)
        output_df.columns = [f"Score_{i}" for i in range(0, output_df.shape[1])]
        logger.info(f"output_df =\n{output_df}")
        return output_df


def _load_generic_model(artifact_path) -> _PytorchWrapper:
    model_conf = utils._get_configuration(artifact_path)
    is_gpu = torch.cuda.is_available()   
    input_args = model_conf.get('inputs', None)
    model = load(artifact_path)
    return _PytorchWrapper(model, is_gpu, input_args)