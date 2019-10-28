import inspect
import os
import shutil
import ast
import sys
import importlib

import cloudpickle
import torch
import torchvision
import yaml
import pandas as pd

from .generic import GenericModel
from .flavor import Flavor
from .model_input import ModelInput
from .model_output import ModelOutput
from .logger import get_logger
from . import utils
from . import constants

logger = get_logger(__name__)

FLAVOR_NAME = "pytorch"
MODEL_FILE_NAME = "model.pkl"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PytorchFlavor(Flavor):

    def __init__(
        self,
        model_file_path: str,
        pytorch_version: str = torch.__version__,
        torchvision_version: str = torchvision.__version__,
        serialization_format: str = "cloudpickle",
        serialization_library_version: str = cloudpickle.__version__):
        self.name = "pytorch"
        self.model_file_path = model_file_path
        self.pytorch_version = pytorch_version
        self.torchvision_version = torchvision_version
        self.serialization_format = serialization_format
        self.serialization_library_version = serialization_library_version


def _get_default_conda_env(additional_pip_deps=[]):
    return utils.generate_conda_env(
        additional_conda_deps=[
            "pytorch={}".format(torch.__version__),
            "torchvision={}".format(torchvision.__version__),
        ],
        additional_pip_deps=additional_pip_deps +
            ["cloudpickle=={}".format(cloudpickle.__version__)],
        additional_conda_channels=[
            "pytorch"
        ])


# TODO: Investigate is it possible to substitute with "torch.save" since we copy all the code now
def _save(pytorch_model, path):
    with open(path, 'wb') as fp:
        cloudpickle.dump(pytorch_model, fp)


def save(
    pytorch_model,
    path: str ="./AzureMLModel",
    conda_env: dict = None,
    additional_pip_deps: list = [],
    local_dependency_path: str = None,
    inputs: list = None,
    outputs: list = None,
    exist_ok: bool = False):
    os.makedirs(path, exist_ok=exist_ok)
    _save(pytorch_model, os.path.join(path, MODEL_FILE_NAME))

    # TODO: Provide the option to save result of "conda env export"
    if conda_env is None:
        conda_env = _get_default_conda_env(additional_pip_deps=additional_pip_deps)
    utils.save_conda_env(path, conda_env)

    # In the cases where customer manually modified sys.path (e.g. sys.path.append("..")), 
    # they would have to specify the code path manually.
    if not local_dependency_path:
        local_dependency_path = os.path.abspath(sys.path[0])
        logger.info(f"using sys.path[0] = {sys.path[0]} as local_dependency_path")
    dst_code_path = os.path.join(path, constants.LOCAL_DEPENDENCY_PATH)
    utils._copytree_include(local_dependency_path, dst_code_path, include_extensions=(".py"), exist_ok=exist_ok)

    # TODO: Parse input/output schema from test data
    if inputs is None:
        try:
            forward_func = getattr(pytorch_model, 'forward')
            args = inspect.getfullargspec(forward_func).args
            if 'self' in args:
                args.remove('self')
            inputs = []
            for arg in args:
                inputs.append(ModelInput(arg, "ndarray"))
        except AttributeError:
            logger.warning("Model without 'forward' function cannot be used to predict", exc_info=True)
    
    flavor = PytorchFlavor(
        model_file_path=MODEL_FILE_NAME,
        pytorch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        serialization_format="cloudpickle",
        serialization_library_version=cloudpickle.__version__
    )

    model_spec = utils.generate_model_spec(
        flavor=flavor,
        conda_file_path=constants.CONDA_FILE_NAME,
        local_dependency=constants.LOCAL_DEPENDENCY_PATH,
        inputs=inputs
    )
    utils.save_model_spec(path, model_spec)
    utils.generate_ilearner_files(path) # temp solution, to remove later


def _load_from_cloudpickle(model_path, pytorch_conf):
    with open(model_path, "rb") as fp:
        model = cloudpickle.load(fp)
    return model


def _load_from_savedmodel(model_path, pytorch_conf):
    model_file_path = os.path.join(model_path, pytorch_conf['model_file'])
    model = torch.load(model_file_path, map_location=device)
    return model


def _load_from_saveddict(model_path, pytorch_conf):
    model_class_package = pytorch_conf['model_class_package']
    model_class_name = pytorch_conf['model_class_name']
    model_class_init_args = os.path.join(model_path, pytorch_conf['model_class_init_args'])
    model_file_path = os.path.join(model_path, pytorch_conf['model_file'])
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
    model_conf = utils.get_configuration(artifact_path)
    pytorch_conf = model_conf["flavor"]
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
        if isinstance(entry, str):
            entry = ast.literal_eval(entry)
        return torch.Tensor(list(entry)).to(self.device)
    
    def predict(self, df):
        logger.info(f"Device type = {self.device}, input_args = {self.input_args}")
        output = []

        with torch.no_grad():
            logger.info(f"input_df =\n {df}")
            # TODO: Consolidate serveral rows together for efficiency
            for _, row in df.iterrows():
                input_params = list(map(self._to_tensor, row[self.input_args]))
                predicted = self.model(*input_params)
                output.append(predicted.tolist())

        output_df = pd.DataFrame(output)
        output_df.columns = [f"Score_{i}" for i in range(0, output_df.shape[1])]
        logger.info(f"output_df =\n{output_df}")
        return output_df


def _load_generic_model(artifact_path) -> _PytorchWrapper:
    model_conf = utils.get_configuration(artifact_path)
    is_gpu = torch.cuda.is_available()   
    inputs = model_conf.get('inputs', None)
    input_args = None
    if inputs:
        input_args = [model_input["name"] for model_input in inputs]
    model = load(artifact_path)
    return _PytorchWrapper(model, is_gpu, input_args)