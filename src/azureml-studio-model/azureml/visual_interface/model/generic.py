import os
import logging
from abc import ABC, abstractmethod

import yaml

from . import constants
from .dependency import DependencyManager
from .utils import _get_configuration

logger = logging.getLogger(__name__)

class GenericModel(ABC):
    """Interface class to be inherited by wrapper of different flavors, and expose unifed init and predict function
    
    Arguments:
        ABC {[type]} -- [description]
    """
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, df):
        pass


def load(artifact_path="./AzureMLModel", install_dependencies=False) -> GenericModel:
    """Load model as GenericModel
    
    Keyword Arguments:
        artifact_path {str} -- path to the ModelDirectory (default: {"./AzureMLModel"})
        install_dependencies {bool} -- if true, this function will try to install the dependencies
            specified by conda.yaml in the artifact_path (default: {False})
    
    Raises:
        ValueError: [description]
    
    Returns:
        GenericModel -- [description]
    """
    model_spec_path = os.path.join(artifact_path, constants.MODEL_SPEC_FILE_NAME)
    logger.info(f'MODEL_FOLDER: {os.listdir(artifact_path)}')
    with open(model_spec_path) as fp:
        config = yaml.safe_load(fp)
    
    if install_dependencies:
        conda_yaml_path = os.path.join(artifact_path, config["conda"]["conda_file_path"])
        dependency_manager = DependencyManager()
        dependency_manager.load(conda_yaml_path)
        dependency_manager.install()
    
    model_conf = _get_configuration(artifact_path)
    framework = model_conf["flavor"]["framework"]
    if framework.lower() == "pytorch":
        from .pytorch import _load_generic_model
        return _load_generic_model(artifact_path)
    elif framework.lower() == "tensorflow":
        pass
    elif framework.lower() == "sklearn":
        pass
    elif framework.lower() == "keras":
        pass
    elif framework.lower() == "python":
        pass
    elif framework.lower() == "onnx":
        pass
    else:
        msg = f"Not Implemented: framework {framework} not supported"
        logger.info(msg)
        raise ValueError(msg)
