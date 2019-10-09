import os
import logging
from abc import ABC, abstractmethod

import yaml

from . import constants
from .dependency import DependencyManager
from . import utils

logger = logging.getLogger(__name__)

class GenericModel(ABC):
    """Interface class to be inherited by wrapper of different flavors, and expose unifed init and predict function
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
        conda_yaml_path = os.path.join(artifact_path, config["conda_file_path"])
        local_dependency_path = config.get("local_dependency_path", None)
        # TODO: Handle the case where local_dependency_path doesn't exist
        if local_dependency_path:
            local_dependency_path = os.path.join(artifact_path, local_dependency_path)
        dependency_manager = DependencyManager()
        dependency_manager.load(conda_yaml_path, local_dependency_path)
        dependency_manager.install()
    
    model_conf = utils.get_configuration(artifact_path)
    vintage = model_conf[constants.VINTAGE_KEY].lower()
    if vintage == "pytorch":
        from .pytorch import _load_generic_model
        return _load_generic_model(artifact_path)
    elif vintage == "tensorflow":
        pass
    elif vintage == "sklearn":
        pass
    elif vintage == "keras":
        pass
    elif vintage == "python":
        pass
    elif vintage == "onnx":
        pass
    else:
        msg = f"Not Implemented: {constants.VINTAGE_KEY} {vintage} not supported"
        logger.info(msg)
        raise ValueError(msg)
