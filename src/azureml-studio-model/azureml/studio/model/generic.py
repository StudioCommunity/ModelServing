import os
import logging

import yaml

from . import constants
from .dependency import DependencyManager
from .utils import _get_configuration

logger = logging.getLogger(__name__)

class GenericModel(object):
    
    def __init__(self):
        pass
    
    def predict(self, df):
        pass

def save(model_path):
    pass

def load(artifact_path="./AzureMLModel", config={}, install_dependencies=False) -> GenericModel:
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
    else:
        msg = f"Not Implemented: framework {framework} not supported"
        logger.info(msg)
        raise ValueError(msg)
