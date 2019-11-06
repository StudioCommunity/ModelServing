import os
import sys

from .generic import GenericModel

import yaml

from . import constants
from . import utils
from .local_dependency import LocalDependencyManager
from .remote_dependency import RemoteDependencyManager
from .logger import get_logger

logger = get_logger(__name__)


class ModelWrapper(object):

    def __init__(self):
        pass

    @classmethod
    def save(
        cls,
        model: GenericModel,
        path: str ="./AzureMLModel",
        exist_ok: bool = True
        ):
        os.makedirs(path, exist_ok=exist_ok)

        # TODO: Provide the option to save result of "conda env export"
        if model.conda is not None:
            utils.save_conda_env(path, model.conda)
        else:
            # TODO: merge additional_conda_env with conda_env
            pass
           
        # In the cases where customer manually modified sys.path (e.g. sys.path.append("..")), 
        # they would have to specify the code path manually.
        if not model.local_dependencies:
            model.local_dependencies = [os.path.abspath(sys.path[0])]
            logger.info(f"using sys.path[0] = {sys.path[0]} as local_dependency_path")
        local_dependency_manager = LocalDependencyManager(model.local_dependencies)
        local_dependency_manager.save(path)

        model_spec = utils.generate_model_spec(
            flavor=model.flavor,
            conda_file_path=constants.CONDA_FILE_NAME,
            local_dependencies=local_dependency_manager.copied_local_dependencies,
            inputs=model.inputs
        )
        utils.save_model_spec(path, model_spec)


    def load(self, artifact_path="./AzureMLModel", install_dependencies=False) -> GenericModel:
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
            conda_yaml_path = os.path.join(artifact_path, config["conda_file"])

            local_dependencies = config.get("local_dependencies", None)
            local_dependency_manager = LocalDependencyManager(local_dependencies)
            local_dependency_manager.load(artifact_path, local_dependencies)
            local_dependency_manager.install()

            remote_dependency_manager = RemoteDependencyManager()
            remote_dependency_manager.load(conda_yaml_path)
            remote_dependency_manager.install()
        
        model_conf = utils.get_configuration(artifact_path)
        flavor_name = model_conf["flavor"]["name"].lower()
        if flavor_name == "pytorch":
            from .pytorch import _load_generic_model
            return _load_generic_model(artifact_path)
        elif flavor_name == "tensorflow":
            pass
        elif flavor_name == "sklearn":
            pass
        elif flavor_name == "keras":
            pass
        elif flavor_name == "python":
            pass
        elif flavor_name == "onnx":
            pass
        else:
            msg = f"Not Implemented: flavor {flavor_name} not supported"
            logger.info(msg)
            raise ValueError(msg)
