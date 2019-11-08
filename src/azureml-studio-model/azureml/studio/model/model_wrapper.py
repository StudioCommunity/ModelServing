import os
import sys

from .generic_model import GenericModel
from .builtin_model import BuiltinModel

import yaml

from . import constants
from . import utils
from .local_dependency import LocalDependencyManager
from .remote_dependency import RemoteDependencyManager
from .logger import get_logger
from .model_factory import ModelFactory

logger = get_logger(__name__)


class ModelWrapper(object):

    @classmethod
    def save(
        cls,
        model: GenericModel,
        artifact_path: str = "./AzureMLModel",
        model_relative_to_artifact_path : str = "model",
        overwrite_if_exists: bool = True
        ):
        os.makedirs(artifact_path, exist_ok=overwrite_if_exists)
        model_path = os.path.join(artifact_path, model_relative_to_artifact_path)
        model.save(model_path, overwrite_if_exists=overwrite_if_exists)

        # TODO: Provide the option to save result of "conda env export"
        if model.conda:
            # TODO: merge additional_conda_env with conda_env
            utils.save_conda_env(artifact_path, model.conda)
        else:
            # TODO: dump local conda env
            pass
           
        # In the cases where customer manually modified sys.path (e.g. sys.path.append("..")), 
        # they would have to specify the code path manually.
        if not model.local_dependencies:
            model.local_dependencies = [os.path.abspath(sys.path[0])]
            logger.info(f"using sys.path[0] = {sys.path[0]} as local_dependency_path")
        local_dependency_manager = LocalDependencyManager(model.local_dependencies)
        local_dependency_manager.save(artifact_path)

        if not isinstance(model, BuiltinModel):
            model.flavor = {
                "name": constants.CUSTOM_MODEL_FLAVOR_NAME,
                "module": model.__class__.__module__,
                "class": model.__class__.__name__
            }

        model_spec = utils.generate_model_spec(
            flavor=model.flavor,
            model_path=model_relative_to_artifact_path,
            conda_file_path=constants.CONDA_FILE_NAME,
            local_dependencies=local_dependency_manager.copied_local_dependencies,
            inputs=model.inputs,
            outputs=model.outputs
        )
        utils.save_model_spec(artifact_path, model_spec)


    @classmethod
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
        logger.info(f"MODEL_FOLDER: {os.listdir(artifact_path)}")
        with open(model_spec_path) as fp:
            config = yaml.safe_load(fp)
            logger.info(f"Successfully loaded {model_spec_path}")
        
        if install_dependencies:
            conda_yaml_path = os.path.join(artifact_path, config["conda_file"])

            local_dependencies = config.get("local_dependencies", None)
            local_dependency_manager = LocalDependencyManager(local_dependencies)
            local_dependency_manager.load(artifact_path, local_dependencies)
            local_dependency_manager.install()

            remote_dependency_manager = RemoteDependencyManager()
            remote_dependency_manager.load(conda_yaml_path)
            remote_dependency_manager.install()
        
        flavor = config["flavor"]
        raw_model_path = os.path.join(artifact_path, config["model_path"])
        model = ModelFactory.load_model(raw_model_path, flavor, inputs)