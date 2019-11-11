import os
import sys

from abc import abstractmethod, abstractclassmethod
import yaml

from . import constants
from . import utils
from .builtin_model import BuiltinModel
from .local_dependency import LocalDependencyManager
from .logger import get_logger
from .model_factory import ModelFactory
from .model_input import ModelInput
from .model_output import ModelOutput
from .raw_model import RawModel
from .remote_dependency import RemoteDependencyManager
from .resource_config import ResourceConfig

logger = get_logger(__name__)


class GenericModel(object):

    raw_model = None
    flavor = None
    conda = None
    local_dependencies = None
    inputs = None
    outputs = None
    serving_config = None

    def __init__(self, raw_model, flavor, conda=None, local_dependencies=None, inputs=None, outputs=None, serving_config=None):
        self.raw_model = raw_model
        self.flavor = flavor
        self.conda = conda
        self.local_dependencies = local_dependencies
        self.inputs = inputs
        self.outputs = outputs
        self.serving_config = serving_config

    def save(
        self,
        artifact_path: str = "./AzureMLModel",
        model_relative_to_artifact_path : str = "model",
        overwrite_if_exists: bool = True
        ):
        os.makedirs(artifact_path, exist_ok=overwrite_if_exists)
        model_path = os.path.join(artifact_path, model_relative_to_artifact_path)
        self.raw_model.save(model_path, overwrite_if_exists=overwrite_if_exists)

        # TODO: Provide the option to save result of "conda env export"
        if self.conda:
            # TODO: merge additional_conda_env with conda_env
            utils.save_conda_env(artifact_path, self.conda)
        else:
            # TODO: dump local conda env
            pass
           
        # In the cases where customer manually modified sys.path (e.g. sys.path.append("..")), 
        # they would have to specify the code path manually.
        if not self.local_dependencies:
            self.local_dependencies = [os.path.abspath(sys.path[0])]
            logger.info(f"using sys.path[0] = {sys.path[0]} as local_dependency_path")
        local_dependency_manager = LocalDependencyManager(self.local_dependencies)
        local_dependency_manager.save(artifact_path)

        if not isinstance(self.raw_model, BuiltinModel):
            self.flavor = {
                "name": constants.CUSTOM_MODEL_FLAVOR_NAME,
                "module": self.raw_model.__class__.__module__,
                "class": self.raw_model.__class__.__name__
            }

        model_spec = utils.generate_model_spec(
            flavor=self.flavor,
            model_path=model_relative_to_artifact_path,
            conda_file_path=constants.CONDA_FILE_NAME,
            local_dependencies=local_dependency_manager.copied_local_dependencies,
            inputs=self.inputs,
            outputs=self.outputs
        )
        utils.save_model_spec(artifact_path, model_spec)

    @classmethod
    def load(cls, artifact_path, install_dependencies=False):
        model_spec_path = os.path.join(artifact_path, constants.MODEL_SPEC_FILE_NAME)
        logger.info(f"MODEL_FOLDER: {os.listdir(artifact_path)}")
        with open(model_spec_path) as fp:
            config = yaml.safe_load(fp)
            logger.info(f"Successfully loaded {model_spec_path}")
        
        flavor = config["flavor"]
        raw_model_class = ModelFactory.get_model_class(flavor)
        raw_model_path = os.path.join(artifact_path, config["model_path"])
        raw_model = raw_model_class.load(raw_model_path)

        # TODO: Use auxiliary method to handle None in loaded yaml file following Module Team
        if config.get("conda_file", None):
            conda_yaml_path = os.path.join(artifact_path, config["conda_file"])
            with open(conda_yaml_path) as fp:
                conda = yaml.safe_load(fp)
                logger.info(f"Successfully loaded {conda_yaml_path}")
        local_dependencies = config.get("local_dpendencies", None)
        if config.get("inputs", None):
            inputs = [ModelInput.from_dict(model_input) for model_input in config["inputs"]]
        if config.get("outputs", None):
            outputs = [ModelInput.from_dict(model_output) for model_output in config["inputs"]]
        if config.get("serving_config", None):
            serving_config = ResourceConfig.from_dict(config["serving_config"])

        if install_dependencies:
            remote_dependency_manager = RemoteDependencyManager()
            remote_dependency_manager.load(conda_yaml_path)
            remote_dependency_manager.install()

            local_dependency_manager = LocalDependencyManager()
            local_dependency_manager.load(artifact_path, local_dependencies)
            local_dependency_manager.install()
        
        return cls(raw_model, flavor, conda, local_dependencies, inputs, outputs, serving_config)
        
    @abstractmethod
    def predict(self, df):
        pass