import os
import sys

from abc import abstractmethod
import yaml

from . import constants
from . import utils
from .utils import ioutils, model_spec_utils
from .logger import get_logger
from .model_factory import ModelFactory
from .builtin_models.builtin_model import BuiltinModel
from .model_spec.local_dependency import LocalDependencyManager
from .model_spec.model_input import ModelInput
from .model_spec.remote_dependency import RemoteDependencyManager
from .model_spec.serving_config import ServingConfig

logger = get_logger(__name__)


class GenericModel(object):

    core_model = None
    conda = None
    local_dependencies = None
    inputs = None
    outputs = None
    serving_config = None

    def __init__(self, core_model, conda=None, local_dependencies=None, inputs=None, outputs=None, serving_config=None):
        self.core_model = core_model
        if not self.core_model.flavor:
            if not isinstance(core_model, BuiltinModel):
                self.core_model.flavor = {
                    "name": constants.CUSTOM_MODEL_FLAVOR_NAME,
                    "module": self.core_model.__class__.__module__,
                    "class": self.core_model.__class__.__name__
                }
            else:
                raise Exception("BuiltinModel Can't be initialized without flavor")
        self.conda = conda
        if not self.conda and isinstance(self.core_model, BuiltinModel):
            self.conda = self.core_model.default_conda
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
        self.core_model.save(model_path, overwrite_if_exists=overwrite_if_exists)

        conda_file_path = None

        # TODO: Provide the option to save result of "conda env export"
        if self.conda:
            ioutils.save_conda_env(artifact_path, self.conda)
            conda_file_path = constants.CONDA_FILE_NAME
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

        model_spec = model_spec_utils.generate_model_spec(
            flavor=self.core_model.flavor,
            model_path=model_relative_to_artifact_path,
            conda_file_path=conda_file_path,
            local_dependencies=local_dependency_manager.copied_local_dependencies,
            inputs=self.inputs,
            outputs=self.outputs
        )
        model_spec_utils.save_model_spec(artifact_path, model_spec)

    @classmethod
    def load(cls, artifact_path, install_dependencies=False):
        model_spec_path = os.path.join(artifact_path, constants.MODEL_SPEC_FILE_NAME)
        logger.info(f"MODEL_FOLDER: {os.listdir(artifact_path)}")
        with open(model_spec_path) as fp:
            model_spec = yaml.safe_load(fp)
            logger.info(f"Successfully loaded {model_spec_path}")
        
        flavor = model_spec["flavor"]
        conda = None
        inputs = None
        outputs = None
        serving_config = None
        
        # TODO: Use auxiliary method to handle None in loaded yaml file following Module Team
        if model_spec.get("conda_file", None):
            conda_yaml_path = os.path.join(artifact_path, model_spec["conda_file"])
            with open(conda_yaml_path) as fp:
                conda = yaml.safe_load(fp)
                logger.info(f"Successfully loaded {conda_yaml_path}")
        local_dependencies = model_spec.get("local_dependencies", None)
        logger.info(f"local_dependencies = {local_dependencies}")
        if model_spec.get("inputs", None):
            inputs = [ModelInput.from_dict(model_input) for model_input in model_spec["inputs"]]
        if model_spec.get("outputs", None):
            outputs = [ModelInput.from_dict(model_output) for model_output in model_spec["inputs"]]
        if model_spec.get("serving_config", None):
            serving_config = ServingConfig.from_dict(model_spec["serving_config"])

        if install_dependencies:
            logger.info("Installing dependencies")
            if conda:
                remote_dependency_manager = RemoteDependencyManager()
                remote_dependency_manager.load(conda_yaml_path)
                remote_dependency_manager.install()

            if local_dependencies:
                local_dependency_manager = LocalDependencyManager()
                local_dependency_manager.load(artifact_path, local_dependencies)
                local_dependency_manager.install()

        core_model_class = ModelFactory.get_model_class(flavor)
        core_model_path = os.path.join(artifact_path, model_spec["model_path"])
        core_model = core_model_class.load(core_model_path)

        if isinstance(core_model, BuiltinModel):
            logger.info("Config BuiltinModel by flavor and inputs")
            core_model.config(model_spec)

        return cls(core_model, conda, local_dependencies, inputs, outputs, serving_config)
        
    # TODO: Support non-dataframe input
    @abstractmethod
    def predict(self, *args, **kwargs):
        # TODO: Some input validation here
        return self.core_model.predict(*args, **kwargs)

    @property
    def raw_model(self):
        if isinstance(self.core_model, BuiltinModel):
            return self.core_model.raw_model
        else:
            return None
