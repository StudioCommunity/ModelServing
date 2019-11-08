from . import constants
from .model_wrapper import ModelWrapper
from .model_factory import ModelFactory

def save_generic_model(
    model,
    path: str = "./AzureMLModel",
    conda=None,
    local_dependencies: list = [],
    overwrite_if_exists: bool = True
    ):
    ModelWrapper.save(model=model,
                      artifact_path=path,
                      model_relative_to_artifact_path=constants.CUSTOM_MODEL_DIRECTORY,
                      overwrite_if_exists=overwrite_if_exists)

def load_generic_model(
    path: str = "./AzureMLModel",
    install_dependencies: bool = False
):
    return ModelWrapper.load(artifact_path=path, install_dependencies=install_dependencies)


def save_pytorch_model(
    pytorch_model,
    path: str = "./AzureMLModel",
    conda=None,
    local_dependencies: list = [],
    inputs: list = [],
    outputs: list = [],
    overwrite_if_exists: bool = True
):
    pass
    