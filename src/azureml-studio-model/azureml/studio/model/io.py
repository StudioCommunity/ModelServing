from . import constants
from .generic_model import GenericModel


def save_generic_model(
    model,
    path: str = "./AzureMLModel",
    conda=None,
    local_dependencies: list = [],
    inputs: list = [],
    outputs: list = [],
    serving_config: dict = None,
    overwrite_if_exists: bool = True
    ):
    generic_model = GenericModel(
        core_model = model,
        conda = conda,
        local_dependencies=local_dependencies,
        inputs=inputs,
        outputs=outputs,
        serving_config=serving_config
    )
    generic_model.save(
        artifact_path=path,
        model_relative_to_artifact_path=constants.CUSTOM_MODEL_DIRECTORY
    )

def load_generic_model(
    path: str = "./AzureMLModel",
    install_dependencies: bool = False
):
    return GenericModel.load(artifact_path=path, install_dependencies=install_dependencies)


def save_pytorch_model(
    pytorch_model,
    path: str = "./AzureMLModel",
    conda=None,
    local_dependencies: list = [],
    inputs: list = [],
    outputs: list = [],
    serving_config: dict = None,
    overwrite_if_exists: bool = True
):
    from .pytorch.cloudpickle import PytorchCloudPickleModel
    model = PytorchCloudPickleModel(pytorch_model, next(pytorch_model.parameters()).is_cuda)
    logger.info(f"Saving model with is_cuda={next(pytorch_model.parameters()).is_cuda}")
    generic_model = GenericModel(
        core_model = model,
        conda = conda,
        local_dependencies=local_dependencies,
        inputs=inputs,
        outputs=outputs,
        serving_config=serving_config
    )
    generic_model.save(
        artifact_path=path,
        model_relative_to_artifact_path=constants.PYTORCH_MODEL_FILE_NAME
    )
