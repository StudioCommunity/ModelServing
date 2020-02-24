from . import constants
from .constants import ModelSpecConstants
from .generic_model import GenericModel
from .logger import get_logger
from .model_spec.task import Task
from .model_spec.task_type import TaskType
from .model_spec.label_map import LabelMap

logger = get_logger(__name__)


def save_generic_model(
    model,
    path: str = ModelSpecConstants.DEFAULT_ARTIFACT_SAVE_PATH,
    conda=None,
    local_dependencies: list = [],
    # TODO provide method to infer input/output schema from sample data
    inputs: list = [],
    outputs: list = [],
    task: Task = None,
    serving_config: dict = None,
    overwrite_if_exists: bool = True
):
    generic_model = GenericModel(
        core_model=model,
        conda=conda,
        local_dependencies=local_dependencies,
        inputs=inputs,
        outputs=outputs,
        task=task,
        serving_config=serving_config
    )
    generic_model.save(
        artifact_path=path,
        model_relative_to_artifact_path=ModelSpecConstants.CUSTOM_MODEL_DIRECTORY,
        overwrite_if_exists=overwrite_if_exists
    )


def load_generic_model(
    path: str = ModelSpecConstants.DEFAULT_ARTIFACT_SAVE_PATH,
    install_dependencies: bool = False
):
    return GenericModel.load(artifact_path=path, install_dependencies=install_dependencies)


def save_pytorch_cloudpickle_model(
    pytorch_model,
    path: str = ModelSpecConstants.DEFAULT_ARTIFACT_SAVE_PATH,
    conda=None,
    local_dependencies: list = [],
    inputs: list = [],
    outputs: list = [],
    task_type: TaskType = None,
    label_map=None,
    ground_truth_column_name=None,
    serving_config: dict = None,
    overwrite_if_exists: bool = True
):
    import torch
    from .builtin_models.pytorch.cloudpickle import PytorchCloudPickleModel
    is_parallel = isinstance(pytorch_model, torch.nn.DataParallel)
    flavor = {
        ModelSpecConstants.IS_CUDA_KEY: next(pytorch_model.parameters()).is_cuda,
        ModelSpecConstants.IS_PARALLEL_KEY: is_parallel
    }
    model = PytorchCloudPickleModel(pytorch_model.module if is_parallel else pytorch_model, flavor)
    logger.info(f"Saving model with is_cuda={next(pytorch_model.parameters()).is_cuda}")

    label_map = LabelMap.create(label_map)
    task = Task(task_type, label_map, ground_truth_column_name)

    generic_model = GenericModel(
        core_model=model,
        conda=conda,
        local_dependencies=local_dependencies,
        inputs=inputs,
        outputs=outputs,
        task=task,
        serving_config=serving_config
    )
    generic_model.save(
        artifact_path=path,
        model_relative_to_artifact_path=ModelSpecConstants.PICKLE_MODEL_FILE_NAME,
        overwrite_if_exists=overwrite_if_exists
    )


def save_pytorch_state_dict_model(
    pytorch_model,
    init_params: dict = {},
    path: str = ModelSpecConstants.DEFAULT_ARTIFACT_SAVE_PATH,
    conda=None,
    local_dependencies: list = [],
    inputs: list = [],
    outputs: list = [],
    task_type: TaskType = None,
    label_map=None,
    ground_truth_column_name=None,
    serving_config: dict = None,
    overwrite_if_exists: bool = True
):
    import torch
    from .builtin_models.pytorch.state_dict import PytorchStateDictModel
    is_parallel = isinstance(pytorch_model, torch.nn.DataParallel)
    flavor = {
        ModelSpecConstants.IS_CUDA_KEY: next(pytorch_model.parameters()).is_cuda,
        ModelSpecConstants.IS_PARALLEL_KEY: is_parallel,
        ModelSpecConstants.INIT_PARAMS_KEY: init_params
    }
    model = PytorchStateDictModel(pytorch_model.module if is_parallel else pytorch_model, flavor)
    logger.info(f"Saving model with flavor: {flavor}")

    label_map = LabelMap.create(label_map)
    task = Task(task_type, label_map, ground_truth_column_name)

    generic_model = GenericModel(
        core_model=model,
        conda=conda,
        local_dependencies=local_dependencies,
        inputs=inputs,
        outputs=outputs,
        task=task,
        serving_config=serving_config
    )
    generic_model.save(
        artifact_path=path,
        model_relative_to_artifact_path=ModelSpecConstants.PYTORCH_STATE_DICT_FILE_NAME,
        overwrite_if_exists=overwrite_if_exists
    )
