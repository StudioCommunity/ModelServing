import os

from datetime import datetime

from ..constants import ModelSpecConstants
from ..model_spec.serving_config import ServingConfig
from ..model_spec.task_type import TaskType
from ..logger import get_logger
from ..utils import yamlutils

logger = get_logger(__name__)


def generate_model_spec(
        flavor: dict,
        model_path: str = ModelSpecConstants.CUSTOM_MODEL_DIRECTORY,
        conda_file_path: str = ModelSpecConstants.CONDA_FILE_NAME,
        local_dependencies: list = [],
        inputs: list = None,
        outputs: list = None,
        task_type: TaskType = None,
        label_map_path: str = None,
        serving_config: ServingConfig = None,
        time_created: datetime = datetime.now()
):
    spec = {
        ModelSpecConstants.FLAVOR_KEY: flavor,
        ModelSpecConstants.MODEL_FILE_KEY: model_path,
        ModelSpecConstants.LOCAL_DEPENDENCIES_KEY: local_dependencies,
        ModelSpecConstants.TIME_CREATED_KEY: time_created.strftime("%Y-%m-%d %H:%M:%S")
    }
    if conda_file_path:
        spec[ModelSpecConstants.CONDA_FILE_KEY]: conda_file_path
    if inputs is not None:
        spec[ModelSpecConstants.INPUTS_KEY] = [model_input.to_dict() for model_input in inputs]
    if outputs is not None:
        spec[ModelSpecConstants.OUTPUTS_KEY] = [model_output.to_dict() for model_output in outputs]
    if task_type:
        spec[ModelSpecConstants.TASK_TYPE_KEY] = task_type.name
    if label_map_path:
        spec[ModelSpecConstants.LABEL_MAP_FILE_KEY] = label_map_path
    if serving_config:
        spec[ModelSpecConstants.SERVING_CONFIG_KEY] = serving_config.to_dict()
    logger.info(f"spec = {spec}")
    return spec


def save_model_spec(path, spec):
    logger.info(f'MODEL_SPEC: {spec}')
    yamlutils.dump_to_yaml_file(spec, os.path.join(path, ModelSpecConstants.MODEL_SPEC_FILE_NAME))
    fn = os.path.join(path, ModelSpecConstants.MODEL_SPEC_FILE_NAME)
    logger.info(f'SAVED MODEL_SPEC: {fn}')


def get_configuration(artifact_path) -> dict:
    model_spec_path = os.path.join(artifact_path, ModelSpecConstants.MODEL_SPEC_FILE_NAME)
    if not os.path.exists(model_spec_path):
        raise Exception(f"Could not find {ModelSpecConstants.MODEL_SPEC_FILE_NAME} in {artifact_path}")
    model_conf = yamlutils.load_yaml_file(model_spec_path)
    return model_conf
