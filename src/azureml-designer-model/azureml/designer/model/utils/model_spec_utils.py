import os

import yaml
from datetime import datetime

from .. import constants
from ..model_spec.serving_config import ServingConfig
from ..logger import get_logger

logger = get_logger(__name__)


def generate_model_spec(
        flavor: dict,
        model_path: str = constants.CUSTOM_MODEL_DIRECTORY,
        conda_file_path: str = constants.CONDA_FILE_NAME,
        local_dependencies: list = [],
        inputs: list = None,
        outputs: list = None,
        serving_config: ServingConfig = None,
        time_created: datetime = datetime.now()
):
    spec = {
        "flavor": flavor,
        "model_path": model_path,
        "local_dependencies": local_dependencies,
        "time_created": time_created.strftime("%Y-%m-%d %H:%M:%S")
    }
    if conda_file_path:
        spec["conda_file"]: conda_file_path
    if inputs is not None:
        spec["inputs"] = [model_input.to_dict() for model_input in inputs]
    if outputs is not None:
        spec["outputs"] = [model_output.to_dict() for model_output in outputs]
    if serving_config:
        spec["serving_config"] = serving_config.to_dict()
    logger.info(f"spec = {spec}")
    return spec


def save_model_spec(path, spec):
    logger.info(f'MODEL_SPEC: {spec}')
    with open(os.path.join(path, constants.MODEL_SPEC_FILE_NAME), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)
    fn = os.path.join(path, constants.MODEL_SPEC_FILE_NAME)
    logger.info(f'SAVED MODEL_SPEC: {fn}')


def get_configuration(artifact_path) -> dict:
    model_spec_path = os.path.join(artifact_path, constants.MODEL_SPEC_FILE_NAME)
    if not os.path.exists(model_spec_path):
        raise Exception(f"Could not find {constants.MODEL_SPEC_FILE_NAME} in {artifact_path}")
    with open(model_spec_path) as fp:
        model_conf = yaml.safe_load(fp)
    return model_conf
