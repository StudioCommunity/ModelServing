import os
import fnmatch
import yaml
import json
import logging
import shutil
import sys
from sys import version_info

from . import constants

logger = logging.getLogger(__name__)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)


def generate_conda_env(path=None, additional_conda_deps=None, additional_pip_deps=None,
                      additional_conda_channels=None, install_azureml=True):
    env = {
        'name' : 'project_environment',
        'channels': ['defaults'],
        'dependencies': [
            "python={}".format(PYTHON_VERSION),
            "git",
            "regex"
        ]
    }
    pip_dependencies = []
    if additional_pip_deps is not None:
        pip_dependencies.extend(additional_pip_deps)
    env["dependencies"].append({"pip": pip_dependencies})
    if additional_conda_deps is not None:
        env["dependencies"] += additional_conda_deps
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env


def save_conda_env(path, conda_env):
    logger.info(f"saving conda to {path}:\n{conda_env}")
    if conda_env is None:
        raise Exception("conda_env is empty")
    if isinstance(conda_env, str) and os.path.isfile(conda_env):
            with open(conda_env, "r") as f:
                conda_env = yaml.safe_load(f)
    if not isinstance(conda_env, dict):
        raise Exception("Could not load conda_env %s" % conda_env)
    logger.info(f'CONDA: {conda_env}')
    with open(os.path.join(path, constants.CONDA_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    fn = os.path.join(path, constants.CONDA_FILE_NAME)
    logger.info(f'CONDA_FILE: {fn}')


def generate_default_model_spec(flavor_name, model_file_name, conda_file_name=constants.CONDA_FILE_NAME, input_args=[], code_path=None):
    """
    Generate default model spec
    
    :flavor_name

    :model_file_name

    :conda_file_name (optional)

    :input_args (optional)
    """
    spec = {
        'flavor' : {
            'framework' : flavor_name
        },
        flavor_name: {
            'model_file_path': model_file_name
        },
        'conda': {
            'conda_file_path': conda_file_name
        },
    }
    if input_args:
        spec['inputs'] = input_args
    if code_path:
        spec["code_path"] = code_path
    logger.info(f'SPEC={spec}')
    return spec


def _save_model_spec(path, spec):
    logger.info(f'MODEL_SPEC: {spec}')
    with open(os.path.join(path, constants.MODEL_SPEC_FILE_NAME), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)
    fn = os.path.join(path, constants.MODEL_SPEC_FILE_NAME)
    print(f'SAVED MODEL_SPEC: {fn}')


# TODO: Support multiple code paths
def save_model_spec(path, flavor_name, model_file_name, conda_file_name=constants.CONDA_FILE_NAME, input_args=[], code_path=None):
    """
    Save model spec to local file

    :path

    :flavor_name

    :model_file_name

    :conda_file_name (optional)

    :input_args (optional)
    """
    spec = generate_default_model_spec(flavor_name, model_file_name, conda_file_name, input_args, code_path)
    _save_model_spec(path, spec)


def generate_ilearner_files(path):
    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "ILearnerDotNet",
        "Name": "ILearner .NET file",
        "ShortName": "Model",
        "Description": "A .NET serialized ILearner",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "ilearner",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": False,
        "AllowModelPromotion": True,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(path, constants.DATA_TYPE_FILE_NAME), 'w') as fp:
        json.dump(dct, fp)
    # Dump data.ilearner as a work around until data type design
    with open(os.path.join(path, constants.DATA_ILEARNER_FILE_NAME), 'w') as fp:
        fp.writelines('{}')


def _get_configuration(artifact_path) -> dict:
    model_spec_path = os.path.join(artifact_path, constants.MODEL_SPEC_FILE_NAME)
    if not os.path.exists(model_spec_path):
        raise Exception(f"Could not find {constants.MODEL_SPEC_FILE_NAME} in {artifact_path}")
    with open(model_spec_path) as fp:
        model_conf = yaml.safe_load(fp)
    return model_conf


def _copytree_include(src_dir, dst_dir, include_extensions: tuple = (), exist_ok=False):
    os.makedirs(dst_dir, exist_ok=exist_ok)
    # Scan and list all included files before copying to avoid recursion
    file_list = []
    src_dir_len = len(src_dir)
    for root, _, files in os.walk(src_dir, topdown=True):
        for name in files:
            if name.endswith(include_extensions):
                file_list.append(os.path.join(root[src_dir_len:].strip('/').strip('\\'), name))
    
    logger.info(f"file_list = {file_list}")
    for file_path in file_list:
        src_file_path = os.path.join(src_dir, file_path)
        dst_file_path = os.path.join(dst_dir, file_path)
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
        shutil.copy(src_file_path, dst_file_path)


def add_code_path_to_syspath(artifact_path, model_conf):
    if "code_path" in model_conf:
        code_path = os.path.join(artifact_path, model_conf["code_path"])
        sys.path.append(code_path)
        logger.info(f"Added {code_path} to sys.path")