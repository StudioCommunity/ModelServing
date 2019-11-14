import os
import shutil

import yaml

from .. import constants
from ..logger import get_logger

logger = get_logger(__name__)


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
