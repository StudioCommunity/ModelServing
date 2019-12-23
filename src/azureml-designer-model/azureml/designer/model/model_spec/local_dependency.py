import os
import sys
import shutil
import tempfile

from ..constants import ModelSpecConstants
from ..utils import ioutils, ziputils
from ..logger import get_logger

logger = get_logger(__name__)


def _is_python_module(directory_path) -> bool:
    """
    Determine whether a directory is python module, i.e. contains __init__.py
    """
    return "__init__.py" in os.listdir(directory_path)


class LocalDependencyManager(object):
    
    def __init__(self, local_dependencies=[]):
        self.local_dependencies = local_dependencies
        self.copied_local_dependencies = []

    def save(self, artifact_path, exist_ok=True) -> list:
        src_abs_paths = [os.path.abspath(_) for _ in self.local_dependencies]

        with tempfile.TemporaryDirectory() as temp_dir_path:
            # Copy pyfiles
            src_py_files = list(filter(lambda x: x.endswith(".py"), src_abs_paths))
            if src_py_files:
                py_filenames = [os.path.split(file_path)[-1] for file_path in src_py_files]
                if len(set(py_filenames)) < len(py_filenames):
                    raise Exception("There are duplication in dependency py file name, which is not allowed.")
                pyfiles_basepath = os.path.join(temp_dir_path, "pyfiles")
                os.makedirs(pyfiles_basepath, exist_ok=True)
                for filename, src_path in zip(py_filenames, src_py_files):
                    shutil.copyfile(src_path, os.path.join(pyfiles_basepath, filename))
                self.copied_local_dependencies.append(os.path.join(ModelSpecConstants.LOCAL_DEPENDENCIES_PATH, "pyfiles"))

                # Copy directories
                src_directories = list(filter(lambda x: not x.endswith(".py"), src_abs_paths))
                if src_directories:
                    dirname_cnt_dict = {}
                    for directory in src_directories:
                        if not os.path.isdir(directory):
                            raise Exception(f"Only py files and directories are supported, got {directory}")

                for src_dir_path in src_directories:
                    is_effective = False
                    dst_dir_name = os.path.split(src_dir_path)[-1]
                    dirname_cnt_dict[dst_dir_name] = dirname_cnt_dict.get(dst_dir_name, 0) + 1
                    if dirname_cnt_dict[dst_dir_name] > 1:
                        dst_dir_name = f"{dst_dir_name}_{dirname_cnt_dict[dst_dir_name] - 1}"
                    dst_dir_path = os.path.join(temp_dir_path, dst_dir_name)
                    for sub_item_name in os.listdir(src_dir_path):
                        src_sub_item_path = os.path.join(src_dir_path, sub_item_name)
                        dst_sub_item_path = os.path.join(dst_dir_path, sub_item_name)
                        if os.path.isfile(src_sub_item_path) and sub_item_name.endswith(".py"):
                            is_effective = True
                            os.makedirs(dst_dir_path, exist_ok=True)
                            shutil.copyfile(src_sub_item_path, dst_sub_item_path)
                        if os.path.isdir(src_sub_item_path) and _is_python_module(src_sub_item_path):
                            is_effective = True
                            ioutils.copytree_include(src_sub_item_path, dst_sub_item_path,
                                                    include_extensions=(".py",), exist_ok=True)
                    if is_effective:
                        self.copied_local_dependencies.append(
                            os.path.join(ModelSpecConstants.LOCAL_DEPENDENCIES_PATH, dst_dir_name))

            if self.copied_local_dependencies:
                zip_file_path = os.path.join(artifact_path, ModelSpecConstants.LOCAL_DEPENDENCIES_ZIP_FILE_NAME)
                ziputils.zip_dir(temp_dir_path, zip_file_path)

    def load(self, artifact_path, relative_paths):
        self.local_dependencies = [os.path.abspath(os.path.join(artifact_path, path)) for path in relative_paths]
        logger.info(f"local_dependencies = {self.local_dependencies}")
        if self.local_dependencies:
            zip_file_path = os.path.join(artifact_path, ModelSpecConstants.LOCAL_DEPENDENCIES_ZIP_FILE_NAME)
            if not os.path.isfile(zip_file_path):
                raise FileNotFoundError(f"Failed to load local_dependencies because {zip_file_path} is missing.")
            local_dependencies_path = os.path.join(artifact_path, ModelSpecConstants.LOCAL_DEPENDENCIES_PATH)
            ziputils.unzip_dir(zip_file_path, local_dependencies_path)
            logger.info(f"Unzipped {zip_file_path} to {local_dependencies_path}.")

    def install(self):
        for local_dependency_path in self.local_dependencies:
            sys.path.append(local_dependency_path)
            logger.info(f"Appended {local_dependency_path} to sys.path")
