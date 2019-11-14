import os
import sys
import shutil

from .. import constants
from ..utils import ioutils
from ..logger import get_logger

logger = get_logger(__name__)


class LocalDependencyManager(object):
    
    def __init__(self, local_dependencies=[]):
        self.local_dependencies = local_dependencies
        self.copied_local_dependencies = []

    def save(self, artifact_path, exist_ok=True) -> list:
        src_abs_paths = [os.path.abspath(_) for _ in self.local_dependencies]
        dst_dir = os.path.join(artifact_path, constants.LOCAL_DEPENDENCIES_PATH)

        # Copy pyfiles
        src_py_files = list(filter(lambda x: x.endswith(".py"), src_abs_paths))
        if src_py_files:
            py_filenames = [os.path.split(file_path)[-1] for file_path in src_py_files]
            if len(set(py_filenames)) < len(py_filenames):
                raise Exception("There are duplication in dependency py file name, which is not allowed.")
            pyfiles_basepath = os.path.join(dst_dir, "pyfiles")
            os.makedirs(pyfiles_basepath, exist_ok=exist_ok)
            for filename, src_path in zip(py_filenames, src_py_files):
                shutil.copyfile(src_path, os.path.join(pyfiles_basepath, filename))
            self.copied_local_dependencies.append(os.path.join(constants.LOCAL_DEPENDENCIES_PATH, "pyfiles"))

        # Copy directories
        src_directories = list(filter(lambda x: not x.endswith(".py"), src_abs_paths))
        if src_directories:
            dirname_cnt_dict = {}
            for directory in src_directories:
                if not os.path.isdir(directory):
                    raise Exception(f"Only py files and directories are supported, got {directory}")

            src_directories.sort()
            i = 0
            while i < len(src_directories):
                src_ancestor = src_directories[i]
                dst_ancester_name = os.path.split(src_ancestor)[-1]
                dirname_cnt_dict[dst_ancester_name] = dirname_cnt_dict.get(dst_ancester_name, 0) + 1
                if dirname_cnt_dict[dst_ancester_name] > 1:
                    dst_ancester_name = f"{dst_ancester_name}_{dirname_cnt_dict[dst_ancester_name] - 1}"
                dst_ancestor_path = os.path.join(dst_dir, dst_ancester_name)
                ioutils._copytree_include(src_ancestor, dst_ancestor_path, include_extensions=(".py"), exist_ok=exist_ok)
                self.copied_local_dependencies.append(os.path.join(constants.LOCAL_DEPENDENCIES_PATH, dst_ancester_name))
                j = i + 1
                while j < len(src_directories):
                    if src_directories[j].startswith(src_ancestor):
                        self.copied_local_dependencies.append(os.path.join(dst_ancester_name, os.path.relpath(src_ancestor, src_directories[j])))
                    else:
                        break
                i = j

    def load(self, artifact_path, relative_paths):
        self.local_dependencies = [os.path.abspath(os.path.join(artifact_path, path)) for path in relative_paths]
        logger.info(f"Loaded {self.local_dependencies}")

    def install(self):
        for local_dependency_path in self.local_dependencies:
            sys.path.append(local_dependency_path)
            logger.info(f"Appended {local_dependency_path} to sys.path")