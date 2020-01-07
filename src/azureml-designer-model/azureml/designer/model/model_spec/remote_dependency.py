import os
import sys
import subprocess

from ..constants import ModelSpecConstants
from ..logger import get_logger
from ..utils import yamlutils

logger = get_logger(__name__)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=sys.version_info.major,
                                                  minor=sys.version_info.minor,
                                                  micro=sys.version_info.micro)


def _run_install_cmds(cmds, command_name):
    logger.info(" ".join(cmds))
    result = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    log_message = f"Finished to install {command_name} dependencies with return_code {result.returncode}"
    if result.returncode == 0:
        logger.info(log_message)
    else:
        logger.warning(log_message)
    logger.info(f"stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr}")


# Temporary workaround to reconstruct the python environment in training phase.
# Should deprecate when Module team support reading the conda.yaml in Model Folder and build image according to that
class RemoteDependencyManager(object):
    
    def __init__(
        self,
        additional_conda_channels=[],
        additional_conda_deps=[],
        additional_pip_deps=[]
    ):
        self.conda_channels = ["defaults"] + additional_conda_channels
        self.conda_dependencies = [f"python={PYTHON_VERSION}"] + additional_conda_deps
        self.pip_dependencies = additional_pip_deps

    def save(self, artifact_path, overwrite_if_exists=True):
        conda_env = {
            "name": ModelSpecConstants.CONDA_ENV_NAME,
            "channels": self.conda_channels,
            "dependencies": self.conda_dependencies + [{"pip": self.pip_dependencies}]
        }
        conda_file_path = os.path.join(artifact_path, ModelSpecConstants.CONDA_FILE_NAME)
        if os.path.isfile(conda_file_path) and not overwrite_if_exists:
            raise Exception(f"File {conda_file_path} exists. Set overwrite_is_exists=True if you want to overwrite it.")
        yamlutils.dump_to_yaml_file(conda_env, conda_file_path)
        logger.info(f"Saved conda to {conda_file_path}")
    
    def load(self, conda_yaml_path):
        logger.info(f"Trying to load conda dependency from {conda_yaml_path}")
        config = yamlutils.load_yaml_file(conda_yaml_path)

        if isinstance(config["channels"], list):
            self.conda_channels = config["channels"]
        else:
            self.conda_channels = [config["channels"]]

        for entry in config["dependencies"]:
            if isinstance(entry, dict) and "pip" in entry:
                self.pip_dependencies = entry["pip"]
            # TODO: Use regex for precision
            elif entry.startswith(("python=", "python>", "python<")):
                pass
            else:
                self.conda_dependencies.append(entry)
        logger.info(f"conda_channels = {', '.join(self.conda_channels)}")
        logger.info(f"conda_dependencies = {', '.join(self.conda_dependencies)}")
        logger.info(f"pip_dependencies = {', '.join(self.pip_dependencies)}")

    def install(self):
        if not self.conda_dependencies:
            logger.info("No conda dependencies to install")
        else:
            conda_cmds = ["conda", "install", "-y"]
            for channel in self.conda_channels:
                conda_cmds += ["-c", channel]
            conda_cmds += self.conda_dependencies
            _run_install_cmds(conda_cmds, "conda")

        if not self.pip_dependencies:
            logger.info("No pip dependencies to install")
        else:
            pip_cmds = ["pip", "install"] + self.pip_dependencies
            _run_install_cmds(pip_cmds, "pip")

