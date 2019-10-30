import os
import sys
import shutil

import yaml
from subprocess import Popen, PIPE, STDOUT

from . import constants
from . import utils
from .logger import get_logger 

logger = get_logger(__name__)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=sys.version_info.major,
                                                  minor=sys.version_info.minor,
                                                  micro=sys.version_info.micro)

def _run_install_cmds(cmds, command_name):
    logger.info(" ".join(cmds))
    p = Popen(cmds, stdout=PIPE, stderr=PIPE)
    p.wait()
    stdout = p.stdout.read().decode("utf-8")
    stderr = p.stderr.read().decode("utf-8")
    logger.info(f"stdout: {stdout}")
    if stderr:
        logger.warning(f"sterr: {stderr}")
    logger.info(f"Finished to install {command_name} dependencies")

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

    def save(self, artifact_path, exist_ok=True):
        conda_env = {
            "name": constants.CONDA_ENV_NAME,
            "channels": self.conda_channels,
            "dependencies": self.conda_dependencies + [{"pip": self.pip_dependencies}]
        }
        conda_file_path = os.path.join(artifact_path, constants.CONDA_FILE_NAME)
        with open(conda_file_path, "w") as f:
            yaml.safe_dump(conda_env, stream=f, default_flow_style=False) 
        logger.info(f"Saved conda to {conda_file_path}")
    
    def load(self, conda_yaml_path):
        logger.info(f"Trying to load conda dependency from {conda_yaml_path}")
        with open(conda_yaml_path) as fp:
            config = yaml.safe_load(fp)

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
            logger.info("No conda denpendencies to install")
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
   