import sys
import logging

import yaml
from pip._internal import main as pipmain
from subprocess import Popen, PIPE, STDOUT

logger = logging.getLogger(__name__)


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
class DependencyManager(object):
    
    def __init__(self):
        self.conda_channels = []
        self.conda_dependencies = []
        self.pip_dependencies = []
        self.local_dependency_path = None
    
    def load(self, conda_yaml_path, local_dependency_path=None):
        logger.info(f"local_dependency_path = {local_dependency_path}")
        self.local_dependency_path = local_dependency_path
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
        if self.local_dependency_path:
            sys.path.append(self.local_dependency_path)
            logger.info(f"Appended {self.local_dependency_path} to sys.path")

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
   
