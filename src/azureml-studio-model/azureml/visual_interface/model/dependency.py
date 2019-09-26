import sys
import logging

import yaml
from pip._internal import main as pipmain

logger = logging.getLogger(__name__)

# Temporary workaround to reconstruct the python environment in training phase.
# Should deprecate when Module team support reading the conda.yaml in Model Folder and build image according to that
class DependencyManager(object):
    
    def __init__(self):
        self.conda_channels = []
        self.conda_dependencies = []
        self.pip_dependencies = []
    
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
        import conda.cli.python_api as Conda

        if len(self.conda_dependencies) == 0:
            logger.info("No conda denpendencies to install")
        else:
            conda_cmds = [Conda.Commands.INSTALL]
            for channel in self.conda_channels:
                conda_cmds += ["-c", channel]
            conda_cmds += self.conda_dependencies
            (stdout_str, sterr_str, return_code_int) = Conda.run_command(
                *conda_cmds, use_exception_handler=True, stdout=sys.stdout, stderr=sys.stderr)
            logger.info("Finished install conda dependencies")
            logger.warn(f"sterr: {sterr_str}")

        if not self.pip_dependencies:
            logger.info("No pip dependencies to install")
        else:
            pipmain(["install"] + self.pip_dependencies)
            logger.info(f"Finished pip install {' '.join(self.pip_dependencies)}")