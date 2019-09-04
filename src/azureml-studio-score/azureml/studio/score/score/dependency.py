import logging
import yaml
from pip._internal import main as pipmain

logger = logging.getLogger(__name__)

class DependencyManager(object):
    
    def __init__(self):
        self.pip_dependencies = None
    
    def load(self, conda_yaml_path):
        logger.info(f"Trying to load conda dependency from {conda_yaml_path}")
        with open(conda_yaml_path) as fp:
            config = yaml.safe_load(fp)

        for entry in config["dependencies"]:
            if isinstance(entry, dict) and "pip" in entry:
                self.pip_dependencies = entry["pip"]
                logger.info(f"pip_dependencies = {', '.join(self.pip_dependencies)}")
        if self.pip_dependencies is None:
            logger.info("No pip dependencies in yaml file")
            return

    def install(self):
        if self.pip_dependencies is None:
            logger.info("No dependencies to install")
            return
        pipmain(["install"] + self.pip_dependencies)
        logger.info(f"Finished pip install {' '.join(self.pip_dependencies)}")