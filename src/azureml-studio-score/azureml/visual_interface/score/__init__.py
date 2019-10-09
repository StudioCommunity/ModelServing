# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from os.path import dirname, abspath

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

PROJECT_ROOT_PATH = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
logger.info(f"PROJECT_ROOT_PATH = {PROJECT_ROOT_PATH}")