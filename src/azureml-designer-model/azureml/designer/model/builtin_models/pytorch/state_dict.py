import os
import importlib

import torch

from .base import PytorchBaseModel
from ...constants import ModelSpecConstants
from ...logger import get_logger
from ...utils import conda_merger, ioutils, dictutils

logger = get_logger(__name__)


class PytorchStateDictModel(PytorchBaseModel):

    serialization_method = "state_dict"
    extra_conda = {}
    default_conda = conda_merger.merge_envs([PytorchBaseModel.default_conda, extra_conda])

    def __init__(self, raw_model, flavor):
        self.flavor[ModelSpecConstants.MODEL_MODULE_KEY] = flavor.get(
            ModelSpecConstants.MODEL_MODULE_KEY,
            raw_model.__class__.__module__)
        self.flavor[ModelSpecConstants.MODEL_CLASS_KEY] = flavor.get(
            ModelSpecConstants.MODEL_CLASS_KEY,
            raw_model.__class__.__name__)
        self.flavor[ModelSpecConstants.INIT_PARAMS_KEY] = flavor.get(
            ModelSpecConstants.INIT_PARAMS_KEY,
            {})
        super().__init__(raw_model, flavor)

    def save(self, save_to, overwrite_if_exists=True):
        ioutils.validate_overwrite(save_to, overwrite_if_exists)
        state_dict = self.raw_model.state_dict()
        torch.save(state_dict, save_to)

    @classmethod
    def load_with_flavor(cls, load_from, flavor):
        model_module = flavor.get(ModelSpecConstants.MODEL_MODULE_KEY, None)
        model_class = torch.nn.Module
        if not model_module:
            logger.warning("No model_module specified, using nn.Module as default.")
        else:
            model_class_name = flavor.get(ModelSpecConstants.MODEL_CLASS_KEY, None)
            if not model_class_name:
                logger.warning("No model_class specified, using nn.Module as default.")
            else:
                try:
                    logger.info(f"Trying to import {model_module}")
                    model_module = importlib.import_module(model_module)
                    model_class = getattr(model_module, model_class_name)
                except Exception as e:
                    logger.error(f"Failed to load {model_class} from {model_module}.", exc_info=True)
                    raise

        init_params = flavor.get(ModelSpecConstants.INIT_PARAMS_KEY, {})
        logger.info(f"Trying to initialize model by calling {model_class}({init_params})")
        model = model_class(**init_params)
        model.load_state_dict(torch.load(load_from))

        return cls(model, flavor)
