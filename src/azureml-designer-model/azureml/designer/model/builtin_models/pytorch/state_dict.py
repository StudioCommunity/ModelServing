import os
import importlib

import torch

from .base import PytorchBaseModel
from ...logger import get_logger
from ...utils import conda_merger, ioutils, dictutils

logger = get_logger(__name__)


class PytorchStateDictModel(PytorchBaseModel):

    serialization_method = "state_dict"
    extra_conda = {}
    default_conda = conda_merger.merge_envs([PytorchBaseModel.default_conda, extra_conda])

    def __init__(self, raw_model, flavor):
        if not flavor.get("model_module", None):
            self.flavor["model_module"] = raw_model.__class__.__module__
        if not flavor.get("model_class", None):
            self.flavor["model_class"] = raw_model.__class__.__name__
        self.flavor["init_params"] = flavor.get("init_params", {})
        super().__init__(raw_model, flavor)

    def save(self, save_to, overwrite_if_exists=True):
        ioutils.validate_overwrite(save_to, overwrite_if_exists)
        state_dict = self.raw_model.module.state_dict() if torch.cuda.device_count() > 1 \
            else self.raw_model.state_dict()
        torch.save(state_dict, save_to)

    @classmethod
    def load_with_flavor(cls, load_from, flavor):
        model_module = flavor.get("model_module", None)
        model_class = torch.nn.Module
        if not model_module:
            logger.warning("No model_module specified, using nn.Module as default.")
        else:
            model_class_name = flavor.get("model_class", None)
            if not model_class_name:
                logger.warning("No model_class specified, using nn.Module as default.")
            else:
                try:
                    model_module = importlib.import_module(model_module)
                    model_class = getattr(model_module, model_class_name)
                except Exception as e:
                    logger.error(f"Failed to load {model_class} from {model_module}")
                    raise

        init_params = flavor.get("init_params", {})
        model = model_class(**init_params)
        model.load_state_dict(torch.load(load_from))

        return cls(model, flavor)
