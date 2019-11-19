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

    @ioutils.validate_path_existence
    def save(self, save_to, overwrite_if_exists=True):
        if os.path.isfile(save_to) and not overwrite_if_exists:
            raise Exception(f"File {save_to} exists. Set overwrite_is_exists=True if you want to overwrite it.")
        state_dict = self.raw_model.module.state_dict() if torch.cuda.device_count() > 1 \
            else self.raw_model.state_dict()
        torch.save(state_dict, save_to)

    @classmethod
    def load_with_modelspec(cls, load_from, model_spec):
        model_module_path = dictutils.get_value_by_key_path(model_spec, "flavor/model_module")
        model_class = torch.nn.Module
        if not model_module_path:
            logger.warning("No model_module specified, using nn.Module as default.")
        else:
            model_class_name = dictutils.get_value_by_key_path(model_spec, "flavor/model_class")
            if not model_class_name:
                logger.warning("No model_class specified, using nn.Module as default.")
            else:
                try:
                    model_module = importlib.import_module(model_module_path)
                    model_class = getattr(model_module, model_class_name)
                except Exception as e:
                    logger.error(f"Failed to load {model_class} from {model_module}")
                    raise

        init_params = dictutils.get_value_by_key_path(model_spec, "flavor/init_params", default_value={})
        model = model_class(**init_params)
        model.load_state_dict(torch.load(load_from))

        return cls(model, model_spec)
