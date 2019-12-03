import ast
import inspect

import pandas as pd
import torch
import torchvision

from ..builtin_model import BuiltinModel
from ...constants import ModelSpecConstants
from ...logger import get_logger
from ...utils import conda_merger
from ...model_spec.task_type import TaskType


logger = get_logger(__name__)


class PytorchBaseModel(BuiltinModel):

    raw_model = None
    _device = "cpu"
    feature_columns_names = None
    extra_conda = {
        "channels": ["pytorch"],
        "dependencies": [
            f"pytorch={torch.__version__}",
            f"torchvision={torchvision.__version__}"
        ]
    }
    default_conda = conda_merger.merge_envs([BuiltinModel.default_conda, extra_conda])

    def __init__(self, raw_model, flavor: dict = {}):
        self.raw_model = raw_model
        is_cuda = flavor.get(ModelSpecConstants.IS_CUDA_KEY, False)
        self.flavor[ModelSpecConstants.IS_CUDA_KEY] = is_cuda
        self._device = "cuda" if is_cuda and torch.cuda.is_available() else "cpu"
        self.raw_model.to(self._device)
        self.raw_model.eval()

    def predict(self, inputs: list):
        outputs = []
        with torch.no_grad():
            logger.info(f"len(inputs) = {len(inputs)}")
            # TODO: Stack several rows together for efficiency
            for row in inputs:
                input_params = list(map(self.to_tensor, row))
                logger.info(f"len(input_params) = {len(input_params)}")
                for i, input_param in enumerate(input_params):
                    logger.info(f"input_params[{i}].shape = {input_param.shape}")
                model_output = self.raw_model(*input_params)
                if self.task_type == TaskType.MultiClassification:
                    softmax = torch.nn.Softmax(dim=1)
                    pred_probs = softmax(model_output).cpu().numpy()[0]
                    pred_index = torch.argmax(model_output, 1)[0].cpu().item()
                    pred_result = pred_index, list(pred_probs)
                    outputs.append(pred_result)
                elif self.task_type == TaskType.Regression:
                    outputs.append(model_output.squeeze(0).tolist())
                else:
                    raise Exception(f"Task_type: {self.task_type.name} not implemented yet.")

        return outputs

    # Convert all elements to tensor, and stack rows of tensors into a 1-dim higher tensor
    def _pre_process(self, input_tuple_list):
        pass

    # Transform raw_model output to task-specified output format
    def _post_process(self):
        pass

    def to_tensor(self, entry):
        if isinstance(entry, str):
            entry = ast.literal_eval(entry)
        return torch.Tensor(list(entry)).to(self._device)
    
    def get_default_feature_columns(self):
        if not self.raw_model:
            logger.warning("Can't get defualt_feature_columns with raw_model uninitialized")
        try:
            forward_func = getattr(self.raw_model, 'forward')
            args = inspect.getfullargspec(forward_func).args
            if 'self' in args:
                args.remove('self')
            return args
        except AttributeError:
            logger.warning("Model without 'forward' function cannot be used to predict", exc_info=True)
            return None
