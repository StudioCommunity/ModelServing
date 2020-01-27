import ast
import inspect

import pandas as pd
import torch
import torchvision
from PIL import Image

from ..builtin_model import BuiltinModel
from ...constants import ModelSpecConstants
from ...logger import get_logger
from ...utils import conda_merger
from ...model_spec.task_type import TaskType


logger = get_logger(__name__)


class PytorchBaseModel(BuiltinModel):

    raw_model: torch.nn.Module = None
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
        is_multi_gpu = flavor.get(ModelSpecConstants.IS_MULTI_GPU_KEY, False)
        self.flavor[ModelSpecConstants.IS_CUDA_KEY] = is_cuda
        self.flavor[ModelSpecConstants.IS_MULTI_GPU_KEY] = is_multi_gpu
        self._device = "cuda" if is_cuda and torch.cuda.is_available() else "cpu"
        self.raw_model.to(self._device)
        self.raw_model.eval()

    def predict(self, inputs: list) -> list:
        """
        Get prediction
        :param inputs: list-like of list-like data structure
        :return: list of list
        """
        logger.info(f"len(inputs) = {len(inputs)}")
        with torch.no_grad():
            model_inputs = self._pre_process(inputs)
            logger.info(f"len(model_inputs) = {len(model_inputs)}")
            for i, model_input in enumerate(model_inputs):
                logger.info(f"model_inputs[{i}].shape = {model_input.shape}")
            model_output = self.raw_model(*model_inputs)
            pred_ret = self._post_process(model_output)
            return pred_ret

    def _pre_process(self, input_tuple_list) -> tuple:
        """
        Convert all elements to tensor, and stack tensors in each column into a 1-dim higher tensor
        e.g. input_tuple_list = [(PIL.Image(3 * 224 * 224), control_tensor0(1 * 5)),
                                 (PIL.Image(3 * 224 * 224), control_tensor1(1 * 5))]
             output would be (images_tensor(2 * 3 * 224 * 224), control_tensor(2 * 1 * 5))
        :param input_tuple_list: A list of tuple with same length, containing PIL.Image or ndarray
        :return:
        """
        input_tensors_list = []
        for input_tuple in input_tuple_list:
            input_tensors = [self._to_tensor(x) for x in input_tuple]
            input_tensors_list.append(input_tensors)

        output_tensors_cnt = len(input_tensors_list[0])
        output_tensors = [None] * output_tensors_cnt
        for j in range(output_tensors_cnt):
            output_tensors[j] = torch.cat([row[j].unsqueeze(0) for row in input_tensors_list])
        return tuple(output_tensors)

    def _post_process(self, model_output: torch.Tensor) -> list:
        """
        Transform raw_model output to task-specified output format
        :param model_output:
        :return:
        """
        if self.task_type == TaskType.MultiClassification:
            softmax = torch.nn.Softmax(dim=1)
            pred_probs = softmax(model_output).cpu().numpy().tolist()
            pred_index = torch.argmax(model_output, 1).cpu().numpy().tolist()
            pred_result = list(zip(pred_index, pred_probs))
            logger.info(f"pred_result = {pred_result}")
            return pred_result
        elif not self.task_type or self.task_type == TaskType.Regression:
            return model_output.squeeze(0).tolist()
        else:
            raise Exception(f"Task_type: {self.task_type.name} has not been implemented yet.")

    def _to_tensor(self, entry):
        if isinstance(entry, Image.Image):
            # transform = torchvision.transforms.ToTensor()
            # Workaround for densenet Demo
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
            )
            logger.info("Applied normalization as workaround.")
            return transform(entry).to(self._device)
        if isinstance(entry, str):
            entry = ast.literal_eval(entry)
        return torch.Tensor(entry).to(self._device)

    def get_default_feature_columns(self):
        if not self.raw_model:
            logger.warning("Can't get default_feature_columns with raw_model uninitialized")
        try:
            if isinstance(self.raw_model, torch.nn.DataParallel):
                forward_func = getattr(self.raw_model.module, 'forward')
            else:
                forward_func = getattr(self.raw_model, 'forward')
            args = inspect.getfullargspec(forward_func).args
            if 'self' in args:
                args.remove('self')
            return args
        except AttributeError:
            logger.warning("Model without 'forward' function cannot be used to predict", exc_info=True)
            return None
