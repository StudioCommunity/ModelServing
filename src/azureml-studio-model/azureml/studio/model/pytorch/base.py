import ast
import inspect

import pandas as pd
import torch

from ..logger import get_logger
from ..generic_model import GenericModel
from ..model_input import ModelInput
from ..model_output import ModelOutput

logger = get_logger(__name__)

def to_tensor(self, entry):
    if isinstance(entry, str):
        entry = ast.literal_eval(entry)
    return torch.Tensor(list(entry)).to(self.device)

def get_input_args(pytorch_model):
    try:
        forward_func = getattr(pytorch_model, 'forward')
        args = inspect.getfullargspec(forward_func).args
        if 'self' in args:
            args.remove('self')
        return args
    except AttributeError:
        logger.warning("Model without 'forward' function cannot be used to predict", exc_info=True)


class PytorchBaseModel(GenericModel):

    raw_model = None
    
    def __init__(self, raw_model, is_cuda=False):
        self._is_cuda = is_cuda
        self._device = "cuda" if is_cuda and torch.cuda.is_available() else "cpu"
        self.raw_model = raw_model.to(self._device)
        if is_cuda and not torch.cuda.is_available():
            logger.warning("The model is saved on gpu but loaded on cpu because cuda is not available")

    def predict(self, df):
        outputs = []
        with torch.no_grad():
            logger.info(f"input_df =\n {df}")
            # TODO: Consolidate serveral rows together for efficiency
            for _, row in df.iterrows():
                input_params = list(map(to_tensor, row[self.input_args]))
                predicted = self.raw_model(*input_params)
                outputs.append(predicted.tolist())

        output_df = pd.DataFrame(outputs)
        output_df.columns = [f"Score_{i}" for i in range(0, output_df.shape[1])]
        logger.info(f"output_df =\n{output_df}")
        return output_df
    
    def init_input_args(self):
        if self.inputs:
            self.input_args = [model_input.name for model_input in inputs]
        else:
            self.input_args = get_input_args(self.raw_model)
        