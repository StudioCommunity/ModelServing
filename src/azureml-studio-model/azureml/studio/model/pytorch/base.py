import ast
import inspect

import pandas as pd
import torch

from ..logger import get_logger
from ..generic import GenericModel
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


class PytorchBase(GenericModel):
    
    def __init__(self, model, is_gpu=False, input_args=None):
        self.device = "cuda" if is_gpu and torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.input_args = input_args if input_args else get_input_args(model)
        self.inputs = [ModelInput(arg, "ndarray") for arg in self.input_args]
      
    def predict(self, df):
        logger.info(f"Device type = {self.device}, input_args = {self.input_args}")
        output = []

        with torch.no_grad():
            logger.info(f"input_df =\n {df}")
            # TODO: Consolidate serveral rows together for efficiency
            for _, row in df.iterrows():
                input_params = list(map(to_tensor, row[self.input_args]))
                predicted = self.model(*input_params)
                output.append(predicted.tolist())

        output_df = pd.DataFrame(output)
        output_df.columns = [f"Score_{i}" for i in range(0, output_df.shape[1])]
        logger.info(f"output_df =\n{output_df}")
        return output_df
