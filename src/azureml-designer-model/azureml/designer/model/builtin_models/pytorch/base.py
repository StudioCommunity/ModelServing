import ast
import inspect

import pandas as pd
import torch
import torchvision

from ...logger import get_logger
from ..builtin_model import BuiltinModel
from ...utils import conda_merger

logger = get_logger(__name__)


def get_input_args(pytorch_model):
    try:
        forward_func = getattr(pytorch_model, 'forward')
        args = inspect.getfullargspec(forward_func).args
        if 'self' in args:
            args.remove('self')
        return args
    except AttributeError:
        logger.warning("Model without 'forward' function cannot be used to predict", exc_info=True)


class PytorchBaseModel(BuiltinModel):

    raw_model = None
    _device = "cpu"
    input_args = None
    extra_conda = {
        "channels": ["pytorch"],
        "dependencies": [
            f"pytorch={torch.__version__}",
            f"torchvision={torchvision.__version__}"
        ]
    }
    default_conda = conda_merger.merge_envs([BuiltinModel.default_conda, extra_conda])

    def __init__(self, raw_model, is_cuda=False):
        self.raw_model = raw_model
        self.flavor["is_cuda"] = is_cuda

    def config(self, model_spec: dict):
        is_cuda = model_spec["flavor"].get("is_cuda", False)
        self.flavor["is_cuda"] = is_cuda
        self._device = "cuda" if is_cuda and torch.cuda.is_available() else "cpu"
        self.raw_model.to(self._device)
        if is_cuda and not torch.cuda.is_available():
            logger.warning("The model is saved on gpu but loaded on cpu because cuda is not available")

        if model_spec.get("inputs", None):
            self.input_args = [model_input["name"] for model_input in model_spec["inputs"]]
        else:
            self.input_args = get_input_args(self.raw_model)

    def predict(self, df):
        outputs = []
        with torch.no_grad():
            logger.info(f"input_df =\n {df}")
            # TODO: Consolidate several rows together for efficiency
            for _, row in df.iterrows():
                input_params = list(map(self.to_tensor, row[self.input_args]))
                predicted = self.raw_model(*input_params)
                outputs.append(predicted.tolist())

        output_df = pd.DataFrame(outputs)
        output_df.columns = [f"Score_{i}" for i in range(0, output_df.shape[1])]
        logger.info(f"output_df =\n{output_df}")
        return output_df

    def to_tensor(self, entry):
        if isinstance(entry, str):
            entry = ast.literal_eval(entry)
        return torch.Tensor(list(entry)).to(self._device)
