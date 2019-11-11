import cloudpickle
import pandas as pd
import torch
import torchvision

from .base import PytorchBaseModel

class PytorchCloudPickleModel(PytorchBaseModel):

    flavor = {
        "name": "pytorch",
        "serializtion_method": "cloudpickle",
        "is_cuda": False
    }

    def save(self, save_to, overwrite_if_exists=True):
        with open(save_to, "wb") as fp:
            cloudpickle.dump(self.raw_model, fp)
    
    @classmethod
    def load(cls, load_from):
        with open(load_from, "rb") as fp:
            model = cloudpickle.load(fp)
        return cls(model)