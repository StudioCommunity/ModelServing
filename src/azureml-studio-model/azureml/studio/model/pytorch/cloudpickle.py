import cloudpickle
import pandas as pd
import torch

from .base import PytorchBase

class PytorchCloudPickle(PytorchBase):
    
    def save(self, save_to, overwrite_if_exists=True):
        with open(save_to, "wb") as fp:
            cloudpickle.dump(self.model, fp)
    
    @classmethod
    def load(cls, load_from, configs={}):
        with open(load_from, "rb") as fp:
            model = cloudpickle.load(fp)
        return cls(model)