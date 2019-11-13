import os

import cloudpickle

from .base import PytorchBaseModel


class PytorchCloudPickleModel(PytorchBaseModel):

    flavor = {
        "name": "pytorch",
        "serialization_method": "cloudpickle",
        "is_cuda": False
    }

    def save(self, save_to, overwrite_if_exists=True):
        if os.path.isfile(save_to) and not overwrite_if_exists:
            raise Exception(f"File {save_to} exists. Set overwrite_is_exists=True if you want to overwrite it.")
        with open(save_to, "wb") as fp:
            cloudpickle.dump(self.raw_model, fp)
    
    @classmethod
    def load(cls, load_from):
        with open(load_from, "rb") as fp:
            model = cloudpickle.load(fp)
        return cls(model)
