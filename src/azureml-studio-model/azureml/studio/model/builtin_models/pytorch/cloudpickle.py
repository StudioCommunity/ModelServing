import os

import cloudpickle

from .base import PytorchBaseModel
from ...utils import conda_merger


class PytorchCloudPickleModel(PytorchBaseModel):

    serialization_method = "cloudpickle"
    extra_conda = {
        "dependencies": [
            {
                "pip": [
                    f"cloudpickle=={cloudpickle.__version__}"
                ]
            }
        ]
    }
    default_conda = conda_merger.merge_envs([PytorchBaseModel.default_conda, extra_conda])

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
