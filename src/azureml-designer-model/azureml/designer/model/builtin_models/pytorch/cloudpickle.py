import os

import cloudpickle

from .base import PytorchBaseModel
from ...utils import conda_merger, ioutils


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
        ioutils.validate_overwrite(save_to, overwrite_if_exists)
        with open(save_to, "wb") as fp:
            cloudpickle.dump(self.raw_model, fp)

    @classmethod
    def load_with_flavor(cls, load_from, flavor):
        with open(load_from, "rb") as fp:
            model = cloudpickle.load(fp)
        return cls(model, flavor)
