import sys
import os
# Assume ../.. this is the module's root path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
print(f"Added {grandparentdir} to sys.path")
print(f"sys.path = {sys.path}")

import azureml.studio.model.pytorch
import azureml.studio.model.generic

from model import Model


if __name__ == "__main__":
    m = Model()
    azureml.studio.model.pytorch.save(m, code_path=".")

    loaded_pytorch_model = azureml.studio.model.pytorch.load()

    loaded_generic_model = azureml.studio.model.generic.load()
    print(dir(loaded_generic_model))
