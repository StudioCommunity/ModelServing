import pytest
from os.path import dirname, abspath

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
# TODO: Substitude visual_interface when Module solves conflict issue in azureml.studio.__init__.py
import azureml.visual_interface.model.pytorch
import azureml.visual_interface.model.generic

from .model import LinearRegression

import sys
print(f"__file__ = {__file__}")
print(f"sys.path = {sys.path}")


def test_save_load():
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LinearRegression(1, 1).to(device)
    # Train
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(3):
        inputs = Variable(torch.from_numpy(x_train)).to(device)
        labels = Variable(torch.from_numpy(y_train)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # code_path = dirname(dirname(abspath(__file__)))
    # azureml.visual_interface.model.pytorch.save(model, code_path=code_path)
    azureml.visual_interface.model.pytorch.save(model)

    loaded_pytorch_model = azureml.visual_interface.model.pytorch.load()

    loaded_generic_model = azureml.visual_interface.model.generic.load()
    df = pd.DataFrame({"x": [[10], [11], [12]]})
    predict_result = loaded_generic_model.predict(df)
    assert predict_result.shape[0] == df.shape[0]