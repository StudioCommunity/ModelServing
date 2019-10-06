import os
import sys
from os.path import dirname, abspath

import pytest
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

# TODO: Substitude visual_interface when Module solves conflict issue in azureml.studio.__init__.py
import azureml.visual_interface.model.pytorch
import azureml.visual_interface.model.generic

from .model import LinearRegression
from ... import module_invoker

def test_save_score():
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
    for _ in range(3):
        inputs = Variable(torch.from_numpy(x_train)).to(device)
        labels = Variable(torch.from_numpy(y_train)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    base_path = dirname(abspath(__file__))
    trained_model_path = os.path.join(base_path, "InputPort1")
    dataset_path = os.path.join(base_path, "InputPort2")
    os.makedirs(dataset_path, exist_ok=True)
    scored_dataset_path = os.path.join(base_path, "OutputPort")
    os.makedirs(scored_dataset_path, exist_ok=True)

    azureml.visual_interface.model.pytorch.save(
        pytorch_model=model,
        path=trained_model_path)
    df = pd.DataFrame({"x": [[10.0], [11.0], [12.0]]})
    df.to_parquet(os.path.join(dataset_path, "data.dataset.parquet"), engine="pyarrow")

    module_invoker.entrance(trained_model_path, dataset_path, scored_dataset_path, True)