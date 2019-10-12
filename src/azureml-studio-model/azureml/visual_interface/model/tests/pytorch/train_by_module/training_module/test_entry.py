import sys
import os

import pytest
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.autograd import Variable
# TODO: Substitude visual_interface when Module solves conflict issue in azureml.studio.__init__.py
from azureml.visual_interface.model import PROJECT_ROOT_PATH
import azureml.visual_interface.model.pytorch
import azureml.visual_interface.model.generic

from .model import LinearRegression


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
    for _ in range(3):
        inputs = Variable(torch.from_numpy(x_train)).to(device)
        labels = Variable(torch.from_numpy(y_train)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    score_test_path = os.path.join(PROJECT_ROOT_PATH, "azureml-studio-score/azureml/visual_interface/score/score/tests/pytorch")
    model_save_path = os.path.join(score_test_path, "InputPort1")
    dataset_save_path = os.path.join(score_test_path, "InputPort2", "data.dataset.parquet")

    azureml.visual_interface.model.pytorch.save(model, path=model_save_path, exist_ok=True)

    loaded_pytorch_model = azureml.visual_interface.model.pytorch.load(model_save_path)
    assert isinstance(loaded_pytorch_model, torch.nn.Module)

    loaded_generic_model = azureml.visual_interface.model.generic.load(model_save_path)
    df = pd.DataFrame({"x": [[10.0], [11.0], [12.0]]})
    if os.path.exists(dataset_save_path):
        os.remove(dataset_save_path)
    df.to_parquet(dataset_save_path, engine="pyarrow")
    predict_result = loaded_generic_model.predict(df)
    assert predict_result.shape[0] == df.shape[0]