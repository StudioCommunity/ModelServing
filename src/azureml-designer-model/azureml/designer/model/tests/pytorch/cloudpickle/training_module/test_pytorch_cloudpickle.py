import os
from os.path import dirname, abspath

import numpy as np
import pandas as pd
import pyarrow.parquet
import torch
from torch.autograd import Variable

from azureml.designer.model.io import save_pytorch_model, load_generic_model

from .model import LinearRegression

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_training_data():
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


def train(model, x_train, y_train):
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


def test_save_load():
    model = LinearRegression(1, 1).to(device)
    x_train, y_train = get_training_data()
    train(model, x_train, y_train)

    model_save_path = os.path.join(dirname(dirname(abspath(__file__))), "AzureMLModel")
    local_dependencies = [dirname(dirname(abspath(__file__)))]

    save_pytorch_model(model, path=model_save_path, local_dependencies=local_dependencies)
    loaded_generic_model = load_generic_model(model_save_path)
    df = pd.DataFrame({"x": [[10.0], [11.0], [12.0]]})
    predict_result = loaded_generic_model.predict(df)
    assert predict_result.shape[0] == df.shape[0]

    loaded_pytorch_model = loaded_generic_model.raw_model
    assert isinstance(loaded_pytorch_model, torch.nn.Module)

    data_save_path = os.path.join(dirname(dirname(abspath(__file__))), "data.dataset.parquet")
    df.to_parquet(data_save_path, engine="pyarrow")
