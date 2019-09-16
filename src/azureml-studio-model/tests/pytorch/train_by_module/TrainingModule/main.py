from os.path import dirname, abspath

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import azureml.studio.model.pytorch
import azureml.studio.model.generic

from .model import LinearRegression


x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
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

    print('epoch {}, loss {}'.format(epoch, loss.item()))

    # ./AzureMLModel
    # azureml.studio.model.pytorch.save(model, code_path=".")
    print(f"__file__ = {__file__}")
    code_path = dirname(dirname(abspath(__file__)))
    print(f"code_path = {code_path}")
    azureml.studio.model.pytorch.save(model, code_path=code_path)

    loaded_pytorch_model = azureml.studio.model.pytorch.load()
    print(f"type(loaded_pytorch_model) = {type(loaded_pytorch_model)}")

    loaded_generic_model = azureml.studio.model.generic.load()
    df = pd.DataFrame({"x": [[10], [11], [12]]})
    predict_result = loaded_generic_model.predict(df)
    print(f"predict_result =\n{predict_result}")