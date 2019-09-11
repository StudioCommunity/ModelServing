import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

import azureml.studio.model.pytorch
import azureml.studio.model.generic

from model import LinearRegression

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
    for epoch in range(10):
        inputs = Variable(torch.from_numpy(x_train)).to(device)
        labels = Variable(torch.from_numpy(y_train)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

    
    azureml.studio.model.pytorch.save(model, code_path=".")

    loaded_pytorch_model = azureml.studio.model.pytorch.load()

    loaded_generic_model = azureml.studio.model.generic.load()
    print(dir(loaded_generic_model))
    df = pd.DataFrame({"x": [[10], [11], [12]]})
    print(loaded_generic_model.predict(df))
