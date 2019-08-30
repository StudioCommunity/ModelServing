import os
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import cloudpickle
import pickle
import yaml
import json

from builtin_models.pytorch import save_model

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MnistNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(MnistNet,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self,x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out


@click.command()
@click.option('--action', default="train", 
        type=click.Choice(['predict', 'train']))
@click.option('--model_path', default="./model/")
def run_pipeline(action, model_path):
  input_size = 784 # img_size = (28,28) ---> 28*28=784 in total
  hidden_size = 500 # number of nodes at hidden layer
  num_classes = 10 # number of output classes discrete range [0,9]
  num_epochs = 5 # number of times which the entire dataset is passed throughout the model
  batch_size = 64 # the size of input data took for one iteration
  lr = 1e-3 # size of step

  train_data = dsets.MNIST(root = './data', train = True,
                          transform = transforms.ToTensor(), download = True)

  test_data = dsets.MNIST(root = './data', train = False,
                        transform = transforms.ToTensor())

  train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                              batch_size = batch_size,
                                              shuffle = True)

  test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                        batch_size = batch_size, 
                                        shuffle = False)

  net = MnistNet(input_size, hidden_size, num_classes)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'DEVICE={device}')
  print(f'os.environ={os.environ}')
  net = net.to(device)

  loss_function = nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam( net.parameters(), lr=lr)

  for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_gen):
      images = Variable(images.view(-1,28*28)).to(device)
      labels = Variable(labels).to(device)
      
      optimizer.zero_grad()
      outputs = net(images)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()
      
      if (i+1) % 100 == 0:
        print('Epoch [%d/%d], Step [%d/%d]'
                  %(epoch+1, num_epochs, i+1, len(train_data)//batch_size))

  save_model(net, model_path, conda_env=None)
  print("save_model Done")
    

# python -m dstest.pytorch.trainer_cloudpickle  --model_path model/pytorch-cloudpickle
if __name__ == '__main__':
    run_pipeline()
    