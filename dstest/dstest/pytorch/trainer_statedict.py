import os
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from dstest.pytorch.MnistModel import MnistNet
import cloudpickle
import pickle
import yaml
import json

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


model_file_name = "model.states"
model_init_args = "model.init"

def save_model_spec(model_path):
    spec = {
        'flavor' : {
            'framework' : 'pytorch'
        },
        "pytorch": {
            "serialization_format": "statedict",
            "model_file_path": model_file_name,
            "model_class_package": "MnistModel",
            "model_class_name" : "MnistNet",
            "model_class_init_args" : model_init_args
        },
    }

    with open(os.path.join(model_path, "model_spec.yml"), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)


def save_model(model_path, model, init_args):
    if(not model_path.endswith('/')):
        model_path += '/'
    
    if not os.path.exists(model_path):
        logger.info(f"{model_path} does not exist")
        os.makedirs(model_path)
    else:
        logger.info(f"{model_path} exists")

    with open(model_path+model_init_args, 'wb') as fp:
        pickle.dump(init_args, fp) 
    torch.save(model.state_dict(), model_path+model_file_name)

    


def save_ilearner(model_path):
    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "ILearnerDotNet",
        "Name": "ILearner .NET file",
        "ShortName": "Model",
        "Description": "A .NET serialized ILearner",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "ilearner",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": False,
        "AllowModelPromotion": True,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(model_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)
    # Dump data.ilearner as a work around until data type design
    visualization = os.path.join(model_path, "data.ilearner")
    with open(visualization, 'w') as file:
        file.writelines('{}')


@click.command()
@click.option('--action', default="train", 
        type=click.Choice(['predict', 'train']))
@click.option('--model_path', default="./model/")
def run_pipeline(action, model_path):
  input_size = 784 # img_size = (28,28) ---> 28*28=784 in total
  hidden_size = 500 # number of nodes at hidden layer
  num_classes = 10 # number of output classes discrete range [0,9]
  num_epochs = 20 # number of times which the entire dataset is passed throughout the model
  batch_size = 100 # the size of input data took for one iteration
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
  if torch.cuda.is_available():
    net.cuda()

  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam( net.parameters(), lr=lr)

  for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_gen):
      images = Variable(images.view(-1,28*28))
      labels = Variable(labels)
      if torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()
      
      optimizer.zero_grad()
      outputs = net(images)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()
      
      if (i+1) % 100 == 0:
        print('Epoch [%d/%d], Step [%d/%d]'
                  %(epoch+1, num_epochs, i+1, len(train_data)//batch_size))

  save_model(model_path, net, (input_size, hidden_size, num_classes))
  save_model_spec(model_path)
  save_ilearner(model_path)
  print("Done")

# python -m dstest.pytorch.trainer_statedict  --model_path model/pytorch-statedict
if __name__ == '__main__':
    run_pipeline()
    