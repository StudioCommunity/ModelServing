import os
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from builtin_score.builtin_score_module import BuiltinScoreModule
import builtin_score.ioutil as ioutil

def load_model_then_predict(model_path = "./model/pytorch-mnist/"):
    batch_size = 64
    test_data = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor())
    test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                        batch_size = batch_size, 
                                        shuffle = False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', device)
    x = test_data.data[:8]
    x_input = x.view(-1, 28*28).float() # input here need to be float type

    input_numpy_array = x_input.cpu().numpy() # construct input dataframe (Only cpu type tensor can convert to df)
    input_df = pd.DataFrame(input_numpy_array)
    params = {
        "Append score columns to output": "False"
    }
    module = BuiltinScoreModule(model_path, params)
    #module = BuiltinScoreModule(model_path)
    result = module.run(input_df)
    print('=====buildinScoreModule=======')
    print(result)
    #result.columns = result.columns.astype(str)
    ioutil.save_parquet(result, './testOutputParquet/')

# python -m dstest.pytorch.saved_model_predict_test
if __name__ == '__main__':
    load_model_then_predict()
