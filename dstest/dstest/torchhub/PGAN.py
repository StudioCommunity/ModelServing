# https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/

import numpy as np
import pandas as pd
import torch
import builtin_models.python
from builtin_score.builtin_score_module import *
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms as T

class PGAN(builtin_models.python.PythonModel):
    def __init__(self, model_path = None):
        #self.device = 'cpu' # cuda only
        use_gpu = True if torch.cuda.is_available() else False

      # trained on high-quality celebrity faces "celebA" dataset
      # this model outputs 512 x 512 pixel images
        model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)
        
        # this model outputs 256 x 256 pixel images
        # model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
        #                        'PGAN', model_name='celebAHQ-256',
        #                        pretrained=True, useGPU=use_gpu)
        #print('# model parameters:', sum(param.numel() for param in model.parameters()))

        self.model = model
    
    #The input to the model is a noise vector of shape (N, 512) where N is the number of images to be generated.
    def predict(self, text):
        num_images = 4
        noise, _ = self.model.buildNoiseData(num_images)
        with torch.no_grad():
            generated_images = self.model.test(noise)

        # let's plot these images using torchvision and matplotlib
        grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
        image_tensor = grid.permute(1, 2, 0).cpu()
        print(image_tensor.shape)
        image_tensor = image_tensor.squeeze(0)
        to_pil = T.ToPILImage()
        img = to_pil(image_tensor)
        img.save("test.jpg", format="JPEG")

        #plt.imshow(graph.cpu().numpy())
        # plt.show()

# python -m dstest.torchhub.PGAN
if __name__ == '__main__':
    model_path = "model/pgan"
    github = 'StudioCommunity/CustomModules:master'
    module = 'dstest/dstest/torchhub/PGAN.py'
    model_class = 'PGAN'

    model = PGAN()
    model.predict(None)

    #model = Tacotron2Model()
    #builtin_models.python.save_model(model, model_path, github = github, module_path = module, model_class= model_class)

    # model1 = builtin_models.python.load_model(model_path, github = github, module_path = module, model_class= model_class, force_reload= True)

    # text = "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."
    # x = np.array([text])
    # # run the models
    # audios = model1.predict(x)
    # #print(audios.shape)
