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

class DCGAN(builtin_models.python.PythonModel):
    def __init__(self, model_path = None):
        #self.device = 'cpu' # cuda only
        use_gpu = True if torch.cuda.is_available() else False

        model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)
        #print('# model parameters:', sum(param.numel() for param in model.parameters()))

        self.model = model
    
    #The input to the model is a noise vector of shape (N, 512) where N is the number of images to be generated.
    def predict(self, num_images = 64):
        noise, _ = self.model.buildNoiseData(num_images)
        with torch.no_grad():
            generated_images = self.model.test(noise)

        # let's plot these images using torchvision and matplotlib
        import matplotlib.pyplot as plt
        import torchvision
        image_tensor = torchvision.utils.make_grid(generated_images).permute(1, 2, 0).cpu()

        plt.imshow(image_tensor.numpy())
        # plt.show()
        print(image_tensor.shape)
        image_tensor = image_tensor.squeeze(0)
        to_pil = T.ToPILImage()
        img = to_pil(image_tensor)
        img.save("test.jpg", format="JPEG")

        #plt.imshow(graph.cpu().numpy())
        # plt.show()

# python -m dstest.torchhub.DCGAN
if __name__ == '__main__':
    model_path = "model/pgan"
    github = 'StudioCommunity/CustomModules:master'
    module = 'dstest/dstest/torchhub/PGAN.py'
    model_class = 'PGAN'

    model = DCGAN()
    model.predict()

    #model = Tacotron2Model()
    #builtin_models.python.save_model(model, model_path, github = github, module_path = module, model_class= model_class)

    # model1 = builtin_models.python.load_model(model_path, github = github, module_path = module, model_class= model_class, force_reload= True)

    # text = "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."
    # x = np.array([text])
    # # run the models
    # audios = model1.predict(x)
    # #print(audios.shape)
