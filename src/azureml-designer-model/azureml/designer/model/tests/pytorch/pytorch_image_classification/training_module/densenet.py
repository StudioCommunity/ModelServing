import sys
import torch.nn as nn
import torchvision.models.densenet as densenet


class DenseNet(nn.Module):
    def __init__(self,
                 model_type='densenet201',
                 pretrained=True,
                 memory_efficient=False,
                 num_classes=20):
        super(DenseNet, self).__init__()
        if not pretrained:
            model_type = "densenet201"
        densenet_func = getattr(densenet, model_type, None)
        if densenet_func is None:
            # todo: catch exception and throw AttributeError
            sys.exit()
        self.model1 = densenet_func(pretrained=pretrained)
        # Pretrained model is trained based on 1000-class ImageNet dataset
        self.model2 = nn.Linear(1000, num_classes)

    def forward(self, input):
        output = self.model1(input)
        output = self.model2(output)
        return output
