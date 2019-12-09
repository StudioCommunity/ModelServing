import fire
import torch.nn as nn
from .basenet import BaseNet


class DenseNet(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_nn()
        logger.info(f"Model init finished, {self.model}.")

    def update_nn(self):
        if self.pretrained:
            num_classes = self.kwargs.get('num_classes', None)
            num_final_in = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_final_in, num_classes)
