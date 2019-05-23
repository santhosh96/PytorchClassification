import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os


class AlexNet():
    
    def __init__(self, trained, classes, model_path):
        self.trained = trained
        self.classes = classes
        self.model_path = model_path
        
    def build_model(self, model_path):
        os.environ['TORCH_HOME'] = self.model_path
        model_ft = models.alexnet(pretrained=self.trained)
        model_ft.classifier[6] = nn.Linear(4096, self.classes)
        return model_ft