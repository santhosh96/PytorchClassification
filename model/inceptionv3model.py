import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from base import BaseModel
import os


class InceptionV3():
    
    def __init__(self, trained, classes, model_path):
        self.trained = trained
        self.classes = classes
        self.model_path = model_path

    def build_model(self, model_path):
        os.environ['TORCH_MODEL_ZOO'] = model_path
        model_ft = models.inception_v3(pretrained=self.trained)
        model_ft.AuxLogits.fc = nn.Linear(768, 5)
        model_ft.fc = nn.Linear(2048, 5)
        return model_ft
