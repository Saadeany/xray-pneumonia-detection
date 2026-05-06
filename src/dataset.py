import torch.nn as nn
from torchvision import models

def get_model():

    model = models.resnet18(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model