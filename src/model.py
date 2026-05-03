from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def get_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model