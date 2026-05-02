from torchvision import models
import torch.nn as nn

def get_model():
    model = models.resnet18(pretrained=True)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model