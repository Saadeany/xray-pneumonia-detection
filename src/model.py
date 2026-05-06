from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def get_model():

    # Load ImageNet pre-trained weights
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    #model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)

    return model