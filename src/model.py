from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def get_model():
    # 1. Load ImageNet pre-trained ResNet18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # 2. Freeze all early layers so we don't destroy the pre-trained ImageNet edge-detectors
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze the final convolutional block (ResNet's equivalent to DenseNet's denseblock4)
    # This allows the AI to learn specific lung textures (like fluid/consolidation)
    for param in model.layer4.parameters():
        param.requires_grad = True


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 Output classes: NORMAL and PNEUMONIA

    return model