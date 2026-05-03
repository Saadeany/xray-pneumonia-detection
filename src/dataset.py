from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# ImageNet mean/std — used for both training and inference so that the
# model always sees inputs in the same distribution it was pretrained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_dataloaders(data_dir, batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)

    class_counts = np.bincount(train_data.targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in train_data.targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)

    val_data = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader(data_dir, batch_size=32):

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_data = datasets.ImageFolder(f"{data_dir}/test", transform=test_transform)
    return DataLoader(test_data, batch_size=batch_size, shuffle=False)