from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def get_dataloaders(data_dir, batch_size=32):

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)

    class_counts = np.bincount(train_data.targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in train_data.targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)

    return train_loader