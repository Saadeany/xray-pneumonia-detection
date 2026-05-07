from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import numpy as np


def get_dataloaders(data_dir, batch_size=32):
    """
    Returns train, val, and test DataLoaders.

    Fixes applied:
    - Added test_loader (was missing entirely)
    - Expanded augmentation pipeline (was only 2 transforms)
    - Auto-expands val set if the default split is too small (<100 samples)
    - WeightedRandomSampler is recalculated after any val split adjustment
    """

    # ── Training transform: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),        # Grayscale → RGB for ResNet
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # NEW
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # NEW
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # ── Val/Test transform: no augmentation, just resize + normalize
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # ── Load splits
    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_data   = datasets.ImageFolder(f"{data_dir}/val",   transform=eval_transform)
    test_data  = datasets.ImageFolder(f"{data_dir}/test",  transform=eval_transform)

    # ── Fix: if val set is too small (default chest_xray val = 16 images), borrow from train
    if len(val_data) < 100:
        print(f"[WARNING] Val set has only {len(val_data)} samples. Splitting 15% from train.")
        val_size   = int(0.15 * len(train_data))
        train_size = len(train_data) - val_size
        train_data, val_data = random_split(train_data, [train_size, val_size])
        # After split, targets are accessed differently
        train_targets = [train_data.dataset.targets[i] for i in train_data.indices]
    else:
        train_targets = train_data.targets

    # ── Class balancing via WeightedRandomSampler (train only)
    class_counts  = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # ── Build loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    print(f"Dataset sizes — Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Class counts  — {dict(zip(train_data.dataset.classes if hasattr(train_data, 'dataset') else train_data.classes, class_counts))}")

    return train_loader, val_loader, test_loader