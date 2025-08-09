#!/usr/bin/env python3
"""
Data loading utilities for dog breed identifier project

This module provides:
1. DogBreedDataset: Custom dataset for multi-class breed classification
2. MyDogDataset: Binary classification for personal dog identification
3. Data transformation utilities with augmentation
4. DataLoader creation with proper train/val/test splits
5. Dataset visualization and analysis tools

Author: Francesco Santi
Date: August 2025
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .config_helper import ConfigHelper


class DogBreedDataset(Dataset):
    """
    Custom Dataset for dog breed classification

    This dataset handles loading and preprocessing of dog breed images for multi-class
    classification. It supports both full dataset loading and subset loading for testing.

    Features:
    - Automatic breed folder detection and indexing
    - Image format support: .jpg, .jpeg, .png
    - Optional breed limiting for quick testing
    - Built-in dataset statistics and distribution analysis

    Args:
        data_dir (str): Directory containing breed folders
        transform (transforms.Compose, optional): Image transformations to apply
        max_breeds (int, optional): Maximum number of breeds to include (for testing)

    Example:
        >>> dataset = DogBreedDataset('data/breeds', max_breeds=10)
        >>> print(f"Dataset has {len(dataset)} images from {len(dataset.breed_names)} breeds")
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        max_breeds: Optional[int] = None,
        allowed_breeds: Optional[List[str]] = None,
    ):
        """Initialize dataset with automatic breed discovery and loading"""
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []  # List of image paths
        self.labels = []  # List of breed indices (0, 1, 2, ...)
        self.breed_names = []  # List of breed names (folder names)
        self.max_breeds = max_breeds
        self.allowed_breeds = allowed_breeds

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Load data and build breed mapping
        self._load_data(max_breeds)
        self._validate_dataset()

    def _load_data(self, max_breeds: Optional[int] = None):
        """
        Load image paths and labels from breed folders

        This method:
        1. Discovers all breed folders in data_dir
        2. Optionally limits to max_breeds for testing
        3. Loads all valid image files from each breed folder
        4. Creates label mapping (breed_name -> index)

        Args:
            max_breeds (int, optional): Limit number of breeds for quick testing
        """
        # Get all breed folders
        available_folders = {
            f.name: f
            for f in self.data_dir.iterdir()
            if f.is_dir() and not f.name.startswith(".")
        }

        # If a subset with explicit order is provided, enforce it
        if self.allowed_breeds:
            breed_folders = []
            for breed in self.allowed_breeds:
                if breed in available_folders:
                    breed_folders.append(available_folders[breed])
                else:
                    print(f"⚠️  Allowed breed '{breed}' not found in {self.data_dir}")
        else:
            # Default: alphabetical order for consistency
            breed_folders = sorted(
                list(available_folders.values()), key=lambda p: p.name
            )

        if not breed_folders:
            raise ValueError(f"No breed folders found in {self.data_dir}")

        # Limit breeds for testing if specified
        if max_breeds and not self.allowed_breeds:
            breed_folders = breed_folders[:max_breeds]
            print(f"🔬 Using only first {max_breeds} breeds for testing")

        print(f"📁 Loading {len(breed_folders)} breeds...")

        # Load images from each breed folder
        for breed_idx, breed_folder in enumerate(breed_folders):
            breed_name = breed_folder.name
            self.breed_names.append(breed_name)

            # Get all supported image files
            image_files = self._get_image_files(breed_folder)

            if not image_files:
                print(f"⚠️  No images found in {breed_name} folder")
                continue

            # Add images and labels
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(breed_idx)

            print(f"   {breed_name}: {len(image_files)} images")

        print(
            f"📊 Total loaded: {len(self.images)} images from {len(self.breed_names)} breeds"
        )

    def _get_image_files(self, folder_path: Path) -> List[Path]:
        """
        Get all valid image files from a folder

        Args:
            folder_path (Path): Path to breed folder

        Returns:
            List[Path]: List of valid image file paths
        """
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_files = []

        for extension in supported_extensions:
            image_files.extend(list(folder_path.glob(extension)))

        return sorted(image_files)  # Sort for consistent ordering

    def _validate_dataset(self):
        """
        Validate dataset consistency and report statistics

        Checks:
        - Minimum images per breed
        - Dataset balance
        - Potential issues
        """
        if len(self.images) == 0:
            raise ValueError("No images loaded. Check your data directory structure.")

        # Calculate breed distribution
        breed_counts = Counter(self.labels)
        min_images = min(breed_counts.values())
        max_images = max(breed_counts.values())
        avg_images = np.mean(list(breed_counts.values()))

        print(f"\n📈 Dataset Statistics:")
        print(f"   Min images per breed: {min_images}")
        print(f"   Max images per breed: {max_images}")
        print(f"   Avg images per breed: {avg_images:.1f}")

        # Check for severely imbalanced breeds
        imbalance_ratio = max_images / min_images if min_images > 0 else float("inf")
        if imbalance_ratio > 5:
            print(f"⚠️  Dataset imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
            print("   Consider balancing the dataset for better training results")

        # Show top and bottom breeds by count
        sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
        print(
            f"\n🏆 Most images: {self.breed_names[sorted_breeds[0][0]]} ({sorted_breeds[0][1]} images)"
        )
        print(
            f"🔽 Least images: {self.breed_names[sorted_breeds[-1][0]]} ({sorted_breeds[-1][1]} images)"
        )

    def get_breed_names(self) -> List[str]:
        """Get list of breed names in order of label indices"""
        return self.breed_names.copy()

    def get_breed_distribution(self) -> Dict[str, int]:
        """
        Get distribution of images per breed

        Returns:
            Dict[str, int]: Mapping of breed_name -> image_count
        """
        breed_counts = Counter(self.labels)
        return {self.breed_names[idx]: count for idx, count in breed_counts.items()}

    def __len__(self) -> int:
        """Return total number of images in dataset"""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at specified index

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Tuple[torch.Tensor, int]: (transformed_image, breed_label)

        Raises:
            IndexError: If idx is out of range
            IOError: If image cannot be loaded
        """
        if idx >= len(self.images):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.images)}"
            )

        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            # Load image and convert to RGB (removes alpha channel if present)
            image = Image.open(img_path).convert("RGB")

            # Apply transformations if provided
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

    def __repr__(self) -> str:
        """String representation of the dataset"""
        return (
            f"DogBreedDataset(num_images={len(self.images)}, "
            f"num_breeds={len(self.breed_names)}, "
            f"data_dir='{self.data_dir}')"
        )


class MyDogDataset(Dataset):
    """Custom Dataset for personal dog identification (binary classification)"""

    def __init__(self, data_dir: str, transform=None):
        """
        Initialize personal dog dataset

        Args:
            data_dir: Directory with 'my_dog' and 'other_dogs' folders
            transform: Image transformations
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        """Load image paths and binary labels"""
        my_dog_dir = self.data_dir / "my_dog"
        other_dogs_dir = self.data_dir / "other_dogs"

        # Load my dog images (label 1)
        if my_dog_dir.exists():
            my_dog_images = (
                list(my_dog_dir.glob("*.jpg"))
                + list(my_dog_dir.glob("*.jpeg"))
                + list(my_dog_dir.glob("*.png"))
            )

            for img_path in my_dog_images:
                self.images.append(str(img_path))
                self.labels.append(1)

        # Load other dogs images (label 0)
        if other_dogs_dir.exists():
            other_dog_images = (
                list(other_dogs_dir.glob("*.jpg"))
                + list(other_dogs_dir.glob("*.jpeg"))
                + list(other_dogs_dir.glob("*.png"))
            )

            for img_path in other_dog_images:
                self.images.append(str(img_path))
                self.labels.append(0)

        print(
            f"🐕 Personal dog dataset: {sum(self.labels)} my dog, {len(self.labels) - sum(self.labels)} other dogs"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and binary label at index"""
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataset_splits(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Create physical train/validation/test splits by copying files to separate folders

    This function creates a proper dataset structure with separate folders for
    train, validation, and test sets. This is better than random splits because:
    1. Ensures consistent splits across different runs
    2. Allows for proper evaluation without data leakage
    3. Makes it easier to debug and analyze specific sets

    Args:
        source_dir (str): Source directory containing breed folders
        output_dir (str): Output directory where splits will be created
        train_ratio (float): Proportion for training set (default: 0.7)
        val_ratio (float): Proportion for validation set (default: 0.15)
        test_ratio (float): Proportion for test set (default: 0.15)
        seed (int): Random seed for reproducible splits

    Directory structure created:
        output_dir/
        ├── train/
        │   ├── breed1/
        │   ├── breed2/
        │   └── ...
        ├── val/
        │   ├── breed1/
        │   ├── breed2/
        │   └── ...
        └── test/
            ├── breed1/
            ├── breed2/
            └── ...

    Example:
        >>> create_dataset_splits('data/breeds', 'data/splits')
        >>> # Creates data/splits/train/, data/splits/val/, data/splits/test/
    """
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Creating dataset splits in {output_dir}")
    print(f"   Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Get all breed folders
    breed_folders = sorted(
        [f for f in source_path.iterdir() if f.is_dir() and not f.name.startswith(".")]
    )

    total_files_copied = 0
    split_stats = {"train": 0, "val": 0, "test": 0}

    for breed_folder in breed_folders:
        breed_name = breed_folder.name
        print(f"   Processing {breed_name}...")

        # Create breed folders in each split
        (train_dir / breed_name).mkdir(exist_ok=True)
        (val_dir / breed_name).mkdir(exist_ok=True)
        (test_dir / breed_name).mkdir(exist_ok=True)

        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(list(breed_folder.glob(ext)))

        if not image_files:
            print(f"     ⚠️  No images found in {breed_name}")
            continue

        # Shuffle images for random split
        image_files = list(image_files)
        np.random.shuffle(image_files)

        # Calculate split indices
        n_images = len(image_files)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        n_test = n_images - n_train - n_val  # Remaining goes to test

        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]

        # Copy files to respective directories
        for files, target_dir, split_name in [
            (train_files, train_dir / breed_name, "train"),
            (val_files, val_dir / breed_name, "val"),
            (test_files, test_dir / breed_name, "test"),
        ]:
            for img_file in files:
                target_path = target_dir / img_file.name
                shutil.copy2(img_file, target_path)
                split_stats[split_name] += 1

        total_files_copied += len(image_files)
        print(f"     ✅ {breed_name}: {n_train} train, {n_val} val, {n_test} test")

    print(f"\n✅ Dataset splitting completed!")
    print(f"   Total files processed: {total_files_copied}")
    print(f"   Train: {split_stats['train']} files")
    print(f"   Validation: {split_stats['val']} files")
    print(f"   Test: {split_stats['test']} files")
    print(f"   Output directory: {output_path}")


def get_transforms(
    image_size: Tuple[int, int] = (224, 224), augmentation_config: Optional[Dict] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation image transforms

    This function creates two separate transform pipelines:
    1. Training transforms: Include data augmentation for better generalization
    2. Validation transforms: Only normalization and resizing (no augmentation)

    The transforms follow ImageNet normalization standards for transfer learning
    compatibility and consistent performance across different architectures.

    Args:
        image_size (Tuple[int, int]): Target image size (width, height)
        augmentation_config (Dict, optional): Configuration for data augmentation
            Expected keys:
            - horizontal_flip (bool): Enable random horizontal flipping
            - rotation (int): Maximum rotation degrees
            - brightness_contrast (List[float]): [brightness_factor, contrast_factor]
            - color_jitter (List[float]): [saturation, hue, -, -] (last two unused)

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transform, val_transform)

    Example:
        >>> train_tf, val_tf = get_transforms((224, 224), {
        ...     'horizontal_flip': True,
        ...     'rotation': 15,
        ...     'brightness_contrast': [0.8, 1.2]
        ... })
    """
    # ImageNet normalization values - standard for most pre-trained models
    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channel means
    IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB channel standard deviations

    # Base transforms for validation (no augmentation)
    # Use Resize(256) + CenterCrop to avoid aspect ratio distortion
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Training transforms with optional augmentation
    if augmentation_config:
        transform_list = []

        # 1. Spatial cropping strategy: RandomResizedCrop or Resize+RandomCrop
        if augmentation_config.get("random_resized_crop", False):
            rrc_scale = augmentation_config.get("rrc_scale", (0.8, 1.0))
            rrc_ratio = augmentation_config.get("rrc_ratio", (0.9, 1.1))
            transform_list.append(
                transforms.RandomResizedCrop(
                    image_size, scale=rrc_scale, ratio=rrc_ratio
                )
            )
        else:
            resize_factor = 1.1  # 10% larger than target
            enlarged_size = (
                int(image_size[0] * resize_factor),
                int(image_size[1] * resize_factor),
            )
            transform_list.append(transforms.Resize(enlarged_size))
            transform_list.append(transforms.RandomCrop(image_size))

        # 2. Random horizontal/vertical flips
        if augmentation_config.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if augmentation_config.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.1))

        # 3. Random rotation
        rotation_degrees = augmentation_config.get("rotation", 0)
        if rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

        # 4. Perspective distortion
        perspective_p = augmentation_config.get("perspective_p", 0.0)
        if perspective_p and perspective_p > 0:
            perspective_scale = augmentation_config.get("perspective_scale", 0.3)
            transform_list.append(
                transforms.RandomPerspective(
                    distortion_scale=perspective_scale, p=perspective_p
                )
            )

        # 5. Color jittering
        brightness_contrast = augmentation_config.get("brightness_contrast", [1.0, 1.0])
        color_jitter = augmentation_config.get("color_jitter", [0.0, 0.0, 0.0, 0.0])
        if brightness_contrast != [1.0, 1.0] or any(
            val > 0 for val in color_jitter[:2]
        ):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=(
                        brightness_contrast[0] if brightness_contrast[0] != 1.0 else 0
                    ),
                    contrast=(
                        brightness_contrast[1] if brightness_contrast[1] != 1.0 else 0
                    ),
                    saturation=color_jitter[0],
                    hue=color_jitter[1],
                )
            )

        # 6. ToTensor + Normalize
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # 7. Random Erasing (applied on tensors)
        erasing_p = augmentation_config.get("erasing_p", 0.0)
        if erasing_p and erasing_p > 0:
            erasing_scale = augmentation_config.get("erasing_scale", (0.02, 0.1))
            erasing_ratio = augmentation_config.get("erasing_ratio", (0.3, 3.3))
            transform_list.append(
                transforms.RandomErasing(
                    p=erasing_p, scale=erasing_scale, ratio=erasing_ratio
                )
            )

        train_transform = transforms.Compose(transform_list)

        print(f"🎨 Data augmentation enabled:")
        print(
            f"   RandomResizedCrop: {augmentation_config.get('random_resized_crop', False)}"
        )
        print(
            f"   Horizontal flip: {augmentation_config.get('horizontal_flip', False)}"
        )
        print(f"   Vertical flip: {augmentation_config.get('vertical_flip', False)}")
        print(f"   Rotation: ±{rotation_degrees}°")
        print(f"   Perspective: p={perspective_p}")
        print(f"   Brightness/Contrast: {brightness_contrast}")
        print(f"   Color jitter (sat/hue): {color_jitter[:2]}")
        print(f"   RandomErasing p: {erasing_p}")

    else:
        # No augmentation - use same transforms as validation
        train_transform = val_transform
        print("🎨 No data augmentation applied")

    return train_transform, val_transform


def create_dataloaders_from_splits(
    splits_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
    allowed_breeds: Optional[List[str]] = None,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders from pre-split dataset directories

    This function loads data from separate train/val/test directories created by
    create_dataset_splits(). This approach is better than random splits because:
    1. Ensures consistent evaluation across runs
    2. Prevents data leakage between sets
    3. Allows for proper statistical analysis

    Args:
        splits_dir (str): Directory containing train/, val/, test/ folders
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        image_size (Tuple[int, int]): Target image size
        augmentation_config (Dict, optional): Data augmentation configuration

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)

    Expected directory structure:
        splits_dir/
        ├── train/
        │   ├── breed1/
        │   └── breed2/
        ├── val/
        │   ├── breed1/
        │   └── breed2/
        └── test/
            ├── breed1/
            └── breed2/

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders_from_splits('data/splits')
    """
    splits_path = Path(splits_dir)

    # Validate directory structure
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        dir_path = splits_path / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")

    # Get transforms
    # Ensure training pipeline uses crop to avoid distortion; validation uses CenterCrop
    # If augmentation_config not provided, enable RandomResizedCrop by default for train
    if augmentation_config is None:
        augmentation_config = {"random_resized_crop": True}
    train_transform, val_transform = get_transforms(image_size, augmentation_config)

    print(f"📁 Loading datasets from splits: {splits_dir}")

    # Create datasets for each split
    train_dataset = DogBreedDataset(
        str(splits_path / "train"),
        transform=train_transform,
        allowed_breeds=allowed_breeds,
    )

    val_dataset = DogBreedDataset(
        str(splits_path / "val"), transform=val_transform, allowed_breeds=allowed_breeds
    )

    test_dataset = DogBreedDataset(
        str(splits_path / "test"),
        transform=val_transform,
        allowed_breeds=allowed_breeds,
    )

    # Verify all datasets have the same breeds
    train_breeds = set(train_dataset.get_breed_names())
    val_breeds = set(val_dataset.get_breed_names())
    test_breeds = set(test_dataset.get_breed_names())

    if not (train_breeds == val_breeds == test_breeds):
        print("⚠️  Warning: Not all splits contain the same breeds!")
        print(f"   Train breeds: {len(train_breeds)}")
        print(f"   Val breeds: {len(val_breeds)}")
        print(f"   Test breeds: {len(test_breeds)}")

        # Find missing breeds
        all_breeds = train_breeds | val_breeds | test_breeds
        for split_name, breeds in [
            ("train", train_breeds),
            ("val", val_breeds),
            ("test", test_breeds),
        ]:
            missing = all_breeds - breeds
            if missing:
                print(f"   Missing from {split_name}: {sorted(missing)}")

    # If allowed_breeds provided, report the enforced order
    if allowed_breeds:
        print(f"🔢 Enforcing breed order: {allowed_breeds}")

    # Create dataloaders
    if use_weighted_sampler:
        # Compute class-balanced sample weights for the training set
        import numpy as _np

        labels_np = _np.array(train_dataset.labels)
        class_counts = _np.bincount(
            labels_np, minlength=len(train_dataset.breed_names)
        ).astype(_np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.max() / class_counts
        sample_weights = [class_weights[label] for label in labels_np]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Drop incomplete batches for consistent training
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"✅ Dataloaders created successfully:")
    print(f"   Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of classes: {len(train_dataset.get_breed_names())}")

    return train_loader, val_loader, test_loader


def create_dataloaders(
    config: ConfigHelper, max_breeds: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        config: Configuration helper
        max_breeds: Maximum number of breeds (for testing)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config.get_data_config()
    augmentation_config = config.get_augmentation_config()

    # Get transforms
    train_transform, val_transform = get_transforms(
        tuple(data_config["image_size"]), augmentation_config
    )

    # Create full dataset
    full_dataset = DogBreedDataset(
        data_config["breed_dataset_path"],
        transform=train_transform,
        max_breeds=max_breeds,
    )

    # Split dataset
    train_size = int(data_config["train_split"] * len(full_dataset))
    val_size = int(data_config["val_split"] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size]
    )

    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )

    # Update transforms for validation and test
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    print(f"📊 Dataset split:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    print(f"   Classes: {len(full_dataset.get_breed_names())}")

    return train_loader, val_loader, test_loader


def visualize_dataset_distribution(
    dataset: DogBreedDataset, save_path: Optional[str] = None
):
    """
    Visualize dataset distribution across breeds

    Args:
        dataset: DogBreedDataset instance
        save_path: Path to save plot
    """
    breed_names = dataset.get_breed_names()
    breed_counts = {}

    for label in dataset.labels:
        breed_name = breed_names[label]
        breed_counts[breed_name] = breed_counts.get(breed_name, 0) + 1

    # Create plot
    plt.figure(figsize=(15, 8))

    breeds = list(breed_counts.keys())
    counts = list(breed_counts.values())

    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    breeds = [breeds[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    # Plot top 20 breeds
    top_n = min(20, len(breeds))
    plt.bar(range(top_n), counts[:top_n])
    plt.xticks(range(top_n), breeds[:top_n], rotation=45, ha="right")
    plt.ylabel("Number of Images")
    plt.title("Dataset Distribution - Top 20 Breeds")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Distribution plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test dataloader
    config = ConfigHelper()

    # Create sample data for testing
    sample_data_dir = Path("data/breeds")
    if not sample_data_dir.exists():
        print("⚠️  No dataset found. Create sample data for testing...")
        sample_data_dir.mkdir(parents=True, exist_ok=True)

        # Create a few sample breed folders
        for i, breed in enumerate(["sample_breed_1", "sample_breed_2"]):
            breed_dir = sample_data_dir / breed
            breed_dir.mkdir(exist_ok=True)
            print(f"📁 Created {breed_dir}")

    # Test dataset loading
    try:
        dataset = DogBreedDataset(str(sample_data_dir), max_breeds=2)
        print(f"✅ Dataset loaded successfully: {len(dataset)} images")
        print(f"📋 Breeds: {dataset.get_breed_names()}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure you have the Stanford Dogs dataset in data/breeds/")
