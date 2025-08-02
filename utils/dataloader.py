#!/usr/bin/env python3
"""
Data loading utilities for dog breed identifier project
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .config_helper import ConfigHelper


class DogBreedDataset(Dataset):
    """Custom Dataset for dog breed classification"""
    
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 max_breeds: Optional[int] = None):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing breed folders
            transform: Image transformations
            max_breeds: Maximum number of breeds to include (for testing)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.breed_names = []
        
        self._load_data(max_breeds)
    
    def _load_data(self, max_breeds: Optional[int] = None):
        """Load image paths and labels"""
        breed_folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        
        if max_breeds:
            breed_folders = breed_folders[:max_breeds]
        
        print(f"📁 Loading {len(breed_folders)} breeds...")
        
        for breed_idx, breed_folder in enumerate(breed_folders):
            breed_name = breed_folder.name
            self.breed_names.append(breed_name)
            
            # Get all image files in breed folder
            image_files = list(breed_folder.glob('*.jpg')) + \
                         list(breed_folder.glob('*.jpeg')) + \
                         list(breed_folder.glob('*.png'))
            
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(breed_idx)
        
        print(f"📊 Loaded {len(self.images)} images from {len(self.breed_names)} breeds")
    
    def get_breed_names(self) -> List[str]:
        """Get list of breed names"""
        return self.breed_names
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label at index"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MyDogDataset(Dataset):
    """Custom Dataset for personal dog identification (binary classification)"""
    
    def __init__(self, 
                 data_dir: str,
                 transform=None):
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
        my_dog_dir = self.data_dir / 'my_dog'
        other_dogs_dir = self.data_dir / 'other_dogs'
        
        # Load my dog images (label 1)
        if my_dog_dir.exists():
            my_dog_images = list(my_dog_dir.glob('*.jpg')) + \
                           list(my_dog_dir.glob('*.jpeg')) + \
                           list(my_dog_dir.glob('*.png'))
            
            for img_path in my_dog_images:
                self.images.append(str(img_path))
                self.labels.append(1)
        
        # Load other dogs images (label 0)
        if other_dogs_dir.exists():
            other_dog_images = list(other_dogs_dir.glob('*.jpg')) + \
                              list(other_dogs_dir.glob('*.jpeg')) + \
                              list(other_dogs_dir.glob('*.png'))
            
            for img_path in other_dog_images:
                self.images.append(str(img_path))
                self.labels.append(0)
        
        print(f"🐕 Personal dog dataset: {sum(self.labels)} my dog, {len(self.labels) - sum(self.labels)} other dogs")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and binary label at index"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size: Tuple[int, int] = (224, 224), 
                  augmentation_config: Optional[Dict] = None) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms
    
    Args:
        image_size: Target image size (width, height)
        augmentation_config: Augmentation configuration
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Base transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Training transforms with augmentation
    if augmentation_config:
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5 if augmentation_config.get('horizontal_flip', False) else 0.0),
            transforms.RandomRotation(degrees=augmentation_config.get('rotation', 0)),
            transforms.ColorJitter(
                brightness=augmentation_config.get('brightness_contrast', [1.0, 1.0])[0],
                contrast=augmentation_config.get('brightness_contrast', [1.0, 1.0])[1],
                saturation=augmentation_config.get('color_jitter', [0.0, 0.0, 0.0, 0.0])[0],
                hue=augmentation_config.get('color_jitter', [0.0, 0.0, 0.0, 0.0])[1]
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = val_transform
    
    return train_transform, val_transform


def create_dataloaders(config: ConfigHelper, 
                      max_breeds: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        tuple(data_config['image_size']), 
        augmentation_config
    )
    
    # Create full dataset
    full_dataset = DogBreedDataset(
        data_config['breed_dataset_path'],
        transform=train_transform,
        max_breeds=max_breeds
    )
    
    # Split dataset
    train_size = int(data_config['train_split'] * len(full_dataset))
    val_size = int(data_config['val_split'] * len(full_dataset))
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
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    print(f"   Classes: {len(full_dataset.get_breed_names())}")
    
    return train_loader, val_loader, test_loader


def visualize_dataset_distribution(dataset: DogBreedDataset, 
                                 save_path: Optional[str] = None):
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
    plt.xticks(range(top_n), breeds[:top_n], rotation=45, ha='right')
    plt.ylabel('Number of Images')
    plt.title('Dataset Distribution - Top 20 Breeds')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        for i, breed in enumerate(['sample_breed_1', 'sample_breed_2']):
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