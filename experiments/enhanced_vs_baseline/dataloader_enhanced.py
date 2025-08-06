"""
Enhanced DataLoader for Dog Breed Classification with Advanced Augmentation
Uses Albumentations for sophisticated data augmentation strategies.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, List, Tuple, Optional

class DogBreedDatasetEnhanced(Dataset):
    """Enhanced dataset with Albumentations augmentation support."""
    
    def __init__(self, data_dir: str, breed_to_idx: Dict[str, int], 
                 albumentations_transform=None, pytorch_transform=None):
        self.data_dir = data_dir
        self.breed_to_idx = breed_to_idx
        self.albumentations_transform = albumentations_transform
        self.pytorch_transform = pytorch_transform
        
        # Load all image paths and labels
        self.samples = []
        for breed_name in os.listdir(data_dir):
            breed_path = os.path.join(data_dir, breed_name)
            if os.path.isdir(breed_path) and breed_name in breed_to_idx:
                breed_idx = breed_to_idx[breed_name]
                for img_name in os.listdir(breed_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(breed_path, img_name)
                        self.samples.append((img_path, breed_idx))
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as RGB
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply Albumentations transform if provided
        if self.albumentations_transform:
            transformed = self.albumentations_transform(image=image_np)
            image = transformed['image']
            
            # If Albumentations includes ToTensorV2, image is already a tensor
            if isinstance(image, torch.Tensor):
                return image, label
            else:
                # Convert to PIL for PyTorch transforms
                image = Image.fromarray(image.astype(np.uint8))
        
        # Apply PyTorch transforms if provided
        if self.pytorch_transform:
            image = self.pytorch_transform(image)
            
        return image, label

def get_enhanced_transforms(image_size: int = 224, split: str = 'train') -> Tuple[Optional[A.Compose], Optional[transforms.Compose]]:
    """
    Get enhanced transforms using Albumentations for training and basic transforms for validation/test.
    
    Args:
        image_size: Target image size
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (albumentations_transform, pytorch_transform)
    """
    
    if split == 'train':
        # Enhanced training augmentations with Albumentations
        albumentations_transform = A.Compose([
            # Geometric transformations
            A.Resize(image_size + 32, image_size + 32),  # Slightly larger for cropping
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1, 
                rotate_limit=15,
                border_mode=0,
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Color augmentations
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ], p=0.7),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1), p=1.0),  # Normalized range 0-1
                A.GaussianBlur(blur_limit=(1, 3), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Cutout/Erase
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 16), 
                hole_width_range=(8, 16),
                fill=0,
                p=0.3
            ),
            
            # Normalization and tensor conversion
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
        
        return albumentations_transform, None
        
    else:
        # Simple transforms for validation/test using PyTorch
        pytorch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return None, pytorch_transform

def create_enhanced_dataloaders_from_splits(
    splits_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    breeds_subset: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Create enhanced dataloaders from physical train/val/test splits.
    
    Args:
        splits_dir: Directory containing train/, val/, test/ subdirectories
        batch_size: Batch size for dataloaders
        image_size: Image size for preprocessing
        num_workers: Number of worker processes
        breeds_subset: Optional list of breeds to include (filters available breeds)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, breed_to_idx, idx_to_breed)
    """
    
    train_dir = os.path.join(splits_dir, 'train')
    val_dir = os.path.join(splits_dir, 'val')
    test_dir = os.path.join(splits_dir, 'test')
    
    # Verify directories exist
    for split_dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Get available breeds from train directory
    available_breeds = [d for d in os.listdir(train_dir) 
                       if os.path.isdir(os.path.join(train_dir, d))]
    
    # Filter breeds if subset specified
    if breeds_subset:
        available_breeds = [breed for breed in available_breeds if breed in breeds_subset]
        print(f"Using breed subset: {available_breeds}")
    
    if not available_breeds:
        raise ValueError("No valid breeds found in the dataset")
    
    # Create breed mappings
    breed_to_idx = {breed: idx for idx, breed in enumerate(sorted(available_breeds))}
    idx_to_breed = {idx: breed for breed, idx in breed_to_idx.items()}
    
    print(f"Found {len(available_breeds)} breeds: {list(breed_to_idx.keys())}")
    
    # Get transforms for each split
    train_albu_transform, _ = get_enhanced_transforms(image_size, 'train')
    _, val_pytorch_transform = get_enhanced_transforms(image_size, 'val')
    _, test_pytorch_transform = get_enhanced_transforms(image_size, 'test')
    
    # Create datasets
    train_dataset = DogBreedDatasetEnhanced(
        train_dir, breed_to_idx, 
        albumentations_transform=train_albu_transform
    )
    
    val_dataset = DogBreedDatasetEnhanced(
        val_dir, breed_to_idx,
        pytorch_transform=val_pytorch_transform
    )
    
    test_dataset = DogBreedDatasetEnhanced(
        test_dir, breed_to_idx,
        pytorch_transform=test_pytorch_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created enhanced dataloaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples") 
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, breed_to_idx, idx_to_breed

if __name__ == "__main__":
    # Test the enhanced dataloader
    splits_dir = "data/quick_splits"
    train_loader, val_loader, test_loader, breed_to_idx, idx_to_breed = create_enhanced_dataloaders_from_splits(
        splits_dir, batch_size=4
    )
    
    print("\nTesting enhanced transforms...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Train batch {batch_idx}: images shape={images.shape}, labels={labels}")
        if batch_idx >= 2:
            break
    
    for batch_idx, (images, labels) in enumerate(val_loader):
        print(f"Val batch {batch_idx}: images shape={images.shape}, labels={labels}")
        if batch_idx >= 1:
            break
