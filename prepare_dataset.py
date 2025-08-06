#!/usr/bin/env python3
"""
Dataset preparation script for Dog Breed Identifier

This script helps organize the dataset into proper train/validation/test splits
and provides dataset analysis tools.

Usage:
    python prepare_dataset.py --split  # Create train/val/test splits
    python prepare_dataset.py --analyze  # Analyze dataset distribution
    python prepare_dataset.py --quick-test  # Create small test dataset

Author: Francesco Santi
Date: August 2025
"""

import argparse
import sys
from pathlib import Path
import shutil
import numpy as np
from collections import Counter

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.dataloader import create_dataset_splits, DogBreedDataset


def analyze_dataset(data_dir: str):
    """
    Analyze dataset distribution and statistics
    
    Args:
        data_dir (str): Path to dataset directory
    """
    print("🔍 DATASET ANALYSIS")
    print("=" * 50)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return
    
    # Get all breed folders
    breed_folders = sorted([f for f in data_path.iterdir() 
                           if f.is_dir() and not f.name.startswith('.')])
    
    if not breed_folders:
        print(f"❌ No breed folders found in {data_dir}")
        return
    
    print(f"📁 Found {len(breed_folders)} breeds")
    
    # Analyze each breed
    breed_stats = {}
    total_images = 0
    
    for breed_folder in breed_folders:
        breed_name = breed_folder.name
        
        # Count images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(breed_folder.glob(ext)))
        
        image_count = len(image_files)
        breed_stats[breed_name] = image_count
        total_images += image_count
    
    # Calculate statistics
    counts = list(breed_stats.values())
    min_images = min(counts)
    max_images = max(counts)
    avg_images = np.mean(counts)
    median_images = np.median(counts)
    std_images = np.std(counts)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total images: {total_images:,}")
    print(f"   Total breeds: {len(breed_folders)}")
    print(f"   Min images per breed: {min_images}")
    print(f"   Max images per breed: {max_images}")
    print(f"   Average images per breed: {avg_images:.1f}")
    print(f"   Median images per breed: {median_images:.1f}")
    print(f"   Standard deviation: {std_images:.1f}")
    
    # Imbalance analysis
    imbalance_ratio = max_images / min_images if min_images > 0 else float('inf')
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 5:
        print("   ⚠️  HIGH IMBALANCE DETECTED!")
    elif imbalance_ratio > 2:
        print("   ⚠️  Moderate imbalance detected")
    else:
        print("   ✅ Dataset is reasonably balanced")
    
    # Show top and bottom breeds
    sorted_breeds = sorted(breed_stats.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP 10 BREEDS (most images):")
    for i, (breed, count) in enumerate(sorted_breeds[:10]):
        print(f"   {i+1:2d}. {breed}: {count} images")
    
    print(f"\n🔽 BOTTOM 10 BREEDS (least images):")
    for i, (breed, count) in enumerate(sorted_breeds[-10:]):
        print(f"   {i+1:2d}. {breed}: {count} images")
    
    # Identify problematic breeds (very few images)
    problematic_breeds = [breed for breed, count in breed_stats.items() if count < 10]
    if problematic_breeds:
        print(f"\n⚠️  BREEDS WITH <10 IMAGES ({len(problematic_breeds)} breeds):")
        for breed in problematic_breeds:
            print(f"   - {breed}: {breed_stats[breed]} images")


def create_quick_test_dataset(source_dir: str, output_dir: str, max_breeds: int = 5):
    """
    Create a small dataset for quick testing
    
    Args:
        source_dir (str): Source dataset directory
        output_dir (str): Output directory for quick test dataset
        max_breeds (int): Maximum number of breeds to include
    """
    print(f"🚀 CREATING QUICK TEST DATASET")
    print("=" * 50)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get breed folders sorted by image count (take most balanced ones)
    print("📊 Analyzing breeds for selection...")
    
    breed_folders = sorted([f for f in source_path.iterdir() 
                           if f.is_dir() and not f.name.startswith('.')])
    
    # Count images per breed
    breed_counts = {}
    for breed_folder in breed_folders:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(breed_folder.glob(ext)))
        breed_counts[breed_folder.name] = len(image_files)
    
    # ALWAYS include Australian_Shepherd_Dog (essential for the project)
    must_include = ['Australian_Shepherd_Dog']
    selected_breeds = []
    
    # First, add must-include breeds if they exist
    for breed_name in must_include:
        if breed_name in breed_counts:
            selected_breeds.append((breed_name, breed_counts[breed_name]))
            print(f"🎯 FORCED inclusion: {breed_name} ({breed_counts[breed_name]} images)")
        else:
            print(f"⚠️  Warning: {breed_name} not found in dataset!")
    
    # Calculate how many more breeds we need
    remaining_breeds_needed = max_breeds - len(selected_breeds)
    
    if remaining_breeds_needed > 0:
        # Select additional breeds with good number of images (avoid extremes)
        available_breeds = [(name, count) for name, count in breed_counts.items() 
                           if name not in must_include]
        sorted_breeds = sorted(available_breeds, key=lambda x: x[1], reverse=True)
        
        # Take breeds from the middle range (not too many, not too few)
        start_idx = len(sorted_breeds) // 4  # Skip top 25%
        end_idx = start_idx + remaining_breeds_needed
        additional_breeds = sorted_breeds[start_idx:end_idx]
        
        selected_breeds.extend(additional_breeds)
    
    print(f"📋 Selected breeds for quick testing:")
    total_images = 0
    for breed, count in selected_breeds:
        marker = "🎯" if breed in must_include else "  "
        print(f"   {marker} {breed}: {count} images")
        total_images += count
    
    print(f"📊 Total images in quick test set: {total_images}")
    
    # Copy selected breeds
    for breed_name, _ in selected_breeds:
        source_breed_dir = source_path / breed_name
        target_breed_dir = output_path / breed_name
        
        print(f"   Copying {breed_name}...")
        shutil.copytree(source_breed_dir, target_breed_dir, dirs_exist_ok=True)
    
    print(f"✅ Quick test dataset created in: {output_path}")
    print(f"   🎯 Australian_Shepherd_Dog included for project requirements!")
    print(f"   Use this for rapid prototyping and testing!")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for dog breed classification")
    parser.add_argument('--source', type=str, default='data/breeds',
                       help='Source dataset directory (default: data/breeds)')
    parser.add_argument('--output', type=str, default='data/splits',
                       help='Output directory for splits (default: data/splits)')
    
    # Action selection
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--split', action='store_true',
                             help='Create train/val/test splits')
    action_group.add_argument('--analyze', action='store_true',
                             help='Analyze dataset distribution')
    action_group.add_argument('--quick-test', action='store_true',
                             help='Create small test dataset')
    
    # Split configuration
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    
    # Quick test configuration
    parser.add_argument('--max-breeds', type=int, default=5,
                       help='Maximum breeds for quick test (default: 5)')
    parser.add_argument('--quick-output', type=str, default='data/quick_test',
                       help='Output directory for quick test (default: data/quick_test)')
    
    args = parser.parse_args()
    
    try:
        if args.analyze:
            analyze_dataset(args.source)
            
        elif args.split:
            create_dataset_splits(
                source_dir=args.source,
                output_dir=args.output,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
            
        elif args.quick_test:
            create_quick_test_dataset(
                source_dir=args.source,
                output_dir=args.quick_output,
                max_breeds=args.max_breeds
            )
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
