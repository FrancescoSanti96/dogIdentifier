#!/usr/bin/env python3
"""
Quick Model Comparison Script
Rapidly compare baseline vs enhanced approaches on a subset of data.
"""

import os
import sys
import torch
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.breed_classifier import SimpleBreedClassifier
from utils.dataloader import create_dataloaders_from_splits
from experiments.enhanced_vs_baseline.dataloader_enhanced import create_enhanced_dataloaders_from_splits

def quick_eval_model(model, data_loader, device, model_name="Model"):
    """Quick evaluation of a model on a data loader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"{model_name} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch and return loss/accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Break early for quick testing (use only first few batches)
        if batch_idx >= 4:  # Only train on first 5 batches for speed
            break
    
    avg_loss = running_loss / min(len(train_loader), 5)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def quick_comparison():
    """Compare baseline vs enhanced approach quickly."""
    print("Quick Model Comparison: Baseline vs Enhanced")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = "../../data/quick_splits"
    batch_size = 16  # Smaller batch for speed
    
    # Create both types of dataloaders
    print("\nCreating baseline dataloaders...")
    baseline_train, baseline_val, baseline_test = create_dataloaders_from_splits(
        data_dir, batch_size=batch_size, num_workers=0  # No workers for speed
    )
    
    print("Creating enhanced dataloaders...")
    enhanced_train, enhanced_val, enhanced_test, breed_to_idx, idx_to_breed = create_enhanced_dataloaders_from_splits(
        data_dir, batch_size=batch_size, num_workers=0
    )
    
    num_classes = len(breed_to_idx)
    print(f"Number of classes: {num_classes}")
    
    # Create models
    baseline_model = SimpleBreedClassifier(num_classes=num_classes, dropout_rate=0.3).to(device)
    enhanced_model = SimpleBreedClassifier(num_classes=num_classes, dropout_rate=0.5).to(device)
    
    # Create optimizers
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    enhanced_optimizer = torch.optim.AdamW(enhanced_model.parameters(), lr=0.001, weight_decay=1e-3)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\nTraining both models for 3 quick epochs...")
    print("-" * 60)
    
    results = {
        'baseline': {'train_acc': [], 'val_acc': []},
        'enhanced': {'train_acc': [], 'val_acc': []}
    }
    
    # Quick training loop
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        # Train baseline
        baseline_loss, baseline_train_acc = train_one_epoch(
            baseline_model, baseline_train, criterion, baseline_optimizer, device
        )
        baseline_val_acc = quick_eval_model(baseline_model, baseline_val, device, "Baseline Val")
        
        # Train enhanced
        enhanced_loss, enhanced_train_acc = train_one_epoch(
            enhanced_model, enhanced_train, criterion, enhanced_optimizer, device
        )
        enhanced_val_acc = quick_eval_model(enhanced_model, enhanced_val, device, "Enhanced Val")
        
        print(f"Baseline - Train Acc: {baseline_train_acc:.1f}%, Val Acc: {baseline_val_acc:.1f}%")
        print(f"Enhanced - Train Acc: {enhanced_train_acc:.1f}%, Val Acc: {enhanced_val_acc:.1f}%")
        
        results['baseline']['train_acc'].append(baseline_train_acc)
        results['baseline']['val_acc'].append(baseline_val_acc)
        results['enhanced']['train_acc'].append(enhanced_train_acc)
        results['enhanced']['val_acc'].append(enhanced_val_acc)
    
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    
    # Final test evaluation
    print("\\nFinal Test Set Evaluation:")
    baseline_test_acc = quick_eval_model(baseline_model, baseline_test, device, "Baseline Test")
    enhanced_test_acc = quick_eval_model(enhanced_model, enhanced_test, device, "Enhanced Test")
    
    # Summary
    print(f"\\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline Final Val:  {results['baseline']['val_acc'][-1]:.1f}%")
    print(f"Enhanced Final Val:  {results['enhanced']['val_acc'][-1]:.1f}%")
    print(f"Improvement:         {results['enhanced']['val_acc'][-1] - results['baseline']['val_acc'][-1]:+.1f}%")
    print()
    print(f"Baseline Test:       {baseline_test_acc:.1f}%")
    print(f"Enhanced Test:       {enhanced_test_acc:.1f}%")
    print(f"Test Improvement:    {enhanced_test_acc - baseline_test_acc:+.1f}%")
    
    # Save results
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'final_val_acc': results['baseline']['val_acc'][-1],
            'final_test_acc': baseline_test_acc,
            'config': 'Standard augmentation, Adam optimizer, dropout 0.3'
        },
        'enhanced': {
            'final_val_acc': results['enhanced']['val_acc'][-1],
            'final_test_acc': enhanced_test_acc,
            'config': 'Albumentations augmentation, AdamW + weight decay, dropout 0.5'
        },
        'improvements': {
            'val_improvement': results['enhanced']['val_acc'][-1] - results['baseline']['val_acc'][-1],
            'test_improvement': enhanced_test_acc - baseline_test_acc
        }
    }
    
    os.makedirs("../../outputs/quick_comparison", exist_ok=True)
    with open("../../outputs/quick_comparison/comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\\nResults saved to: ../../outputs/quick_comparison/comparison_results.json")
    
    # Decision recommendation
    print(f"\\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    
    if enhanced_test_acc > baseline_test_acc + 2.0:
        print("✅ ENHANCED approach shows significant improvement!")
        print("   Recommend using enhanced training for full dataset.")
    elif enhanced_test_acc > baseline_test_acc:
        print("⚡ ENHANCED approach shows modest improvement.")
        print("   Consider enhanced training, but gains may be small.")
    else:
        print("❌ Enhanced approach does not show improvement.")
        print("   Stick with baseline approach for full training.")
    
    return comparison_results

if __name__ == "__main__":
    results = quick_comparison()
