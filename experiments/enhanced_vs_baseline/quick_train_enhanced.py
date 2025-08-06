#!/usr/bin/env python3
"""
Enhanced Quick Training Script for Dog Breed Classification
Includes advanced data augmentation, improved regularization, and optimized training settings.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.breed_classifier import SimpleBreedClassifier
from experiments.enhanced_vs_baseline.dataloader_enhanced import create_enhanced_dataloaders_from_splits
from utils.early_stopping import EarlyStopping
from utils.metrics import calculate_metrics

def label_smoothing_loss(outputs, targets, num_classes, smoothing=0.1):
    """
    Apply label smoothing to cross entropy loss.
    """
    confidence = 1.0 - smoothing
    smooth_positives = confidence
    smooth_negatives = smoothing / (num_classes - 1)
    
    # Create smoothed labels
    batch_size = targets.size(0)
    smoothed_labels = torch.full((batch_size, num_classes), smooth_negatives, device=targets.device)
    smoothed_labels.scatter_(1, targets.unsqueeze(1), smooth_positives)
    
    # Calculate loss using log probabilities
    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / batch_size
    
    return loss

def train_model_enhanced(
    model, train_loader, val_loader, test_loader, 
    num_epochs=15, patience=5, save_dir="../../outputs/quick_enhanced",
    use_label_smoothing=True, smoothing=0.1
):
    """
    Enhanced training function with improved regularization and optimization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Enhanced optimizer - AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-3,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler - Cosine annealing with restarts
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Loss function with optional label smoothing
    if use_label_smoothing:
        criterion = lambda outputs, targets: label_smoothing_loss(
            outputs, targets, model.num_classes, smoothing
        )
        print(f"Using label smoothing with factor: {smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, delta=0)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\nStarting enhanced training for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping(val_loss_avg):
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            break
        
        print("-" * 60)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights for final evaluation")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if use_label_smoothing:
                # Use standard CE for test evaluation
                loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss_avg = test_loss / len(test_loader)
    test_acc = 100.0 * test_correct / test_total
    
    print(f"Test Loss: {test_loss_avg:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'best_model_enhanced.pth')
    torch.save({
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'num_classes': model.num_classes,
        'final_val_acc': best_val_acc,
        'final_test_acc': test_acc,
        'history': history
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history_enhanced.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save summary report
    summary = {
        'training_config': {
            'num_epochs': num_epochs,
            'patience': patience,
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 1e-3,
            'scheduler': 'CosineAnnealingLR',
            'use_label_smoothing': use_label_smoothing,
            'label_smoothing_factor': smoothing if use_label_smoothing else None,
            'gradient_clipping': 1.0,
            'data_augmentation': 'Enhanced Albumentations'
        },
        'final_results': {
            'best_val_accuracy': best_val_acc,
            'final_test_accuracy': test_acc,
            'total_training_time': total_time,
            'total_epochs': len(history['train_loss'])
        },
        'model_info': {
            'architecture': 'SimpleBreedClassifier',
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'num_classes': model.num_classes
        }
    }
    
    summary_path = os.path.join(save_dir, 'training_summary_enhanced.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {summary_path}")
    
    return model, history, best_val_acc, test_acc

def main():
    print("Enhanced Dog Breed Classifier Training")
    print("=" * 50)
    
    # Configuration
    data_dir = "../../data/quick_splits"
    batch_size = 32
    num_epochs = 12  # Quick validation run
    patience = 4
    save_dir = "../../outputs/quick_enhanced"
    
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Save directory: {save_dir}")
    
    # Create enhanced dataloaders
    print("\nCreating enhanced dataloaders...")
    train_loader, val_loader, test_loader, breed_to_idx, idx_to_breed = create_enhanced_dataloaders_from_splits(
        data_dir, batch_size=batch_size, num_workers=2
    )
    
    print(f"\nBreed mappings:")
    for breed, idx in breed_to_idx.items():
        print(f"  {idx}: {breed}")
    
    # Create enhanced model with increased dropout
    print(f"\nCreating enhanced model...")
    model = SimpleBreedClassifier(
        num_classes=len(breed_to_idx),
        dropout_rate=0.5  # Increased dropout for better regularization
    )
    
    print(f"Model created with {len(breed_to_idx)} classes")
    
    # Train enhanced model
    model, history, best_val_acc, test_acc = train_model_enhanced(
        model, train_loader, val_loader, test_loader,
        num_epochs=num_epochs, patience=patience, save_dir=save_dir,
        use_label_smoothing=True, smoothing=0.1
    )
    
    print(f"\n{'='*60}")
    print("ENHANCED TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Results saved to: {save_dir}")
    
    # Quick comparison note
    print(f"\n{'='*60}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*60}")
    print("Enhanced features implemented:")
    print("• Advanced Albumentations data augmentation")
    print("• AdamW optimizer with weight decay (1e-3)")
    print("• Label smoothing (0.1)")
    print("• Increased dropout (0.5)")
    print("• Cosine annealing LR scheduler")
    print("• Gradient clipping")
    print("")
    print("Run the baseline quick_train.py to compare results!")

if __name__ == "__main__":
    main()
