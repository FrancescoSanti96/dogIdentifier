#!/usr/bin/env python3
"""
Training rapido per test con dataset ridotto
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_helper import ConfigHelper
from utils.dataloader import create_dataloaders_from_splits, get_transforms
from models.breed_classifier import create_breed_classifier
from utils.early_stopping import EarlyStopping

def quick_train():
    """Training rapido con dataset ridotto"""
    print("🚀 Training Rapido - Test Setup")
    print("==================================================")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Definisco data_dir per il quick test con Australian Shepherd
    data_dir = 'data/quick_splits'  # Usa il dataset con splits separati incluso Australian Shepherd

    # Configurazione intermedia
    num_epochs = 12  # Epoche intermedie
    batch_size = 32
    learning_rate = 0.0008  # Learning rate intermedio
    patience = 7  # Early stopping patience intermedia
    
    # Carica dataloaders da splits preorganizzati
    print("Caricando dataset da splits preorganizzati...")
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        image_size=(224, 224)
    )
    
    # Verifica breed names dalla struttura del train set
    train_dataset = train_loader.dataset
    breed_names = train_dataset.get_breed_names()
    print(f"🎯 Razze nel dataset: {breed_names}")
    
    if 'Australian_Shepherd_Dog' in breed_names:
        print(f"✅ Australian_Shepherd_Dog trovato!")
    else:
        print(f"⚠️  Australian_Shepherd_Dog NON trovato nel dataset!")
    
    num_classes = len(breed_names)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {num_classes}")
    
    # Modello con dropout intermedio
    model = create_breed_classifier(
        model_type='simple',
        num_classes=num_classes,
        dropout_rate=0.4  # Dropout intermedio
    )
    model = model.to(device)
    
    # Training setup intermedio
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)  # Weight decay intermedio
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {num_epochs} epochs...")
    
    # Initialize accuracy variables for final reporting
    train_acc = 0.0
    val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if early_stopping(avg_val_loss):
            print(f"🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            break
    
    # Salva modello
    os.makedirs('outputs/quick_splits', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'epoch': num_epochs,
        'train_acc': train_acc,
        'val_acc': val_acc
    }, 'outputs/quick_splits/quick_model.pth')
    
    print(f"\n✅ Training completato!")
    print(f"📁 Modello salvato in: outputs/quick_splits/quick_model.pth")
    print(f"🎯 Accuracy finale: Train {train_acc:.2f}%, Val {val_acc:.2f}%")

if __name__ == "__main__":
    quick_train() 