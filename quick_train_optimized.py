#!/usr/bin/env python3
"""
Training ottimizzato con early stopping e regolarizzazione
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_helper import ConfigHelper
from utils.dataloader import DogBreedDataset, get_transforms
from models.breed_classifier import create_breed_classifier

class EarlyStopping:
    """Early stopping per evitare overfitting"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def quick_train_optimized():
    """Training ottimizzato con early stopping e regolarizzazione"""
    print("🚀 Training Ottimizzato - Anti-Overfitting")
    print("="*50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configurazione ottimizzata (parametri migliori)
    data_dir = 'data/breeds'
    num_epochs = 10  # Epoche fisse (senza early stopping)
    batch_size = 32
    learning_rate = 0.001  # Learning rate originale
    patience = 5  # Early stopping patience
    
    # Carica dataset
    print("Caricando dataset...")
    transform, _ = get_transforms((224, 224))
    full_dataset = DogBreedDataset(data_dir, transform=transform)
    
    # Riduci dataset per test veloce (solo prime 10 razze)
    breed_names = full_dataset.get_breed_names()
    if len(breed_names) > 10:
        print(f"Usando solo prime 10 razze per test veloce")
        # Filtra solo prime 10 razze
        filtered_indices = []
        for i, (img_path, label) in enumerate(zip(full_dataset.images, full_dataset.labels)):
            if label < 10:  # Solo prime 10 classi
                filtered_indices.append(i)
        
        # Crea subset
        from torch.utils.data import Subset
        subset_dataset = Subset(full_dataset, filtered_indices)
        num_classes = 10
    else:
        subset_dataset = full_dataset
        num_classes = len(breed_names)
    
    # Split dataset
    total_size = len(subset_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        subset_dataset, [train_size, val_size, test_size]
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {num_classes}")
    
    # Modello con dropout originale
    model = create_breed_classifier(
        model_type='simple',
        num_classes=num_classes,
        dropout_rate=0.3  # Dropout originale (non 0.5)
    )
    model = model.to(device)
    
    # Training setup originale
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Senza weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {num_epochs} epochs (parametri originali)...")
    
    # Training loop ottimizzato
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calcola accuracy
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✅ Nuovo miglior modello! Val Acc: {val_acc:.2f}%")
        
        # Early stopping disabilitato per usare epoche fisse
        # if early_stopping(avg_val_loss):
        #     print(f"🛑 Early stopping! Nessun miglioramento per {patience} epoche")
        #     break
    
    # Carica il miglior modello
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Caricato miglior modello con Val Acc: {best_val_acc:.2f}%")
    
    # Salva modello
    output_dir = 'outputs/quick_test'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'quick_model_optimized.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_acc': train_acc,
        'val_acc': best_val_acc,
        'epoch': epoch + 1,
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    
    print(f"✅ Training completato!")
    print(f"📁 Modello salvato in: {model_path}")
    print(f"🎯 Accuracy finale: Train {train_acc:.2f}%, Val {best_val_acc:.2f}%")
    
    return model, train_acc, best_val_acc

if __name__ == "__main__":
    quick_train_optimized() 