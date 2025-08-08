#!/usr/bin/env python3
"""
Quick training con 5 razze complete e TensorBoard integrato
Basato su quick_train.py originale con monitoring completo
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_helper import ConfigHelper
from utils.dataloader import create_dataloaders_from_splits, get_transforms
from models.breed_classifier import create_breed_classifier
from utils.early_stopping import EarlyStopping

def quick5_tensorboard_train():
    """Training con 5 razze complete e TensorBoard completo"""
    
    # Header con informazioni
    print("🚀 QUICK 5 BREEDS + TENSORBOARD")
    print("=================================")
    print("📊 Training con 5 razze complete e monitoring TensorBoard")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = f"outputs/tensorboard/quick5_{timestamp}"
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(tb_logdir)
    print(f"📊 TensorBoard logging: {tb_logdir}")
    print(f"   🌐 Avvia TensorBoard: tensorboard --logdir outputs/tensorboard")
    print(f"   🔗 URL: http://localhost:6006")
    
    # Configurazione
    data_dir = 'data/quick_splits'  # Dataset con 5 razze complete
    num_epochs = 15  # Epoche intermedie per test completo
    batch_size = 32
    learning_rate = 0.0008
    patience = 5
    dropout_rate = 0.4
    weight_decay = 5e-5
    
    print(f"\n⚡ CONFIGURAZIONE:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Weight decay: {weight_decay}")
    
    # Log hyperparameters a TensorBoard
    hparams = {
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'dataset': '5_breeds_complete'
    }
    writer.add_hparams(hparams, {'hparam/accuracy': 0, 'hparam/loss': 0})
    
    # Carica dataloaders da splits preorganizzati
    print(f"\n📂 Caricando dataset da: {data_dir}")
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        image_size=(224, 224)
    )
    
    # Verifica breed names
    train_dataset = train_loader.dataset
    breed_names = train_dataset.get_breed_names()
    print(f"🎯 Breeds nel dataset: {len(breed_names)}")
    for i, breed in enumerate(breed_names):
        print(f"   {i}: {breed}")
    
    if 'Australian_Shepherd_Dog' in breed_names:
        australian_idx = breed_names.index('Australian_Shepherd_Dog')
        print(f"✅ Australian_Shepherd_Dog trovato! (indice: {australian_idx})")
    else:
        print(f"⚠️  Australian_Shepherd_Dog NON trovato nel dataset!")
        australian_idx = -1
    
    num_classes = len(breed_names)
    
    print(f"\n📊 Dataset info:")
    print(f"   Training: {len(train_loader.dataset)} samples")
    print(f"   Validation: {len(val_loader.dataset)} samples")
    print(f"   Test: {len(test_loader.dataset)} samples")
    print(f"   Classes: {num_classes}")
    print(f"   Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")
    
    # Modello
    model = create_breed_classifier(
        model_type='simple',
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    print(f"\n🔧 Modello configurato:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametri totali: {total_params:,}")
    print(f"   Parametri trainable: {trainable_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience)
    
    print(f"\n🎯 OBIETTIVI:")
    print(f"   Target generale: >70% validation accuracy")
    if australian_idx >= 0:
        print(f"   Target Australian Shepherd: >60% accuracy")
    print(f"   Train-val gap: <15% (evitare overfitting)")
    
    # Training tracking
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("🚀 STARTING QUICK 5 BREEDS TRAINING")
    print("="*60)
    
    # Training loop
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📅 Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f}")
        
        # ================================
        # TRAINING PHASE
        # ================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Per-class tracking
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Per-class statistics
            for i in range(target.size(0)):
                label = target[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{current_acc:.1f}%',
                'LR': f'{current_lr:.6f}'
            })
            
            # Log batch-level metrics every 5 batches
            if batch_idx % 5 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)
                writer.add_scalar('Batch/Train_Accuracy', current_acc, global_step)
                writer.add_scalar('Batch/Learning_Rate', current_lr, global_step)
        
        # Calculate epoch training metrics
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ================================
        # VALIDATION PHASE
        # ================================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Per-class validation tracking
        val_class_correct = np.zeros(num_classes)
        val_class_total = np.zeros(num_classes)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                # Per-class statistics
                for i in range(target.size(0)):
                    label = target[i].item()
                    pred = predicted[i].item()
                    val_class_total[label] += 1
                    if label == pred:
                        val_class_correct[label] += 1
                
                current_acc = 100. * val_correct / val_total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{current_acc:.1f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # ================================
        # EPOCH LOGGING & ANALYSIS
        # ================================
        
        # Log main metrics to TensorBoard
        writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch + 1)
        writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch + 1)
        writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch + 1)
        writer.add_scalar('Epoch/Learning_Rate', new_lr, epoch + 1)
        
        # Train-val gap analysis
        train_val_gap = train_acc - val_acc
        writer.add_scalar('Analysis/Train_Val_Gap', train_val_gap, epoch + 1)
        
        # Per-class accuracies
        print(f"\n📊 Per-class accuracies (Epoch {epoch+1}):")
        for i, breed in enumerate(breed_names):
            if val_class_total[i] > 0:
                class_acc = 100. * val_class_correct[i] / val_class_total[i]
                writer.add_scalar(f'PerClass/Val_{breed}', class_acc, epoch + 1)
                
                if i == australian_idx:
                    print(f"   {breed[:20]:<20}: {class_acc:5.1f}% ⭐")
                else:
                    print(f"   {breed[:20]:<20}: {class_acc:5.1f}%")
        
        # Target progress tracking
        target_general = 70.0
        target_australian = 60.0
        progress_general = min(100, (val_acc / target_general) * 100)
        writer.add_scalar('Target/General_Progress', progress_general, epoch + 1)
        
        if australian_idx >= 0 and val_class_total[australian_idx] > 0:
            australian_acc = 100. * val_class_correct[australian_idx] / val_class_total[australian_idx]
            progress_australian = min(100, (australian_acc / target_australian) * 100)
            writer.add_scalar('Target/Australian_Progress', progress_australian, epoch + 1)
        
        # Overfitting indicators
        if train_val_gap > 15:
            overfitting_risk = "🟡 MEDIUM"
        elif train_val_gap > 25:
            overfitting_risk = "🔴 HIGH"
        else:
            overfitting_risk = "🟢 LOW"
        
        writer.add_scalar('Analysis/Overfitting_Risk', train_val_gap, epoch + 1)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Current LR: {new_lr:.6f}")
        print(f"   📈 Train-Val Gap: {train_val_gap:.2f}% - Risk: {overfitting_risk}")
        
        if val_acc >= target_general:
            print(f"   🎯 Target raggiunto: {val_acc:.2f}% >= {target_general}%")
        else:
            print(f"   🎯 Progresso target: {val_acc:.2f}% < {target_general}%")
        
        if australian_idx >= 0 and val_class_total[australian_idx] > 0:
            australian_acc = 100. * val_class_correct[australian_idx] / val_class_total[australian_idx]
            if australian_acc >= target_australian:
                print(f"   ⭐ Australian Shepherd: {australian_acc:.1f}% >= {target_australian}% ✅")
            else:
                print(f"   ⭐ Australian Shepherd: {australian_acc:.1f}% < {target_australian}%")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"   🏆 NEW BEST: {val_acc:.2f}% (epoch {epoch+1}) (+{val_acc - best_val_acc:.2f}%)")
            
            # Save best model
            os.makedirs('outputs/quick5', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'breed_names': breed_names,
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'hyperparameters': hparams
            }, 'outputs/quick5/best_model.pth')
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\n🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            writer.add_text('Training/Early_Stop', f'Stopped at epoch {epoch+1}', epoch + 1)
            break
    
    # ================================
    # FINAL ANALYSIS & CLEANUP
    # ================================
    
    print(f"\n📊 FINAL ANALYSIS")
    print("="*40)
    
    # Calculate final per-class accuracies
    final_class_accs = []
    for i, breed in enumerate(breed_names):
        if val_class_total[i] > 0:
            class_acc = 100. * val_class_correct[i] / val_class_total[i]
            final_class_accs.append((breed, class_acc, val_class_total[i]))
    
    # Sort by accuracy
    final_class_accs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP PERFORMERS:")
    for i, (breed, acc, samples) in enumerate(final_class_accs[:3]):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        star = " ⭐" if "Australian" in breed else ""
        print(f"   {medal} {breed:<25}: {acc:5.1f}% ({samples} samples){star}")
    
    # Overall summary
    print(f"\n" + "="*60)
    print(f"🎉 QUICK 5 BREEDS TRAINING COMPLETATO!")
    print("="*60)
    print(f"📊 TensorBoard logs: {tb_logdir}")
    print(f"🌐 Lancia TensorBoard: tensorboard --logdir outputs/tensorboard")
    print(f"🔗 Vai su: http://localhost:6006")
    
    print(f"\n📈 Final Results:")
    print(f"   Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Final Val Acc: {val_acc:.2f}%")
    print(f"   Train-Val Gap: {train_val_gap:.2f}%")
    print(f"   Total Epochs: {epoch+1}")
    
    if australian_idx >= 0 and val_class_total[australian_idx] > 0:
        final_australian_acc = 100. * val_class_correct[australian_idx] / val_class_total[australian_idx]
        print(f"   Australian Shepherd: {final_australian_acc:.1f}%")
    
    # Log final summary to TensorBoard
    writer.add_hparams(hparams, {
        'hparam/final_val_accuracy': val_acc,
        'hparam/best_val_accuracy': best_val_acc,
        'hparam/final_train_accuracy': train_acc,
        'hparam/epochs_completed': epoch + 1
    })
    
    # Final model save
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'breed_names': breed_names,
        'epoch': epoch + 1,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'hyperparameters': hparams,
        'final_results': {
            'best_epoch': best_epoch,
            'train_val_gap': train_val_gap,
            'per_class_accuracies': final_class_accs
        }
    }, 'outputs/quick5/final_model.pth')
    
    print(f"\n💡 TensorBoard Features:")
    print(f"   ✅ Real-time loss/accuracy curves")
    print(f"   ✅ Learning rate scheduling visualization")
    print(f"   ✅ Per-class accuracy tracking")
    print(f"   ✅ Target progress monitoring")
    print(f"   ✅ Overfitting risk analysis")
    print(f"   ✅ Batch-level metrics")
    print(f"   ✅ Hyperparameter logging")
    
    print(f"\n📁 Modelli salvati:")
    print(f"   🏆 Best: outputs/quick5/best_model.pth")
    print(f"   📊 Final: outputs/quick5/final_model.pth")
    
    print(f"\n📊 TRAINING COMPLETE!")
    
    # Close TensorBoard writer
    writer.close()
    
    return {
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'train_acc': train_acc,
        'epochs': epoch + 1,
        'tensorboard_dir': tb_logdir
    }

if __name__ == "__main__":
    results = quick5_tensorboard_train()
    print(f"\n🎯 Training Results: {results}")
