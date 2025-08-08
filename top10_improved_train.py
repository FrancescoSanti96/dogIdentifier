#!/usr/bin/env python3
"""
Training MIGLIORATO per TOP 10 BALANCED
Learning Rate ottimizzato + ReduceLROnPlateau
Obiettivo: >50% validation accuracy dal 28.81% attuale
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

def top10_improved_train():
    """Training MIGLIORATO per TOP 10 razze bilanciate"""
    print("🚀 TOP 10 IMPROVED TRAINING")
    print("===========================")
    print("⚡ LR Ottimizzato + ReduceLROnPlateau")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 📊 TENSORBOARD SETUP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = f'outputs/tensorboard/top10_improved_{timestamp}'
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)
    print(f"📊 TensorBoard logging: {tb_log_dir}")
    print(f"   Run: tensorboard --logdir {tb_log_dir}")
    
    # Dataset TOP 10 bilanciato (stesso di prima)
    data_dir = 'data/top10_balanced'
    
    # ⚡ CONFIGURAZIONE MIGLIORATA
    num_epochs = 25          # +10 epoche (vs 15)
    batch_size = 32          # Stesso
    learning_rate = 0.001    # 2x più aggressivo (vs 0.0005)
    patience_early = 8       # +2 pazienza early stopping (vs 6)
    dropout_rate = 0.4       # Stesso (funzionava bene)
    
    print(f"\\n⚡ CONFIGURAZIONE MIGLIORATA:")
    print(f"   Epochs: {num_epochs} (vs 15 previous)")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate} (vs 0.0005 - 2x più aggressivo)")
    print(f"   Early stopping patience: {patience_early} (vs 6)")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Scheduler: ReduceLROnPlateau (adattivo)")
    
    # Carica dataloaders (identici a prima)
    print(f"\\nCaricando dataset TOP 10 balanced...")
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
    print(f"📋 Lista breeds: {sorted(breed_names)}")
    
    if 'Australian_Shepherd_Dog' in breed_names:
        print(f"✅ Australian_Shepherd_Dog trovato!")
        aus_shep_idx = breed_names.index('Australian_Shepherd_Dog')
        print(f"   Indice classe: {aus_shep_idx}")
    else:
        print(f"⚠️  Australian_Shepherd_Dog NON trovato!")
    
    num_classes = len(breed_names)
    
    print(f"\\n📊 Dataset info:")
    print(f"   Training: {len(train_loader.dataset)} samples")
    print(f"   Validation: {len(val_loader.dataset)} samples")
    print(f"   Test: {len(test_loader.dataset)} samples")
    print(f"   Classes: {num_classes}")
    print(f"   Batches per epoch: {len(train_loader)} train, {len(val_loader)} val")
    
    # Modello identico ma con stesso dropout
    model = create_breed_classifier(
        model_type='simple',
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    # Training setup MIGLIORATO
    criterion = nn.CrossEntropyLoss()
    
    # ⚡ Adam con LR più aggressivo
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4
    )
    
    # 🎯 ReduceLROnPlateau scheduler (CHIAVE MIGLIORAMENTO)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',              # Monitor validation accuracy
        factor=0.6,              # Riduce del 40% quando plateau
        patience=4,              # Aspetta 4 epoche senza miglioramento
        min_lr=5e-5,             # LR minimo (0.00005)
        threshold=0.01,          # Miglioramento minimo 1%
        cooldown=2               # Aspetta 2 epoche prima di ridurre ancora
    )
    print(f"   Scheduler: ReduceLROnPlateau configurato (mode=max, factor=0.6, patience=4)")
    
    early_stopping = EarlyStopping(patience=patience_early, delta=0)
    
    print(f"\\n🔧 Modello configurato:")
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametri totali: {model_params:,}")
    print(f"   Parametri trainable: {trainable_params:,}")
    
    # 📊 Log hyperparameters to TensorBoard
    writer.add_hparams({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'dropout_rate': dropout_rate,
        'patience_early': patience_early,
        'scheduler': 'ReduceLROnPlateau',
        'weight_decay': 5e-4,
        'num_classes': num_classes,
        'total_params': model_params,
        'trainable_params': trainable_params
    }, {'baseline_val_acc': 28.81, 'baseline_aus_acc': 38.1})
    
    print(f"\\n🎯 OBIETTIVI MIGLIORATI:")
    print(f"   Target primario: Validation Accuracy > 50% (vs 28.81% previous)")
    print(f"   Target stretch: Validation Accuracy > 60%")
    print(f"   Train-Val gap: < 25% (mantieni non-overfitting)")
    print(f"   Australian Shepherd: > 50% (vs 38.1% previous)")
    
    # Training tracking
    best_val_acc = 0.0
    best_epoch = 0
    train_accs = []
    val_accs = []
    lr_history = []
    
    print(f"\\n" + "=" * 60)
    print(f"🚀 STARTING TOP 10 IMPROVED TRAINING")
    print(f"=" * 60)
    
    # Training loop con LR tracking
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f"\\n📅 Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f}")
        
        # Training phase
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
            
            # Gradient clipping (stesso di prima)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar ogni 5 batches 
            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%',
                    'LR': f'{current_lr:.6f}'
                })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Per-class tracking
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        
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
                
                # Per-class statistics
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_total[label] += 1
                    if predicted[i] == target[i]:
                        class_correct[label] += 1
                
                # Update progress bar ogni batch
                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{100.*val_correct/val_total:.1f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Store accuracies
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 📊 TensorBoard Logging
        global_step = epoch
        
        # Loss and accuracy curves
        writer.add_scalars('Loss', {
            'Train': avg_train_loss,
            'Validation': avg_val_loss
        }, global_step)
        
        writer.add_scalars('Accuracy', {
            'Train': train_acc,
            'Validation': val_acc
        }, global_step)
        
        # Learning rate tracking
        writer.add_scalar('Learning_Rate', current_lr, global_step)
        
        # Train-Val gap analysis
        gap = abs(train_acc - val_acc)
        writer.add_scalar('Train_Val_Gap', gap, global_step)
        
        # Per-class accuracies per TensorBoard
        class_accs_dict = {}
        aus_acc_current = 0
        for i, breed in enumerate(breed_names):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                breed_clean = breed.replace('_', ' ')
                class_accs_dict[breed_clean] = class_acc
                
                if breed == 'Australian_Shepherd_Dog':
                    aus_acc_current = class_acc
                    writer.add_scalar('Australian_Shepherd_Accuracy', class_acc, global_step)
        
        # Log all class accuracies
        writer.add_scalars('Per_Class_Accuracy', class_accs_dict, global_step)
        
        # Target achievement tracking
        writer.add_scalars('Target_Achievement', {
            'Val_vs_50%_Target': val_acc - 50,
            'Val_vs_Baseline': val_acc - 28.81,
            'AusShep_vs_50%_Target': aus_acc_current - 50,
            'AusShep_vs_Baseline': aus_acc_current - 38.1
        }, global_step)
        
        # Results display con miglioramenti evidenziati
        print(f"\\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Current LR: {current_lr:.6f}")
        
        # Overfitting analysis
        print(f"   📈 Train-Val Gap: {gap:.2f}%", end="")
        
        if gap < 10:
            print(" ✅ EXCELLENT!")
        elif gap < 20:
            print(" 👍 Good")
        elif gap < 30:
            print(" 🟡 OK")
        else:
            print(" ⚠️ Overfitting!")
        
        # Target achievement check MIGLIORATO
        if val_acc >= 60:
            print(f"   🎯🎉 STRETCH TARGET! {val_acc:.2f}% >= 60%")
        elif val_acc >= 50:
            print(f"   🎯✅ PRIMARY TARGET! {val_acc:.2f}% >= 50%")
        elif val_acc >= 40:
            print(f"   🎯👍 Great progress: {val_acc:.2f}% >= 40% (vs 28.81% baseline)")
        elif val_acc >= 30:
            print(f"   🎯📈 Good progress: {val_acc:.2f}% >= 30% (improving from baseline)")
        
        # Australian Shepherd tracking
        if 'Australian_Shepherd_Dog' in breed_names:
            aus_idx = breed_names.index('Australian_Shepherd_Dog')
            if class_total[aus_idx] > 0:
                aus_acc = 100. * class_correct[aus_idx] / class_total[aus_idx]
                print(f"   ⭐ Australian Shepherd: {aus_acc:.1f}%", end="")
                
                if aus_acc >= 60:
                    print(" 🎉 EXCELLENT!")
                elif aus_acc >= 50:
                    print(" ✅ TARGET REACHED!")
                elif aus_acc >= 40:
                    print(" 👍 Good (vs 38.1% baseline)")
                elif aus_acc >= 30:
                    print(" 📈 Improving")
                else:
                    print(" ⚠️")
        
        # Save best model
        improvement = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            improvement = f" (+{val_acc - val_accs[-2] if len(val_accs) > 1 else val_acc:.2f}%)"
            print(f"   🏆 NEW BEST: {best_val_acc:.2f}% (epoch {best_epoch}){improvement}")
        
        # ⚡ ReduceLROnPlateau step (CHIAVE)
        scheduler.step(val_acc)  # Passa validation accuracy
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\\n🛑 Early stopping after {epoch+1} epochs")
            print(f"   No improvement for {patience_early} epochs")
            break
    
    # 📊 Log final training curves to TensorBoard
    for i, (train_acc, val_acc) in enumerate(zip(train_accs, val_accs)):
        writer.add_scalars('Final_Training_Curves', {
            'Train_Accuracy': train_acc,
            'Val_Accuracy': val_acc
        }, i)
    
    # Final per-class analysis (stesso di prima)
    print(f"\\n" + "=" * 60)
    print(f"📊 FINAL PER-CLASS ANALYSIS")
    print(f"=" * 60)
    
    final_class_accuracies = []
    aus_data = None  # Initialize to avoid unbound error
    for i, breed in enumerate(breed_names):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            samples = int(class_total[i])
            final_class_accuracies.append((breed, class_acc, samples))
            if breed == 'Australian_Shepherd_Dog':
                aus_data = (breed, class_acc, samples)
    
    # Sort by accuracy
    final_class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\\n🏆 RANKING FINALE (validation accuracy):")
    for i, (breed, acc, samples) in enumerate(final_class_accuracies, 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        target_marker = " ⭐" if breed == 'Australian_Shepherd_Dog' else "  "
        
        # Evidenzia miglioramenti vs baseline
        improvement_marker = ""
        if breed == 'Australian_Shepherd_Dog':
            if acc >= 50:
                improvement_marker = " 🚀 BIG IMPROVEMENT!"
            elif acc > 38.1:
                improvement_marker = f" 📈 +{acc-38.1:.1f}% vs baseline"
        
        print(f"   {status} {breed.replace('_', ' '):20s}: {acc:5.1f}% ({samples} samples){target_marker}{improvement_marker}")
    
    # Australian Shepherd specific analysis con confronto
    if 'Australian_Shepherd_Dog' in breed_names:
        aus_data = next((breed, acc, samples) for breed, acc, samples in final_class_accuracies if breed == 'Australian_Shepherd_Dog')
        position = next(i for i, (breed, _, _) in enumerate(final_class_accuracies, 1) if breed == 'Australian_Shepherd_Dog')
        baseline_aus_acc = 38.1
        
        print(f"\\n⭐ AUSTRALIAN SHEPHERD ANALYSIS:")
        print(f"   Final accuracy: {aus_data[1]:.1f}%")
        print(f"   Baseline accuracy: {baseline_aus_acc}%")
        print(f"   Improvement: {aus_data[1] - baseline_aus_acc:+.1f}%")
        print(f"   Ranking position: {position}/{len(breed_names)}")
        print(f"   Validation samples: {aus_data[2]}")
        
        if aus_data[1] >= 60:
            print(f"   Status: 🎉 STRETCH TARGET ACHIEVED!")
        elif aus_data[1] >= 50:
            print(f"   Status: ✅ PRIMARY TARGET ACHIEVED!")
        elif aus_data[1] > baseline_aus_acc:
            print(f"   Status: 📈 MIGLIORAMENTO vs baseline!")
        else:
            print(f"   Status: ⚠️ Sotto baseline")
    
    # Save model con info miglioramenti
    os.makedirs('outputs/top10_improved', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'breed_names': breed_names,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_val_acc': val_accs[-1] if val_accs else 0,
        'class_accuracies': {breed: acc for breed, acc, _ in final_class_accuracies},
        'training_history': {
            'train_accs': train_accs,
            'val_accs': val_accs,
            'lr_history': lr_history
        },
        'improvements_vs_baseline': {
            'baseline_val_acc': 28.81,
            'baseline_aus_acc': 38.1,
            'improved_val_acc': best_val_acc,
            'improved_aus_acc': aus_data[1] if aus_data else 0
        },
        'config': {
            'model_type': 'simple_improved',
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'scheduler': 'ReduceLROnPlateau',
            'dataset_type': 'top10_balanced_improved',
            'from_scratch': True
        }
    }, 'outputs/top10_improved/top10_improved_model.pth')
    
    # Final summary con confronto baseline
    final_gap = abs(train_accs[-1] - val_accs[-1]) if train_accs and val_accs else 0
    baseline_val_acc = 28.81
    
    # 📊 Final TensorBoard logging
    writer.add_text('Training_Summary', f"""
    ## Training Summary
    
    **Best Validation Accuracy:** {best_val_acc:.2f}% (epoch {best_epoch})
    **Baseline Comparison:** {best_val_acc - baseline_val_acc:+.2f}%
    **Australian Shepherd:** {aus_data[1] if aus_data else 0:.1f}% (vs 38.1% baseline)
    **Final Train-Val Gap:** {final_gap:.2f}%
    **Training Status:** {'SUCCESS' if best_val_acc >= 50 else 'PARTIAL SUCCESS' if best_val_acc > baseline_val_acc else 'NEEDS IMPROVEMENT'}
    
    **Hyperparameters:**
    - Learning Rate: {learning_rate}
    - Batch Size: {batch_size}  
    - Epochs Trained: {len(train_accs)}
    - Scheduler: ReduceLROnPlateau
    - Dropout: {dropout_rate}
    """)
    
    # Log final metrics
    writer.add_scalars('Final_Results', {
        'Best_Val_Accuracy': best_val_acc,
        'Final_Train_Accuracy': train_accs[-1] if train_accs else 0,
        'Final_Val_Accuracy': val_accs[-1] if val_accs else 0,
        'Train_Val_Gap': final_gap,
        'Improvement_vs_Baseline': best_val_acc - baseline_val_acc
    })
    
    if aus_data:
        writer.add_scalar('Final_Australian_Shepherd_Accuracy', aus_data[1])
        writer.add_scalar('Australian_Shepherd_Improvement', aus_data[1] - 38.1)
    
    # Close TensorBoard writer
    writer.close()
    print(f"📊 TensorBoard logs saved to: {tb_log_dir}")
    
    print(f"\\n" + "=" * 60)
    print(f"🚀 TRAINING MIGLIORATO COMPLETATO!")
    print(f"=" * 60)
    print(f"📁 Model saved: outputs/top10_improved/top10_improved_model.pth")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    print(f"\\n📊 CONFRONTO BASELINE vs MIGLIORATO:")
    print(f"   Baseline Val Acc: {baseline_val_acc:.2f}%")
    print(f"   Improved Val Acc: {best_val_acc:.2f}%")
    print(f"   Miglioramento: {best_val_acc - baseline_val_acc:+.2f}%")
    
    print(f"\\n📈 Final Results:")
    print(f"   Train: {train_accs[-1] if train_accs else 0:.2f}%")
    print(f"   Val: {val_accs[-1] if val_accs else 0:.2f}%")
    print(f"   Gap: {final_gap:.2f}%")
    print(f"🎯 Dataset: {num_classes} balanced breeds")
    
    # Success evaluation con soglie aggiornate
    if best_val_acc >= 50:
        print(f"\\n🎯🎉 PRIMARY TARGET ACHIEVED! {best_val_acc:.2f}% >= 50%")
        print(f"✅ Excellent improvement from baseline {baseline_val_acc:.2f}%!")
        print(f"🚀 Ready for production deployment!")
        return True
    elif best_val_acc > baseline_val_acc + 5:
        print(f"\\n📈 SIGNIFICANT IMPROVEMENT! +{best_val_acc - baseline_val_acc:.2f}% vs baseline")
        print(f"👍 Good progress, consider further tuning")
        return True
    elif best_val_acc > baseline_val_acc:
        print(f"\\n📈 MIGLIORAMENTO: +{best_val_acc - baseline_val_acc:.2f}% vs baseline")
        print(f"💡 Progress made, try different strategies")
        return False
    else:
        print(f"\\n⚠️ Performance vs baseline: {best_val_acc - baseline_val_acc:+.2f}%")
        print(f"💡 Try alternative approaches")
        return False

if __name__ == "__main__":
    print("🚀 TOP 10 Improved Training - LR Ottimizzato")
    print("=" * 50)
    
    success = top10_improved_train()
    
    if success:
        print("\\n🎉 SUCCESS! Miglioramenti significativi raggiunti!")
    else:
        print("\\n🔄 Ulteriori ottimizzazioni raccomandate.")
