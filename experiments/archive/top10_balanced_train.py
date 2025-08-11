#!/usr/bin/env python3
"""
Training ottimizzato per TOP 10 BALANCED
9 razze più popolari + Australian Shepherd
Obiettivo: >60% validation accuracy senza overfitting
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
from utils.seed_utils import set_deterministic


def top10_balanced_train():
    """Training per TOP 10 razze bilanciate"""
    print("🎯 TOP 10 BALANCED TRAINING")
    print("===========================")
    print("🏆 9 razze popolari + Australian Shepherd")

    # Setup
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset TOP 10 bilanciato
    data_dir = "data/top10_balanced"

    # Configurazione BILANCIATA per 10 classi
    num_epochs = 15  # Più epoche per convergenza completa
    batch_size = 32  # Batch size standard
    learning_rate = 0.0005  # LR moderato per from-scratch
    patience = 6  # Patience bilanciata

    print(f"\\n⚙️  CONFIGURAZIONE BILANCIATA:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Patience: {patience}")
    print(f"   Dropout: 0.4 (moderato)")

    # Carica dataloaders (abilita WeightedRandomSampler per bilanciare classi)
    print(f"\\nCaricando dataset TOP 10 balanced...")
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        image_size=(224, 224),
        augmentation_config={
            "random_resized_crop": True,
            "horizontal_flip": True,
            "rotation": 10,
            "brightness_contrast": [0.85, 1.15],
        },
        use_weighted_sampler=True,
    )

    # Verifica breed names
    train_dataset = train_loader.dataset
    breed_names = train_dataset.get_breed_names()
    print(f"🎯 Breeds nel dataset: {len(breed_names)}")
    print(f"📋 Lista breeds: {sorted(breed_names)}")

    if "Australian_Shepherd_Dog" in breed_names:
        print(f"✅ Australian_Shepherd_Dog trovato!")
        aus_shep_idx = breed_names.index("Australian_Shepherd_Dog")
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

    # Modello bilanciato per 10 classi (opzione transfer learning via env)
    use_tl = bool(int(os.getenv("USE_TL", "0")))
    if use_tl:
        print("\n🧠 Using transfer learning backbone: ResNet18 (frozen)")
        model = create_breed_classifier(
            model_type="simple",  # ignored if pretrained_backbone set
            num_classes=num_classes,
            dropout_rate=0.4,
            pretrained_backbone="resnet18",
            freeze_backbone=True,
        )
    else:
        model = create_breed_classifier(
            model_type="simple", num_classes=num_classes, dropout_rate=0.4
        )
    model = model.to(device)

    # Training setup bilanciato
    criterion = nn.CrossEntropyLoss()

    # Adam con weight decay moderato
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4  # Weight decay moderato
    )

    # Scheduler step-based più prevedibile
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.8  # Riduce ogni 5 epoche  # Riduce del 20%
    )

    early_stopping = EarlyStopping(patience=patience, delta=0)

    print(f"\\n🔧 Modello configurato:")
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametri totali: {model_params:,}")
    print(f"   Parametri trainable: {trainable_params:,}")

    print(f"\\n🎯 OBIETTIVI:")
    print(f"   Target primario: Validation Accuracy > 50%")
    print(f"   Target stretch: Validation Accuracy > 60%")
    print(f"   Train-Val gap: < 20% (evitare overfitting)")
    print(f"   Australian Shepherd: > 50%")

    # Training tracking
    best_val_acc = 0.0
    best_epoch = 0
    train_accs = []
    val_accs = []

    print(f"\\n" + "=" * 50)
    print(f"🚀 STARTING TOP 10 BALANCED TRAINING")
    print(f"=" * 50)

    # Training loop
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
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

            # Gradient clipping moderato
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            # Update progress bar ogni 5 batches
            if batch_idx % 5 == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.3f}",
                        "Acc": f"{100.*train_correct/train_total:.1f}%",
                    }
                )

        train_acc = 100.0 * train_correct / train_total
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
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.3f}",
                        "Acc": f"{100.*val_correct/val_total:.1f}%",
                    }
                )

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Store accuracies
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Results display
        print(f"\\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Overfitting analysis
        gap = abs(train_acc - val_acc)
        print(f"   📈 Train-Val Gap: {gap:.2f}%", end="")

        if gap < 10:
            print(" ✅ EXCELLENT!")
        elif gap < 20:
            print(" 👍 Good")
        elif gap < 30:
            print(" 🟡 OK")
        else:
            print(" ⚠️ Overfitting!")

        # Target achievement check
        if val_acc >= 60:
            print(f"   🎯🎉 STRETCH TARGET! {val_acc:.2f}% >= 60%")
        elif val_acc >= 50:
            print(f"   🎯✅ PRIMARY TARGET! {val_acc:.2f}% >= 50%")
        elif val_acc >= 40:
            print(f"   🎯👍 Good progress: {val_acc:.2f}% >= 40%")

        # Australian Shepherd tracking
        if "Australian_Shepherd_Dog" in breed_names:
            aus_idx = breed_names.index("Australian_Shepherd_Dog")
            if class_total[aus_idx] > 0:
                aus_acc = 100.0 * class_correct[aus_idx] / class_total[aus_idx]
                print(f"   ⭐ Australian Shepherd: {aus_acc:.1f}%", end="")

                if aus_acc >= 60:
                    print(" 🎉")
                elif aus_acc >= 50:
                    print(" ✅")
                elif aus_acc >= 40:
                    print(" 👍")
                else:
                    print(" ⚠️")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"   🏆 NEW BEST: {best_val_acc:.2f}% (epoch {best_epoch})")

        # Learning rate step
        scheduler.step()

        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\\n🛑 Early stopping after {epoch+1} epochs")
            print(f"   No improvement for {patience} epochs")
            break

    # Final per-class analysis
    print(f"\\n" + "=" * 60)
    print(f"📊 FINAL PER-CLASS ANALYSIS")
    print(f"=" * 60)

    final_class_accuracies = []
    for i, breed in enumerate(breed_names):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            samples = int(class_total[i])
            final_class_accuracies.append((breed, class_acc, samples))

    # Sort by accuracy
    final_class_accuracies.sort(key=lambda x: x[1], reverse=True)

    print(f"\\n🏆 RANKING FINALE (validation accuracy):")
    for i, (breed, acc, samples) in enumerate(final_class_accuracies, 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        target_marker = " ⭐" if breed == "Australian_Shepherd_Dog" else "  "
        print(
            f"   {status} {breed.replace('_', ' '):20s}: {acc:5.1f}% ({samples} samples){target_marker}"
        )

    # Australian Shepherd specific analysis
    if "Australian_Shepherd_Dog" in breed_names:
        aus_data = next(
            (breed, acc, samples)
            for breed, acc, samples in final_class_accuracies
            if breed == "Australian_Shepherd_Dog"
        )
        position = next(
            i
            for i, (breed, _, _) in enumerate(final_class_accuracies, 1)
            if breed == "Australian_Shepherd_Dog"
        )

        print(f"\\n⭐ AUSTRALIAN SHEPHERD ANALYSIS:")
        print(f"   Final accuracy: {aus_data[1]:.1f}%")
        print(f"   Ranking position: {position}/{len(breed_names)}")
        print(f"   Validation samples: {aus_data[2]}")

        if aus_data[1] >= 60:
            print(f"   Status: 🎉 STRETCH TARGET ACHIEVED!")
        elif aus_data[1] >= 50:
            print(f"   Status: ✅ PRIMARY TARGET ACHIEVED!")
        elif aus_data[1] >= 40:
            print(f"   Status: 👍 Good performance")
        else:
            print(f"   Status: ⚠️ Needs improvement")

    # Save model
    os.makedirs("outputs/top10_balanced", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "breed_names": breed_names,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_train_acc": train_accs[-1] if train_accs else 0,
            "final_val_acc": val_accs[-1] if val_accs else 0,
            "class_accuracies": {
                breed: acc for breed, acc, _ in final_class_accuracies
            },
            "training_history": {"train_accs": train_accs, "val_accs": val_accs},
            "config": {
                "model_type": "simple_balanced",
                "dropout_rate": 0.4,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "dataset_type": "top10_balanced_popular",
                "from_scratch": True,
            },
        },
        "outputs/top10_balanced/top10_balanced_model.pth",
    )

    # Final summary
    final_gap = abs(train_accs[-1] - val_accs[-1]) if train_accs and val_accs else 0

    print(f"\\n" + "=" * 60)
    print(f"🎉 TRAINING COMPLETED!")
    print(f"=" * 60)
    print(f"📁 Model saved: outputs/top10_balanced/top10_balanced_model.pth")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"📊 Final Results:")
    print(f"   Train: {train_accs[-1] if train_accs else 0:.2f}%")
    print(f"   Val: {val_accs[-1] if val_accs else 0:.2f}%")
    print(f"   Gap: {final_gap:.2f}%")
    print(f"🎯 Dataset: {num_classes} popular dog breeds")

    # Success evaluation
    if best_val_acc >= 60:
        print(f"\\n🎯🎉 STRETCH TARGET ACHIEVED! {best_val_acc:.2f}% >= 60%")
        print(f"✅ Excellent performance for TOP 10 balanced!")
        print(f"🚀 Ready for scaling to more breeds!")
        return True
    elif best_val_acc >= 50:
        print(f"\\n🎯✅ PRIMARY TARGET ACHIEVED! {best_val_acc:.2f}% >= 50%")
        print(f"👍 Good performance for TOP 10 balanced!")
        print(f"💡 Consider scaling to 15-20 breeds")
        return True
    elif best_val_acc >= 40:
        print(f"\\n👍 Decent performance: {best_val_acc:.2f}% >= 40%")
        print(f"💡 Try hyperparameter tuning")
        return False
    else:
        print(f"\\n⚠️  Performance below expectations: {best_val_acc:.2f}% < 40%")
        print(f"💡 Consider data augmentation or architecture changes")
        return False


if __name__ == "__main__":
    print("🎯 TOP 10 Balanced Training")
    print("=" * 30)

    success = top10_balanced_train()

    if success:
        print("\\n🚀 SUCCESS! Ready for next challenges!")
    else:
        print("\\n🔄 Further optimization recommended.")
