#!/usr/bin/env python3
"""
Quick training migliorato - Baseline con parametri originali che funzionavano
Obiettivo: Raggiungere 66.2% test accuracy e 60.9% Australian Shepherd
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_helper import ConfigHelper
from utils.dataloader import create_dataloaders_from_splits, get_transforms
from models.breed_classifier import create_breed_classifier
from utils.early_stopping import EarlyStopping
from utils.seed_utils import set_deterministic


def quick_train_improved():
    """Training migliorato con parametri baseline originali"""
    print("🚀 QUICK TRAINING MIGLIORATO - BASELINE ORIGINALE + TENSORBOARD")
    print("=" * 70)

    # Setup
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset path
    data_dir = "data/quick_splits"

    # TensorBoard setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/improved_training_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard logs: {log_dir}")
    print(f"   Avvia con: tensorboard --logdir=runs")

    # PARAMETRI ORIGINALI CHE FUNZIONAVANO (66.2% test accuracy)
    num_epochs = 12
    batch_size = 32
    learning_rate = 0.001  # ORIGINALE (non 0.0008)
    patience = 7
    dropout_rate = 0.3  # ORIGINALE (non 0.4)

    print(f"\n⚙️ PARAMETRI BASELINE ORIGINALI:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Patience: {patience}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Optimizer: Adam")
    print(f"   Scheduler: ReduceLROnPlateau")

    # DATA AUGMENTATION con RandomResizedCrop abilitato (evita distorsione)
    train_transform, val_transform = get_transforms(
        image_size=(224, 224),
        augmentation_config={
            "random_resized_crop": True,
            "rrc_scale": (0.85, 1.0),
            "rrc_ratio": (0.9, 1.1),
            "horizontal_flip": True,
            "rotation": 15,
            "brightness_contrast": [0.8, 1.2],
            "color_jitter": [0.1, 0.1, 0.1, 0.05],
        },
    )

    # Carica dataloaders
    print(f"\n📂 Caricando dataset da: {data_dir}")
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        image_size=(224, 224),
        augmentation_config={
            "random_resized_crop": True,
            "rrc_scale": (0.85, 1.0),
            "rrc_ratio": (0.9, 1.1),
            "horizontal_flip": True,
            "rotation": 15,
            "brightness_contrast": [0.8, 1.2],
            "color_jitter": [0.1, 0.1, 0.1, 0.05],
        },
        use_weighted_sampler=True,
    )

    # Verifica breed names
    train_dataset = train_loader.dataset
    breed_names = train_dataset.get_breed_names()
    num_classes = len(breed_names)

    print(f"\n📊 Dataset info:")
    print(f"   Classes: {breed_names}")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_loader.dataset)} samples")
    print(f"   Test: {len(test_loader.dataset)} samples")

    # Trova Australian Shepherd
    australian_idx = -1
    if "Australian_Shepherd_Dog" in breed_names:
        australian_idx = breed_names.index("Australian_Shepherd_Dog")
        print(f"   ⭐ Australian_Shepherd_Dog trovato (indice: {australian_idx})")
    else:
        print(f"   ⚠️ Australian_Shepherd_Dog NON trovato!")

    # MODELLO: abilita facilmente transfer learning via env (mantiene baseline by default)
    use_tl = bool(int(os.getenv("USE_TL", "0")))
    if use_tl:
        print("\n🧠 Using transfer learning backbone: ResNet18 (frozen)")
        model = create_breed_classifier(
            model_type="simple",  # ignored when pretrained_backbone is set
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained_backbone="resnet18",
            freeze_backbone=True,
        )
    else:
        model = create_breed_classifier(
            model_type="simple",
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    model = model.to(device)

    print(f"\n🔧 Modello SimpleBreedClassifier:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {total_params:,}")
    print(f"   Classi: {num_classes}")
    print(f"   Dropout: {dropout_rate}")

    # Log model info to TensorBoard
    writer.add_text(
        "Model/Info",
        f"""
    **Model Type**: SimpleBreedClassifier
    **Parameters**: {total_params:,}
    **Classes**: {num_classes}
    **Dropout**: {dropout_rate}
    **Breeds**: {', '.join(breed_names)}
    """,
    )

    # Log hyperparameters
    writer.add_hparams(
        {
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "patience": patience,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        },
        {},
    )

    # TRAINING SETUP ORIGINALE
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer (originale, non AdamW)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,  # 0.001 originale
        weight_decay=1e-4,  # Weight decay moderato
    )

    # ReduceLROnPlateau scheduler (originale)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    early_stopping = EarlyStopping(patience=patience)

    print(f"\n🎯 OBIETTIVI:")
    print(f"   Target Test Accuracy: 66.2%")
    print(f"   Target Australian Shepherd: 60.9%")
    print(f"   Evitare overfitting (train-val gap < 30%)")

    # Training tracking
    best_val_acc = 0.0
    best_epoch = 0
    train_accs = []
    val_accs = []

    print(f"\n🚀 STARTING TRAINING")
    print("=" * 60)

    # Training loop
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n📅 Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
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

            current_acc = 100.0 * train_correct / train_total
            pbar.set_postfix(
                {"Loss": f"{loss.item():.3f}", "Acc": f"{current_acc:.1f}%"}
            )

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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

                current_acc = 100.0 * val_correct / val_total
                pbar.set_postfix(
                    {"Loss": f"{loss.item():.3f}", "Acc": f"{current_acc:.1f}%"}
                )

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Store accuracies
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate train-val gap
        train_val_gap = train_acc - val_acc

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("Accuracy/Train_Val_Gap", train_val_gap, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Log train-val comparison
        writer.add_scalars(
            "Accuracy/Train_vs_Val", {"Train": train_acc, "Validation": val_acc}, epoch
        )

        writer.add_scalars(
            "Loss/Train_vs_Val",
            {"Train": avg_train_loss, "Validation": avg_val_loss},
            epoch,
        )

        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Train-Val Gap: {train_val_gap:.2f}%", end="")

        if train_val_gap > 30:
            print(" ⚠️ OVERFITTING!")
        elif train_val_gap > 20:
            print(" 🟡 Moderato")
        else:
            print(" ✅ OK")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"   🏆 NEW BEST: {val_acc:.2f}% (epoch {epoch+1})")

            # Save best model
            os.makedirs("outputs/improved", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "breed_names": breed_names,
                    "epoch": epoch + 1,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "hyperparameters": {
                        "learning_rate": learning_rate,
                        "dropout_rate": dropout_rate,
                        "batch_size": batch_size,
                        "optimizer": "Adam",
                        "scheduler": "ReduceLROnPlateau",
                    },
                },
                "outputs/improved/best_model.pth",
            )

        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\n🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            break

    # Final test evaluation
    print(f"\n📊 FINAL TEST EVALUATION")
    print("=" * 40)

    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = list(0.0 for i in range(num_classes))
    class_total = list(0.0 for i in range(num_classes))

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

            # Per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_acc = 100.0 * test_correct / test_total

    # Log final test results to TensorBoard
    writer.add_scalar("Final/Test_Accuracy", test_acc, 0)
    writer.add_scalar("Final/Best_Val_Accuracy", best_val_acc, 0)

    print(f"\n🎯 RISULTATI FINALI:")
    print(f"   Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Test Acc: {test_acc:.2f}%")

    # Per-class results
    print(f"\n📋 Per-Class Test Accuracy:")
    class_results = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            class_results.append(
                (breed_names[i], class_acc, int(class_correct[i]), int(class_total[i]))
            )

    # Sort by accuracy
    class_results.sort(key=lambda x: x[1], reverse=True)

    # Log per-class accuracy to TensorBoard
    for breed, acc, correct_count, total_count in class_results:
        writer.add_scalar(f"Per_Class_Test_Accuracy/{breed}", acc, 0)

    for i, (breed, acc, correct_count, total_count) in enumerate(class_results):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        star = " ⭐" if "Australian" in breed else ""
        print(
            f"   {medal} {breed:<25}: {acc:5.1f}% ({correct_count}/{total_count}){star}"
        )

    # Australian Shepherd specific
    if australian_idx >= 0 and class_total[australian_idx] > 0:
        australian_acc = (
            100.0 * class_correct[australian_idx] / class_total[australian_idx]
        )
        print(f"\n⭐ Australian Shepherd Performance: {australian_acc:.1f}%")

        # Log Australian Shepherd performance
        writer.add_scalar("Australian_Shepherd/Test_Accuracy", australian_acc, 0)
        writer.add_scalar(
            "Australian_Shepherd/Target_Reached",
            1.0 if australian_acc >= 60.9 else 0.0,
            0,
        )

        if australian_acc >= 60.9:
            print("   ✅ Target Australian Shepherd raggiunto!")
        else:
            print("   ⚠️ Target Australian Shepherd non raggiunto (60.9%)")

    # Overall assessment
    print(f"\n📊 CONFRONTO CON TARGET:")
    print(f"   Test Accuracy: {test_acc:.2f}% vs 66.2% (target)")
    if test_acc >= 66.2:
        print("   ✅ TARGET GENERALE RAGGIUNTO!")
    else:
        print(f"   ⚠️ Target non raggiunto ({66.2 - test_acc:.1f}% sotto)")

    print(f"\n💾 Modello salvato: outputs/improved/best_model.pth")
    print(f"📊 TensorBoard logs salvati in: {log_dir}")
    print(f"   Visualizza con: tensorboard --logdir=runs")

    # Close TensorBoard writer
    writer.close()

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": epoch + 1,
        "class_results": class_results,
        "australian_acc": australian_acc if australian_idx >= 0 else None,
        "target_reached": test_acc >= 66.2,
        "australian_target_reached": (
            australian_acc >= 60.9 if australian_idx >= 0 else False
        ),
    }


if __name__ == "__main__":
    results = quick_train_improved()
    if results:
        print(f"\n🎯 Training Results Summary:")
        print(f"   Test Accuracy: {results['test_acc']:.2f}%")
        if results["target_reached"]:
            print("   🎉 TARGET RAGGIUNTO!")
        else:
            print("   ⚠️ Serve ancora miglioramento")
