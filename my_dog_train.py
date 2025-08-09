#!/usr/bin/env python3
"""
Fase 2 - Training per identificazione binaria del mio cane Australian Shepherd
Classificazione binaria: il mio cane vs altri cani
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

from utils.dataloader import MyDogDataset, get_transforms
from models.breed_classifier import create_breed_classifier
from utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader


def my_dog_train():
    """Training per identificazione binaria del mio cane Australian Shepherd"""
    print("🐕 MY DOG BINARY CLASSIFICATION TRAINING")
    print("========================================")
    print("🎯 Il mio Australian Shepherd vs Altri cani")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Verifica dataset
    data_dir = "data/my_dog_vs_others"
    if not os.path.exists(data_dir):
        print(f"❌ Dataset non trovato: {data_dir}")
        print("📋 Struttura richiesta:")
        print("   data/my_dog_vs_others/")
        print("   ├── my_dog/           # Foto del tuo Australian Shepherd")
        print("   └── other_dogs/       # Foto di altri cani")
        print("\n💡 Crea questa struttura e riprova!")
        return

    # TensorBoard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = f"outputs/tensorboard/my_dog_{timestamp}"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)
    print(f"📊 TensorBoard logging: {tb_log_dir}")

    # Configurazione
    num_epochs = 20
    batch_size = 16  # Più piccolo per dataset personale
    learning_rate = 0.0005
    patience = 5
    dropout_rate = 0.3  # Meno aggressivo per binario

    print(f"\n⚙️ CONFIGURAZIONE BINARIA:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Patience: {patience}")
    print(f"   Dropout: {dropout_rate}")

    # Transforms
    train_transform, val_transform = get_transforms(
        image_size=(224, 224),
        augmentation_config={
            "horizontal_flip": True,
            "rotation": 10,  # Meno aggressivo
            "brightness_contrast": [0.9, 1.1],
            "color_jitter": [0.05, 0.05, 0.0, 0.0],
        },
    )

    # Dataset
    print(f"\n📂 Caricando dataset binario...")
    full_dataset = MyDogDataset(data_dir, transform=train_transform)

    if len(full_dataset) == 0:
        print("❌ Dataset vuoto! Aggiungi immagini in my_dog/ e other_dogs/")
        return

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size]
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )

    # Update transforms for validation/test
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"📊 Dataset split:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Modello binario (2 classi)
    model = create_breed_classifier(
        model_type="simple",
        num_classes=2,  # Binario: mio cane vs altri
        dropout_rate=dropout_rate,
    )
    model = model.to(device)

    print(f"\n🔧 Modello binario:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {total_params:,}")
    print(f"   Classi: 2 (il mio cane vs altri)")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(patience=patience)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n🚀 STARTING BINARY TRAINING")
    print("=" * 50)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # Logging
        writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch + 1)
        writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch + 1)
        writer.add_scalar("Epoch/Val_Loss", avg_val_loss, epoch + 1)
        writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch + 1)
        writer.add_scalar("Epoch/Learning_Rate", current_lr, epoch + 1)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"   🏆 NEW BEST: {val_acc:.2f}% (epoch {epoch+1})")

            # Save best model
            os.makedirs("outputs/my_dog", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 2,
                    "epoch": epoch + 1,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                },
                "outputs/my_dog/best_model.pth",
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

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

    test_acc = 100.0 * test_correct / test_total

    print(f"🎯 FINAL RESULTS:")
    print(f"   Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Test Acc: {test_acc:.2f}%")
    print(f"   Model saved: outputs/my_dog/best_model.pth")
    print(f"   TensorBoard: {tb_log_dir}")

    writer.close()

    if test_acc >= 85:
        print(f"\n🎉 EXCELLENT! Il modello riconosce bene il tuo cane!")
    elif test_acc >= 70:
        print(f"\n✅ GOOD! Performance accettabile, considera più dati di training")
    else:
        print(f"\n⚠️ NEEDS IMPROVEMENT! Aggiungi più immagini diverse per il training")

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": epoch + 1,
        "tensorboard_dir": tb_log_dir,
    }


if __name__ == "__main__":
    results = my_dog_train()
    if results:
        print(f"\n🎯 Training Results: {results}")
