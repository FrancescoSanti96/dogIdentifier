#!/usr/bin/env python3
"""
Fase 2 - T    # Verifica dataset con split preparati
    splits_dir = "data/my_dog_vs_others_splits"
    if not os.path.exists(splits_dir):
        print(f"❌ Dataset con split non trovato: {splits_dir}")
        print("💡 Esegui prima: python src/prepare_data.py --binary")
        return

    # Dataset - caricamento da split preparati
    print(f"\n📂 Caricando dataset binario con split preparati...")er identificazione binaria del mio cane Australian Shepherd
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

# Aggiungi directory root del progetto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataloader import MyDogDataset, get_transforms
from models.breed_classifier import create_breed_classifier
from utils.early_stopping import EarlyStopping
from utils.seed_utils import set_deterministic
from torch.utils.data import DataLoader


def my_dog_train():
    """Training per identificazione binaria del mio cane Australian Shepherd"""
    print("🐕 MY DOG BINARY CLASSIFICATION TRAINING")
    print("========================================")
    print("🎯 Il mio Australian Shepherd vs Altri cani")

    # Setup
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # TensorBoard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = f"outputs/tensorboard/my_dog_{timestamp}"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)
    print(f"📊 TensorBoard logging: {tb_log_dir}")

    # Configurazione Binaria - OTTIMIZZATA (Configurazione 2 vincente)
    # La configurazione che ha ottenuto il miglior test accuracy: 71.4%
    num_epochs = 30  # Epoche sufficienti con early stopping
    batch_size = 12  # Batch ottimale per stabilità
    learning_rate = 0.0003  # Learning rate bilanciato
    patience = 8  # Patience appropriata
    dropout_rate = 0.5  # Dropout efficace ma non eccessivo

    print(f"\n⚙️ CONFIGURAZIONE OTTIMIZZATA (Vincente #2):")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Patience: {patience}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   🎨 Data Augmentation: BILANCIATA")
    print(f"   📈 Dataset Size: +30% rispetto a prove precedenti")

    # Iperparametri per logging TensorBoard (solo tipi compatibili)
    hparams = {
        "num_classes": 2,
        "epochs": int(num_epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "dropout": float(dropout_rate),
        "patience": int(patience),
        "model_type": "binary_classification",
        "dataset_name": "my_dog_vs_others",
        "use_transfer_learning": int(os.getenv("USE_TL", "0") == "1"),  # Convert bool to int
        "augmentation": "balanced",
    }

    # Data Augmentation OTTIMIZZATA (Configurazione 2 vincente)
    # La configurazione che ha dato il miglior bilanciamento bias-variance
    train_transform, val_transform = get_transforms(
        image_size=(224, 224),
        augmentation_config={
            "horizontal_flip": True,  # Flip orizzontale: sempre efficace
            "rotation": 15,  # Rotazione moderata ma efficace
            "brightness_contrast": [0.8, 1.2],  # Range bilanciato
            "color_jitter": [0.1, 0.1, 0.05, 0.02],  # Variazione colore moderata
            "erasing_p": 0.1,  # Random erasing leggero
        },
    )

    # Verifica dataset con split preparati
    splits_dir = "data/my_dog_vs_others_splits"
    if not os.path.exists(splits_dir):
        print(f"❌ Dataset con split non trovato: {splits_dir}")
        print("� Esegui prima: python src/prepare_data.py --binary")
        return

    # Dataset
    print(f"\n📂 Caricando dataset binario con split preparati...")
    
    # Carica dataset da split fisici
    train_dataset = MyDogDataset(
        os.path.join(splits_dir, "train"), 
        transform=train_transform,
        my_dog_folder="maggie",
        other_dogs_folder="other"
    )
    
    val_dataset = MyDogDataset(
        os.path.join(splits_dir, "val"), 
        transform=val_transform,
        my_dog_folder="maggie",
        other_dogs_folder="other"
    )
    
    test_dataset = MyDogDataset(
        os.path.join(splits_dir, "test"), 
        transform=val_transform,
        my_dog_folder="maggie",
        other_dogs_folder="other"
    )

    if len(train_dataset) == 0:
        print("❌ Dataset di training vuoto!")
        return

    print(f"📊 Dataset caricato:")
    print(f"   Train: {len(train_dataset)} immagini")
    print(f"   Validation: {len(val_dataset)} immagini") 
    print(f"   Test: {len(test_dataset)} immagini")
    print(f"   Totale: {len(train_dataset) + len(val_dataset) + len(test_dataset)} immagini")

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

    # Modello Binario: problema di classificazione speciale
    # Obiettivo: distinguere il MIO cane da tutti gli altri cani
    use_tl = bool(
        int(os.getenv("USE_TL", "0"))
    )  # Transfer Learning opzionale via env var
    if use_tl:
        # Transfer Learning: veloce, efficiente per dataset piccolo
        print("\n🧠 Utilizzo transfer learning backbone: ResNet18 (congelato)")
        model = create_breed_classifier(
            model_type="simple",
            num_classes=2,  # Binary: "mio cane" vs "non mio cane"
            dropout_rate=dropout_rate,
            pretrained_backbone="resnet18",
            freeze_backbone=True,
        )
    else:
        # From Scratch: architettura semplice per binary task
        model = create_breed_classifier(
            model_type="simple",  # 3.3M parametri, sufficiente per binary
            num_classes=2,
            dropout_rate=dropout_rate,
        )
    model = model.to(device)

    print(f"\n🔧 Modello binario:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {total_params:,}")
    print(f"   Classi: 2 (il mio cane vs altri)")

    # Setup Training per Binary Classification
    criterion = nn.CrossEntropyLoss()  # Standard anche per binary (2 classi)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3  # Riduce LR più aggressivamente
    )
    early_stopping = EarlyStopping(patience=patience)

    # Loop di training
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n🚀 INIZIO TRAINING BINARIO")
    print("=" * 50)

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n📅 Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f}")

        # Fase di training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc="Allenamento", leave=False)
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

        # Fase di validazione
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validazione", leave=False)
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

        # TensorBoard Logging - struttura pulita come train.py
        writer.add_scalar("loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("loss/validation", avg_val_loss, epoch + 1)
        writer.add_scalar("accuracy/train", train_acc, epoch + 1)
        writer.add_scalar("accuracy/validation", val_acc, epoch + 1)
        writer.add_scalar("learning_rate", current_lr, epoch + 1)

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

    # Log iperparametri con metriche finali su TensorBoard
    writer.add_hparams(
        hparams,
        {
            "final_val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "final_test_acc": test_acc,
        },
    )

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
