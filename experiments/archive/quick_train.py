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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Definisco data_dir per il quick test con Australian Shepherd
    data_dir = "data/quick_splits"  # Usa il dataset con splits separati incluso Australian Shepherd

    # Configurazione intermedia (migliorata)
    num_epochs = int(os.getenv("EPOCHS", "20"))  # override con EPOCHS se impostata
    batch_size = 32
    learning_rate = 0.001  # LR leggermente più alto (con AdamW)
    patience = 3  # Early stopping più reattivo

    # Carica dataloaders da splits preorganizzati
    print("Caricando dataset da splits preorganizzati...")
    cfg = ConfigHelper()
    augmentation_config = cfg.get_augmentation_config()
    # Ordine desiderato: 1 australian, 2 Norwich_terrier, 3 Japanese_spaniel, 4 husky, 5 ciquaqua
    desired_breeds = [
        "Australian_Shepherd_Dog",
        "Norwich_terrier",
        "Japanese_spaniel",
        "Siberian_husky",  # husky
        "Chihuahua",  # ciquaqua
    ]
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        image_size=(224, 224),
        augmentation_config=augmentation_config,
        allowed_breeds=desired_breeds,
        use_weighted_sampler=True,
    )

    # Verifica breed names dalla struttura del train set
    train_dataset = train_loader.dataset
    breed_names = train_dataset.get_breed_names()
    print(f"🎯 Razze nel dataset: {breed_names}")
    # Avvisa se l'ordine non coincide con quello richiesto
    if [b.lower() for b in breed_names] != [b.lower() for b in desired_breeds]:
        print(
            "⚠️  L'ordine effettivo dei breed non coincide perfettamente con quello richiesto."
        )
        print(f"   Richiesto: {desired_breeds}")

    if "Australian_Shepherd_Dog" in breed_names:
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
        model_type="simple",
        num_classes=num_classes,
        dropout_rate=0.4,  # Dropout intermedio
    )
    model = model.to(device)

    # Calcolo class weights per gestire sbilanciamento
    try:
        import numpy as np

        labels_np = np.array(train_loader.dataset.labels)
        num_classes = len(breed_names)
        counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        class_weights = counts.max() / counts
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
    except Exception:
        class_weights_tensor = None

    # Loss con label smoothing (fallback automatico) + class weights
    try:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor, label_smoothing=0.05
        )
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer e scheduler migliorati
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(patience=patience)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {num_epochs} epochs...")

    # Initialize accuracy variables for final reporting
    train_acc = 0.0
    val_acc = 0.0

    # Tracking best
    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = None

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100.*train_correct/train_total:.2f}%",
                }
            )

        train_acc = 100.0 * train_correct / train_total
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

                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.*val_correct/val_total:.2f}%",
                    }
                )

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Scheduler step e best tracking
        scheduler.step(avg_val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            print(f"   🏆 NEW BEST: {best_val_acc:.2f}% (epoch {best_epoch})")

        if early_stopping(avg_val_loss):
            print(f"🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            break

    # Salva modello
    os.makedirs("outputs/quick_splits", exist_ok=True)
    # Salva il best model (se presente), altrimenti l'ultimo
    state_to_save = (
        best_state_dict if best_state_dict is not None else model.state_dict()
    )
    torch.save(
        {
            "model_state_dict": state_to_save,
            "num_classes": num_classes,
            "breed_names": breed_names,
            "epoch": best_epoch if best_epoch > 0 else epoch + 1,
            "train_acc": train_acc,
            "val_acc": best_val_acc if best_epoch > 0 else val_acc,
        },
        "outputs/quick_splits/quick_model.pth",
    )

    print(f"\n✅ Training completato!")
    print(f"📁 Modello salvato in: outputs/quick_splits/quick_model.pth")
    final_val_to_report = best_val_acc if best_epoch > 0 else val_acc
    print(
        f"🎯 Accuracy finale: Train {train_acc:.2f}%, Val {final_val_to_report:.2f}% (best epoch {best_epoch if best_epoch>0 else epoch+1})"
    )


if __name__ == "__main__":
    quick_train()
