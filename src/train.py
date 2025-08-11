#!/usr/bin/env python3
"""
Training unificato per classificazione razze canine con TensorBoard

Supporta training progressivo da 5 a 121 razze con configurazione automatica.

Usage:
    python src/train.py --breeds 5    # 5 razze baseline
    python src/train.py --breeds 10   # 10 razze
    python src/train.py --breeds 30   # 30 razze
    python src/train.py --breeds 60   # 60 razze
    python src/train.py --breeds 90   # 90 razze
    python src/train.py --breeds 121  # 121 razze complete

Environment variables:
    USE_TL=1          # Transfer Learning (default: 1)
    EPOCHS=45         # Number of epochs (default: auto per breeds)
    BATCH_SIZE=32     # Batch size (default: 32)
    LR=0.0008         # Learning rate (default: auto per breeds)
    PATIENCE=10       # Early stopping patience (default: auto per breeds)
    DROPOUT=0.4       # Dropout rate (default: 0.4)
    WD=5e-4           # Weight decay (default: 5e-4)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_helper import ConfigHelper
from utils.dataloader import create_dataloaders_from_splits
from models.breed_classifier import create_breed_classifier
from utils.early_stopping import EarlyStopping
from utils.seed_utils import set_deterministic


# Configurazioni ottimali per ogni scala di razze
BREED_CONFIGS = {
    5: {
        "data_dir": "data/breeds_5",
        "epochs": 6,
        "lr": 0.0008,
        "patience": 3,
        "description": "5 breeds baseline",
    },
    10: {
        "data_dir": "data/top10_balanced",
        "epochs": 15,
        "lr": 0.0008,
        "patience": 6,
        "description": "10 breeds balanced",
    },
    30: {
        "data_dir": "data/top30_balanced",
        "epochs": 20,
        "lr": 0.0008,
        "patience": 8,
        "description": "30 breeds balanced",
    },
    60: {
        "data_dir": "data/top60_balanced",
        "epochs": 30,
        "lr": 0.0008,
        "patience": 8,
        "description": "60 breeds balanced",
    },
    90: {
        "data_dir": "data/top90_balanced",
        "epochs": 30,
        "lr": 0.0008,
        "patience": 8,
        "description": "90 breeds balanced",
    },
    121: {
        "data_dir": "data/full121_balanced",
        "epochs": 45,
        "lr": 0.0008,
        "patience": 10,
        "description": "121 breeds complete",
    },
}


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Calculate top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    return res


def train_breeds(num_breeds: int):
    """
    Training unificato per il numero specificato di razze

    Args:
        num_breeds: Numero di razze da addestrare (5, 10, 30, 60, 90, 121)
    """
    if num_breeds not in BREED_CONFIGS:
        raise ValueError(
            f"Numero razze {num_breeds} non supportato. "
            f"Supportati: {list(BREED_CONFIGS.keys())}"
        )

    config = BREED_CONFIGS[num_breeds]

    print(f"🚀 TRAINING {num_breeds} BREEDS + TENSORBOARD")
    print("=" * 50)
    print(f"📊 Training {config['description']}")

    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = (
        f"outputs/tensorboard/breeds_{num_breeds}/breeds_{num_breeds}_{timestamp}"
    )
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(tb_logdir)
    print(f"📊 TensorBoard logging: {tb_logdir}")
    print(f"   🌐 Avvia TensorBoard: python scripts/launch_tensorboard.py")
    print(f"   🔗 URL: http://localhost:6006")

    # Configuration (with environment overrides)
    data_dir = os.getenv("SPLITS_DIR", config["data_dir"])
    num_epochs = int(os.getenv("EPOCHS", str(config["epochs"])))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    learning_rate = float(os.getenv("LR", str(config["lr"])))
    patience = int(os.getenv("PATIENCE", str(config["patience"])))
    dropout_rate = float(os.getenv("DROPOUT", "0.4"))
    weight_decay = float(os.getenv("WD", "5e-4"))

    print(f"\n⚡ CONFIGURAZIONE:")
    print(f"   Dataset: {data_dir}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Weight decay: {weight_decay}")

    # Data loading
    cfg = ConfigHelper()
    augmentation_config = cfg.get_augmentation_config() or {}
    augmentation_config.setdefault("random_resized_crop", True)
    augmentation_config.setdefault("rrc_scale", (0.85, 1.0))
    augmentation_config.setdefault("rrc_ratio", (0.9, 1.1))

    print(f"\n📂 Caricando dataset da: {data_dir}")
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        image_size=(224, 224),
        augmentation_config=augmentation_config,
        use_weighted_sampler=True,
    )

    breed_names = train_loader.dataset.get_breed_names()
    num_classes = len(breed_names)
    print(f"🎯 Breeds nel dataset: {num_classes}")

    # Model creation
    use_tl = bool(int(os.getenv("USE_TL", "1")))
    if use_tl:
        print("\n🧠 Using transfer learning backbone: ResNet18 (frozen)")
    else:
        print("\n🧠 Training from scratch")

    model = create_breed_classifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_transfer_learning=use_tl,
        freeze_backbone=True,
    )
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=3, verbose=True
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    # Hyperparameters for logging
    hparams = {
        "num_breeds": num_breeds,
        "num_classes": num_classes,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout_rate,
        "weight_decay": weight_decay,
        "use_transfer_learning": use_tl,
        "patience": patience,
        "dataset": data_dir,
    }

    print(f"\n🏋️ Inizio training...")
    print(f"   Parametri modello: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"   Parametri trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    best_val_acc = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Train {epoch+1}")
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            train_bar.set_postfix(
                {
                    "Loss": f"{running_loss/(batch_idx+1):.4f}",
                    "Acc": f"{100.*correct/total:.2f}%",
                }
            )

        train_acc = 100.0 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_top5_correct = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Val {epoch+1}")
            for data, target in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

                # Top-5 accuracy (if applicable)
                if num_classes >= 5:
                    top1, top5 = topk_accuracy(output, target, topk=(1, 5))
                    val_top5_correct += (top5 * target.size(0)) / 100

                val_bar.set_postfix(
                    {
                        "Loss": f"{val_loss/(len(val_bar.iterable)):.4f}",
                        "Acc": f"{100.*val_correct/val_total:.2f}%",
                    }
                )

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_top5_acc = 100.0 * val_top5_correct / val_total if num_classes >= 5 else 0

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        writer.add_scalars(
            "Loss", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch + 1
        )
        writer.add_scalars(
            "Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch + 1
        )
        if num_classes >= 5:
            writer.add_scalar("Top5_Accuracy/Validation", val_top5_acc, epoch + 1)
        writer.add_scalar("Learning_Rate", current_lr, epoch + 1)

        print(f"   Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(
            f"   Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%"
            + (f", Top-5: {val_top5_acc:.2f}%" if num_classes >= 5 else "")
        )
        print(f"   Current LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"   🏆 NEW BEST: {val_acc:.2f}% (epoch {epoch+1})")

            os.makedirs(f"outputs/models/breeds_{num_breeds}", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "breed_names": breed_names,
                    "epoch": epoch + 1,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "hyperparameters": hparams,
                },
                f"outputs/models/breeds_{num_breeds}/best_model.pth",
            )

        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\n🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            writer.add_text(
                "Training/Early_Stop", f"Stopped at epoch {epoch+1}", epoch + 1
            )
            break

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "breed_names": breed_names,
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "hyperparameters": hparams,
        },
        f"outputs/models/breeds_{num_breeds}/final_model.pth",
    )

    # Log hyperparameters with final metrics
    writer.add_hparams(
        hparams,
        {
            "final_val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "final_train_acc": train_acc,
        },
    )

    print(f"\n📈 Final Results:")
    print(f"   Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Final Val Acc: {val_acc:.2f}%")
    print(f"   Models saved: outputs/models/breeds_{num_breeds}/")
    print(f"   TensorBoard: {tb_logdir}")

    writer.close()

    return {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "epochs": epoch + 1,
        "tensorboard_dir": tb_logdir,
        "num_breeds": num_breeds,
    }


def main():
    parser = argparse.ArgumentParser(description="Training unificato razze canine")
    parser.add_argument(
        "--breeds",
        type=int,
        required=True,
        choices=[5, 10, 30, 60, 90, 121],
        help="Numero di razze da addestrare",
    )

    args = parser.parse_args()

    try:
        results = train_breeds(args.breeds)
        print(f"\n✅ Training completato con successo!")
        print(f"   Breeds: {results['num_breeds']}")
        print(f"   Best accuracy: {results['best_val_acc']:.2f}%")

    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
