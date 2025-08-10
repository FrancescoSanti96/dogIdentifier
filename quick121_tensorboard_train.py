#!/usr/bin/env python3
"""
Training completo 121 razze con TensorBoard

Caratteristiche:
- Transfer Learning ResNet18 (USE_TL=1) con opzione di sblocco layer4
- Data augmentation con RandomResizedCrop, sampler pesato
- Label smoothing, AdamW, ReduceLROnPlateau, gradient clipping
- Logging anche Top-5 Accuracy
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_helper import ConfigHelper
from utils.dataloader import create_dataloaders_from_splits
from models.breed_classifier import create_breed_classifier
from utils.early_stopping import EarlyStopping
from utils.seed_utils import set_deterministic


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
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


def quick121_tensorboard_train():
    print("🚀 FULL 121 BREEDS + TENSORBOARD")
    print("=================================")
    print("📊 Training con 121 razze e monitoring TensorBoard")

    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = f"outputs/tensorboard/quick121_{timestamp}"
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(tb_logdir)
    print(f"📊 TensorBoard logging: {tb_logdir}")
    print(f"   🌐 Avvia TensorBoard: tensorboard --logdir outputs/tensorboard")
    print(f"   🔗 URL: http://localhost:6006")

    # Config
    data_dir = os.getenv("SPLITS_DIR", "data/full121_balanced")
    num_epochs = int(os.getenv("EPOCHS", "45"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    learning_rate = float(os.getenv("LR", "0.0008"))
    patience = int(os.getenv("PATIENCE", "10"))
    dropout_rate = float(os.getenv("DROPOUT", "0.4"))
    weight_decay = float(os.getenv("WD", "5e-4"))

    print(f"\n⚡ CONFIGURAZIONE:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Weight decay: {weight_decay}")

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

    use_tl = bool(int(os.getenv("USE_TL", "1")))
    if use_tl:
        print("\n🧠 Using transfer learning backbone: ResNet18 (frozen)")
        model = create_breed_classifier(
            model_type="simple",
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained_backbone="resnet18",
            freeze_backbone=True,
        )
    else:
        model = create_breed_classifier(
            model_type="simple", num_classes=num_classes, dropout_rate=dropout_rate
        )
    model = model.to(device)

    if bool(int(os.getenv("UNFREEZE_LAYER4", "0"))):
        print("🔓 Unfreezing layer4 for fine-tuning")
        for name, param in model.named_parameters():
            if name.startswith("layer4.") or name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    print(f"\n🔧 Modello configurato:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametri totali: {total_params:,}")
    print(f"   Parametri trainable: {trainable_params:,}")

    try:
        labels_np = np.array(train_loader.dataset.labels)
        counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        class_weights = counts.max() / counts
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
        print(f"⚖️ Class weights: {class_weights.round(2).tolist()}")
    except Exception:
        class_weights_tensor = None

    try:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor, label_smoothing=0.05
        )
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(patience=patience)

    hparams = {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "dataset": "121_breeds",
        "label_smoothing": 0.05,
        "augmentation": True,
        "log_top5": True,
    }
    writer.add_hparams(hparams, {"hparam/accuracy": 0, "hparam/loss": 0})

    print("\n" + "=" * 60)
    print("🚀 STARTING FULL 121 BREEDS TRAINING")
    print("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n📅 Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f}")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_top5_sum = 0.0
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for data, target in pbar:
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
            train_top5_sum += topk_accuracy(output.detach(), target, topk=(5,))[0]
        train_acc = 100.0 * train_correct / train_total
        train_top5 = train_top5_sum / len(train_loader)
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_top5_sum = 0.0
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
                val_top5_sum += topk_accuracy(output, target, topk=(5,))[0]
        val_acc = 100.0 * val_correct / val_total
        val_top5 = val_top5_sum / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch + 1)
        writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch + 1)
        writer.add_scalar("Epoch/Train_Top5", train_top5, epoch + 1)
        writer.add_scalar("Epoch/Val_Loss", avg_val_loss, epoch + 1)
        writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch + 1)
        writer.add_scalar("Epoch/Val_Top5", val_top5, epoch + 1)
        writer.add_scalar("Epoch/Learning_Rate", new_lr, epoch + 1)
        writer.add_scalar("Analysis/Train_Val_Gap", train_acc - val_acc, epoch + 1)

        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5:.2f}%")
        print(f"   Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.2f}% | Val   Top-5: {val_top5:.2f}%")
        print(f"   Current LR: {new_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"   🏆 NEW BEST: {val_acc:.2f}% (epoch {epoch+1})")
            os.makedirs("outputs/top121", exist_ok=True)
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
                "outputs/top121/best_model.pth",
            )

        if early_stopping(avg_val_loss):
            print(f"\n🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            writer.add_text(
                "Training/Early_Stop", f"Stopped at epoch {epoch+1}", epoch + 1
            )
            break

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
        "outputs/top121/final_model.pth",
    )

    print(f"\n📈 Final Results:")
    print(f"   Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Final Val Acc: {val_acc:.2f}%")
    print(f"   TensorBoard: {tb_logdir}")

    writer.close()

    return {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "epochs": epoch + 1,
        "tensorboard_dir": tb_logdir,
    }


if __name__ == "__main__":
    results = quick121_tensorboard_train()
    print(f"\n🎯 Training Results: {results}")


