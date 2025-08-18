#!/usr/bin/env python3
"""
Training unificato per classificazione razze canine con TensorBoard

Questo script implementa il cuore del progetto Dog Breed Identifier, con supporto
per training progressivo scalabile da 5 a 121 razze canine.

CARATTERISTICHE PRINCIPALI:
-  CNN from-scratch personalizzata (134M parametri)
-  Transfer Learning opzionale (ResNet18) - PER CONFRONTO SCIENTIFICO
-  Scaling progressivo validato: 5→10→30→60→90→121 razze
-  Configurazioni ottimizzate per ogni scala
-  TensorBoard logging completo con hyperparameters
-  Focus Australian Shepherd (obiettivo personale)
-  Riproducibilità garantita (seed deterministico)

ARCHITETTURE SUPPORTATE:
1. FROM SCRATCH - BreedClassifier (full): 134M parametri, VGG-like personalizzata
2. FROM SCRATCH - SimpleBreedClassifier: 3.3M parametri, per test rapidi
3. TRANSFER LEARNING - ResNet18: backbone congelato, solo classificatore trainable

METODOLOGIA SCIENTIFICA:
- Confronto rigoroso FROM SCRATCH vs TRANSFER LEARNING
- Early stopping per prevenire overfitting
- Data augmentation con RandomResizedCrop + WeightedSampler
- Label smoothing e gradient clipping per stabilità
- Metriche multiple: Accuracy, Top-5, per-class analysis

Usage:
    # Training from scratch (quello che vuole il professore)
    python src/train.py --breeds 30

    # Transfer learning (per confronto scientifico)
    USE_TL=1 python src/train.py --breeds 30

    # Architettura completa 134M parametri
    MODEL_TYPE=full USE_TL=0 python src/train.py --breeds 121

    # Test rapido con architettura semplice
    MODEL_TYPE=simple python src/train.py --breeds 5

Variabili d'ambiente (per switching rapido):
    USE_TL=1          # Transfer Learning (predefinito: 1 per efficienza)
    MODEL_TYPE=full   # Architettura modello: 'full' o 'simple' (predefinito: full)
    EPOCHS=45         # Numero di epoche (predefinito: automatico per scala)
    BATCH_SIZE=32     # Dimensione batch (predefinito: 32)
    LR=0.0008         # Learning rate (predefinito: automatico per scala)
    PATIENCE=10       # Pazienza early stopping (predefinito: automatico per scala)
    DROPOUT=0.4       # Tasso di dropout (predefinito: 0.4)
    WD=5e-4           # Weight decay (predefinito: 5e-4)

Output:
    - outputs/models/breeds_{N}/best_model.pth: Miglior modello
    - outputs/tensorboard/breeds_{N}/: Log TensorBoard
    - Iperparametri e metriche salvati nel checkpoint
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
        "description": "5 razze baseline",
    },
    10: {
        "data_dir": "data/top10_balanced",
        "epochs": 15,
        "lr": 0.0008,
        "patience": 6,
        "description": "10 razze bilanciate",
    },
    30: {
        "data_dir": "data/top30_balanced",
        "epochs": 20,
        "lr": 0.0008,
        "patience": 8,
        "description": "30 razze bilanciate",
    },
    60: {
        "data_dir": "data/top60_balanced",
        "epochs": 30,
        "lr": 0.0008,
        "patience": 8,
        "description": "60 razze bilanciate",
    },
    90: {
        "data_dir": "data/top90_balanced",
        "epochs": 30,
        "lr": 0.0008,
        "patience": 8,
        "description": "90 razze bilanciate",
    },
    121: {
        "data_dir": "data/full121_balanced",
        "epochs": 45,
        "lr": 0.0008,
        "patience": 10,
        "description": "121 razze complete",
    },
}


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Calcola accuratezza top-k"""
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


def train_breeds(
    num_breeds: int,
    config_path: str = "config.json",
    profile: str | None = None,
    cli_overrides: dict | None = None,
):
    """
    Training unificato per il numero specificato di razze

    Questa funzione implementa il training progressivo scalabile da 5 a 121 razze
    con configurazioni ottimizzate per ogni scala. Supporta sia training from-scratch
    che transfer learning tramite variabili d'ambiente.

    Args:
        num_breeds: Numero di razze da addestrare (5, 10, 30, 60, 90, 121)
        config_path: Percorso al file di configurazione JSON
        profile: Nome del profilo di configurazione da utilizzare
        cli_overrides: Override dei parametri da command line

    Examples:
        >>> # Training from scratch 30 razze
        >>> train_breeds(30)
        >>> # Transfer learning 121 razze
        >>> os.environ['USE_TL'] = '1'
        >>> train_breeds(121)
    """
    if num_breeds not in BREED_CONFIGS:
        raise ValueError(
            f"Numero razze {num_breeds} non supportato. "
            f"Supportati: {list(BREED_CONFIGS.keys())}"
        )

    # Valori predefiniti di base dalla mappatura hardcoded
    base_defaults = BREED_CONFIGS[num_breeds].copy()
    # Aggiungi defaults globali non presenti in BREED_CONFIGS
    base_defaults.setdefault("batch_size", 32)
    base_defaults.setdefault("dropout", 0.4)
    base_defaults.setdefault("weight_decay", 5e-4)
    base_defaults.setdefault("use_tl", 1)

    # Applica profilo se specificato
    if profile:
        config = ConfigHelper(config_path)
        if config.apply_profile(profile):
            print(f"✅ Profilo '{profile}' applicato")
        else:
            print(f"⚠️ Profilo '{profile}' non trovato")
            available = config.get_profile_names()
            if available:
                print(f"   Profili disponibili: {', '.join(available)}")

    merged_defaults = base_defaults

    print(f"🚀 TRAINING {num_breeds} BREEDS + TENSORBOARD")
    print("=" * 50)
    print(f"📊 Training {BREED_CONFIGS[num_breeds]['description']}")

    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configurazione TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = (
        f"outputs/tensorboard/breeds_{num_breeds}/breeds_{num_breeds}_{timestamp}"
    )
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(tb_logdir)
    print(f"📊 TensorBoard logging: {tb_logdir}")
    print(f"   🌐 Avvia TensorBoard: python scripts/launch_tensorboard.py")
    print(f"   🔗 URL: http://localhost:6006")

    # Precedenza configurazione: CLI > ENV > profile > defaults
    # Questo sistema a cascata permette massima flessibilità:
    # 1. Defaults hardcoded per ogni scala (BREED_CONFIGS)
    # 2. Profile da config.json (opzionale)
    # 3. Environment variables (per switching rapido)
    # 4. CLI arguments (per override specifici)
    data_dir = merged_defaults["data_dir"]
    num_epochs = merged_defaults["epochs"]
    learning_rate = merged_defaults["lr"]
    patience = merged_defaults["patience"]
    batch_size = merged_defaults["batch_size"]
    dropout_rate = merged_defaults["dropout"]
    weight_decay = merged_defaults["weight_decay"]
    use_tl_default = merged_defaults["use_tl"]

    # Applica override da variabili d'ambiente
    data_dir = os.getenv("SPLITS_DIR", data_dir)
    num_epochs = int(os.getenv("EPOCHS", str(num_epochs)))
    batch_size = int(os.getenv("BATCH_SIZE", str(batch_size)))
    learning_rate = float(os.getenv("LR", str(learning_rate)))
    patience = int(os.getenv("PATIENCE", str(patience)))
    dropout_rate = float(os.getenv("DROPOUT", str(dropout_rate)))
    weight_decay = float(os.getenv("WD", str(weight_decay)))
    use_tl = int(os.getenv("USE_TL", str(use_tl_default)))
    model_type = os.getenv("MODEL_TYPE", "full").lower()  # 'full' o 'simple'

    # Applica override CLI per ultimi
    if cli_overrides:
        if cli_overrides.get("data_dir"):
            data_dir = cli_overrides["data_dir"]
        if cli_overrides.get("epochs") is not None:
            num_epochs = int(cli_overrides["epochs"])
        if cli_overrides.get("batch_size") is not None:
            batch_size = int(cli_overrides["batch_size"])
        if cli_overrides.get("lr") is not None:
            learning_rate = float(cli_overrides["lr"])
        if cli_overrides.get("patience") is not None:
            patience = int(cli_overrides["patience"])
        if cli_overrides.get("dropout") is not None:
            dropout_rate = float(cli_overrides["dropout"])
        if cli_overrides.get("weight_decay") is not None:
            weight_decay = float(cli_overrides["weight_decay"])
        if cli_overrides.get("use_tl") is not None:
            use_tl = int(cli_overrides["use_tl"])
        if cli_overrides.get("model_type") is not None:
            model_type = cli_overrides["model_type"].lower()

    print(f"\n⚡ CONFIGURAZIONE:")
    print(f"   Dataset: {data_dir}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Weight decay: {weight_decay}")

    # Carica configurazione data augmentation
    config = ConfigHelper(config_path)
    augmentation_config = config.get_augmentation_config()
    # Defaults se config vuoto
    if not augmentation_config:
        augmentation_config = {
            "random_resized_crop": True,
            "rrc_scale": (0.85, 1.0),
            "rrc_ratio": (0.9, 1.1),
            "horizontal_flip": True,
            "rotation": 10,
        }

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

    # Model creation - Switching intelligente FROM SCRATCH vs TRANSFER LEARNING
    # Questo è il cuore del confronto scientifico del progetto:
    # - FROM SCRATCH: CNN personalizzata (requisito professore)
    # - TRANSFER LEARNING: ResNet18 pre-addestrato (per confronto performance)
    use_tl = bool(use_tl)
    if use_tl:
        print("\n🧠 Utilizzo transfer learning backbone: ResNet18 (congelato)")
        # Transfer Learning: backbone congelato, solo classificatore trainable
        model = create_breed_classifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained_backbone="resnet18",
            freeze_backbone=True,  # Solo ~61K parametri trainable
        )
    else:
        print(f"\n🧠 Training da zero - Architettura: {model_type.upper()}")
        # From Scratch: architettura CNN personalizzata completa
        model = create_breed_classifier(
            model_type=model_type,  # 'full' = 134M params, 'simple' = 3.3M params
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=True,  # Batch normalization per stabilità
        )
    model = model.to(device)

    # Training setup - Configurazione ottimizzata per deep learning
    # Loss function con label smoothing per evitare overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Smoothing = 10%

    # Optimizer AdamW: versione migliorata di Adam con weight decay corretto
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,  # L2 regularization
    )

    # Learning rate scheduler: riduce LR quando validation loss plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # Monitora validation loss (minimizzazione)
        factor=0.8,  # Riduce LR del 20% ad ogni plateau
        patience=3,  # Aspetta 3 epoche prima di ridurre
    )

    # Early stopping per prevenire overfitting
    early_stopping = EarlyStopping(patience=patience, delta=0.001)

    # Iperparametri per logging
    hparams = {
        "num_breeds": num_breeds,
        "num_classes": num_classes,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout_rate,
        "weight_decay": weight_decay,
        "use_transfer_learning": use_tl,
        "model_type": model_type if not use_tl else "resnet18",
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

    # Loop di training
    for epoch in range(num_epochs):
        print(f"\n📅 Epoca {epoch+1}/{num_epochs}")

        # Fase di training
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

            # Gradient clipping per stabilità training (previene exploding gradients)
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

        # Fase di validazione
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

                # Accuratezza Top-5 (se applicabile)
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

        # Scheduling learning rate
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

        # Salva miglior modello
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

        # Controllo early stopping
        if early_stopping(avg_val_loss):
            print(f"\n🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            writer.add_text(
                "Training/Early_Stop", f"Stopped at epoch {epoch+1}", epoch + 1
            )
            break

    # Salva modello finale
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

    # Log iperparametri con metriche finali
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
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Percorso al file di configurazione JSON",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Nome del profilo da applicare",
    )
    # Override CLI opzionali (hanno precedenza su env/profile/defaults)
    parser.add_argument("--epochs", type=int, help="Sovrascrive numero epoche")
    parser.add_argument("--lr", type=float, help="Sovrascrive learning rate")
    parser.add_argument(
        "--patience", type=int, help="Sovrascrive patience early stopping"
    )
    parser.add_argument("--batch-size", type=int, help="Sovrascrive dimensione batch")
    parser.add_argument("--dropout", type=float, help="Sovrascrive tasso dropout")
    parser.add_argument("--weight-decay", type=float, help="Sovrascrive weight decay")
    parser.add_argument(
        "--data-dir", type=str, help="Sovrascrive directory dataset splits"
    )
    parser.add_argument(
        "--use-tl",
        type=int,
        choices=[0, 1],
        help="Sovrascrive uso transfer learning (1/0)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["full", "simple"],
        help="Sovrascrive architettura modello (full/simple)",
    )

    args = parser.parse_args()

    try:
        overrides = {
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "data_dir": args.data_dir,
            "use_tl": args.use_tl,
            "model_type": args.model_type,
        }
        results = train_breeds(args.breeds, args.config, args.profile, overrides)
        print(f"\n✅ Training completato con successo!")
        print(f"   Breeds: {results['num_breeds']}")
        print(f"   Best accuracy: {results['best_val_acc']:.2f}%")

    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
