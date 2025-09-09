#!/usr/bin/env python3
"""
Training unificato per classificazione razze canine con TensorBoard, con supporto per training progressivo scalabile da 5 a 121 razze canine.

CARATTERISTICHE PRINCIPALI:
-  CNN from-scratch personalizzata (134M parametri)
-  Transfer Learning opzionale (ResNet18) 
-  CHECKPOINT RESUME: Riprendi training da qualsiasi punto intermedio
-  Scaling progressivo validato: 5→10→30→60→90→121 razze
-  Configurazioni ottimizzate per ogni scala
-  TensorBoard logging completo con hyperparameters
-  Focus Australian Shepherd (obiettivo personale)
-  Riproducibilità garantita (seed deterministico)

ARCHITETTURE SUPPORTATE:
1. FROM SCRATCH - BreedClassifier (full): 134M parametri, VGG-like personalizzata
2. FROM SCRATCH - SimpleBreedClassifier: 3.3M parametri, per test rapidi
3. TRANSFER LEARNING - ResNet18: backbone congelato, solo classificatore trainable

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


def log_hparams_final(writer, hparams, metrics):
    """
    Log finale degli hyperparameters con metriche complete.

    Args:
        writer: SummaryWriter di TensorBoard
        hparams: Dictionary degli hyperparameters
        metrics: Dictionary delle metriche finali
    """
    try:
        # Crea una sub-directory per hparam tuning 
        hparam_log_dir = os.path.join(writer.log_dir, "hparam_tuning")
        with SummaryWriter(hparam_log_dir) as hp_writer:
            hp_writer.add_hparams(hparams, metrics)
        print(f"✅ Hyperparameters salvati in: {hparam_log_dir}")
    except Exception as e:
        print(f"❌ Errore nel salvare hyperparameters: {e}")


def log_hparams_final(writer, hparams, metrics):
    """
    Log finale degli hyperparameters con metriche complete.
    
    Args:
        writer: SummaryWriter di TensorBoard
        hparams: Dictionary degli hyperparameters
        metrics: Dictionary delle metriche finali
    """
    try:
        # Crea una sub-directory per hparam tuning
        hparam_log_dir = os.path.join(writer.log_dir, "hparam_tuning")
        with SummaryWriter(hparam_log_dir) as hp_writer:
            hp_writer.add_hparams(hparams, metrics)
        print(f"✅ Hyperparameters salvati in: {hparam_log_dir}")
    except Exception as e:
        print(f"❌ Errore nel salvare hyperparameters: {e}")


def train_breeds(
    num_breeds: int,
    config_path: str = "config.json",
    profile: str | None = None,
    cli_overrides: dict | None = None,
    resume_from: str | None = None,
):
    """
    Esegue il training per la classificazione di `num_breeds` razze.

    Implementa training progressivo (5→121), resume da checkpoint e confronto
    from-scratch vs transfer learning, con logging su TensorBoard.

    Args:
        num_breeds (int): Numero di razze (5, 10, 30, 60, 90, 121)
        config_path (str): Percorso al file di configurazione JSON
        profile (str | None): Profilo di configurazione da applicare
        cli_overrides (dict | None): Override CLI dei parametri
        resume_from (str | None): Checkpoint da cui riprendere il training

    Returns:
        dict: Risultati principali e metadati (best acc, epochs, tb dir,...)
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
    # Le tre righe sopra forniscono un header chiaro all'inizio della run

    # Riproducibilità e scelta device
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configurazione TensorBoard (consente tracking di tutte le metriche)
    # Ogni run ottiene una directory dedicata su TensorBoard (per confronto esperimenti)
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
    data_dir = merged_defaults["data_dir"]  # Path agli split fisici
    num_epochs = merged_defaults["epochs"]
    learning_rate = merged_defaults["lr"]
    patience = merged_defaults["patience"]
    batch_size = merged_defaults["batch_size"]
    dropout_rate = merged_defaults["dropout"]
    weight_decay = merged_defaults["weight_decay"]
    use_tl_default = merged_defaults["use_tl"]

    # Applica override da variabili d'ambiente (comodo per lanciare test veloci)
    data_dir = os.getenv("SPLITS_DIR", data_dir)
    num_epochs = int(os.getenv("EPOCHS", str(num_epochs)))
    batch_size = int(os.getenv("BATCH_SIZE", str(batch_size)))
    learning_rate = float(os.getenv("LR", str(learning_rate)))
    patience = int(os.getenv("PATIENCE", str(patience)))
    dropout_rate = float(os.getenv("DROPOUT", str(dropout_rate)))
    weight_decay = float(os.getenv("WD", str(weight_decay)))
    use_tl = int(os.getenv("USE_TL", str(use_tl_default)))
    model_type = os.getenv("MODEL_TYPE", "full").lower()  # 'full' o 'simple'
    # Nota: MODEL_TYPE rilevante solo in training from scratch

    # Applica override CLI per ultimi (hanno massima priorità)
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

    print(f"\n⚡ CONFIGURAZIONE:")  # Riepilogo finale dei parametri effettivi
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
    # Defaults se config vuoto (garantisce esecuzione anche senza file di config)
    if not augmentation_config:
        augmentation_config = {
            "random_resized_crop": True,
            "rrc_scale": (0.85, 1.0),
            "rrc_ratio": (0.9, 1.1),
            "horizontal_flip": True,
            "rotation": 10,
        }
        # Nota: valori conservativi per ridurre over-augmentation sui cani

    print(f"\n📂 Caricando dataset da: {data_dir}")
    # Crea i DataLoader per i tre split; usa weighted sampler per class imbalance
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
    model = model.to(device)  # Sposta pesi su GPU se disponibile

    # 📊 TensorBoard Model Graph - Visualizza architettura modello 
    try:
        # Prendi un sample batch per creare il graph
        sample_batch = next(iter(train_loader))[0][:1].to(device)  # 1 immagine del batch
        writer.add_graph(model, sample_batch)
        print(f"📈 Model graph aggiunto a TensorBoard")
    except Exception as e:
        print(f"⚠️ Impossibile aggiungere model graph: {e}")

    # Training setup - Configurazione ottimizzata per deep learning
    # Loss function con label smoothing per evitare overconfidence sui dati training
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )  # Smoothing = 10% previene overfitting

    # Optimizer AdamW: versione migliorata di Adam con decoupled weight decay
    # AdamW applica weight decay direttamente sui pesi, non sui gradienti (più efficace)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,  # L2 regularization per prevenire overfitting
    )
    # Nota: AdamW separa weight decay dall'adaptive moment correction (meglio di Adam)

    # Learning rate scheduler: riduce LR quando validation loss plateau (adattivo)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # Monitora validation loss (minimizzazione)
        factor=0.8,  # Riduce LR del 20% ad ogni plateau (decadimento morbido)
        patience=3,  # Attende 3 epoche senza miglioramenti prima del decay
    )

    # Early stopping per prevenire overfitting
    early_stopping = EarlyStopping(patience=patience, delta=0)
    # delta=0 per considerare qualsiasi miglioramento (anche minimo)

    # Iperparametri per logging (solo tipi compatibili con TensorBoard)
    hparams = {
        "hp/num_breeds": int(num_breeds),
        "hp/num_classes": int(num_classes),
        "hp/epochs": int(num_epochs),
        "hp/batch_size": int(batch_size),
        "hp/learning_rate": float(learning_rate),
        "hp/dropout": float(dropout_rate),
        "hp/weight_decay": float(weight_decay),
        "hp/transfer_learning": int(use_tl),  # Convert bool to int
        "hp/model_type": str(model_type if not use_tl else "resnet18"),
        "hp/architecture": "from_scratch" if not use_tl else "transfer_learning",
        "hp/optimizer": "adamw",
        "hp/scheduler": "reduce_lr_on_plateau",
        "hp/label_smoothing": 0.1,
        "patience": int(patience),
        "dataset_name": str(os.path.basename(data_dir)),  # Solo nome directory, non path completo
    }

    print(f"\n🏋️ Inizio training...")
    print(f"   Parametri modello: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"   Parametri trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    best_val_acc = 0.0  # Migliore accuracy di validazione osservata
    best_epoch = 0
    start_epoch = 0
    epoch = 0  # Inizializza epoch per gestire casi edge (early stopping immediato)
    val_acc = 0.0  # Inizializza per evitare errori se training si interrompe subito
    train_acc = 0.0  # Inizializza anche train_acc per consistenza
    avg_val_loss = 0.0  # Inizializza avg_val_loss per evitare errori nel logging finale
    
    # 🔄 CHECKPOINT RESUME - Carica stato training da checkpoint intermedio
    if resume_from:
        print(f"\n🔄 Resuming training da checkpoint: {resume_from}")
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Checkpoint non trovato: {resume_from}")
            
        checkpoint = torch.load(resume_from, map_location=device)  # Safe load su CPU/GPU
        
        # Carica stato modello
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   ✅ Model state caricato")
        
        # Carica stato optimizer se disponibile
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"   ✅ Optimizer state caricato")
            
        # Carica stato scheduler se disponibile
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"   ✅ Scheduler state caricato")
            
        # Carica training progress
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        
        print(f"   📊 Resuming da epoca {start_epoch + 1}")
        print(f"   🏆 Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        
        # Aggiorna TensorBoard path per continuità
        tb_logdir = checkpoint.get("tensorboard_dir", tb_logdir)
        writer = SummaryWriter(tb_logdir)  # Continua a scrivere nello stesso run dir
        print(f"   📊 TensorBoard logging continua: {tb_logdir}")

    # Loop di training principale - cuore dell'addestramento
    for epoch in range(start_epoch, num_epochs):
        print(f"\n📅 Epoca {epoch+1}/{num_epochs}")

        # Fase di training - modello in modalità allenamento (dropout attivo, batchnorm updating)
        # Modalità training: abilita dropout e aggiorna statistiche BatchNorm
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Train {epoch+1}")
        for batch_idx, (data, target) in enumerate(train_bar):
            # Sposta batch su GPU/CPU
            data, target = data.to(device), target.to(device)

            # Step standard training: forward → loss → backward → optimize
            optimizer.zero_grad()  # Reset gradienti batch precedente
            output = model(data)  # Forward pass: predizioni del modello
            loss = criterion(output, target)  # Calcola CrossEntropy loss
            loss.backward()  # Backpropagation

            # Gradient clipping per stabilità training (previene exploding gradients in CNN profonde)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Aggiorna pesi del modello

            # Statistiche batch per monitoring
            running_loss += loss.item()
            _, predicted = output.max(1)  # Classe con probabilità più alta
            total += target.size(0)
            correct += predicted.eq(target).sum().item()  # Conta predizioni corrette

            # Aggiorna barra di progresso con media loss e accuracy corrente (online)
            train_bar.set_postfix(
                {
                    "Loss": f"{running_loss/(batch_idx+1):.4f}",
                    "Acc": f"{100.*correct/total:.2f}%",
                }
            )

        train_acc = 100.0 * correct / total  # Accuracy media epoca (train)
        avg_train_loss = running_loss / len(train_loader)

        # Fase di validazione - valuta performance su dati mai visti durante training
        model.eval()  # Disabilita dropout e blocca BatchNorm (modalità inference)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disabilita calcolo gradienti (più veloce + meno memoria)
            val_bar = tqdm(val_loader, desc=f"Val {epoch+1}")
            for data, target in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)  # Forward pass senza gradienti (no backward)
                loss = criterion(output, target)

                # Accumula statistiche validazione (no aggiornamento pesi!)
                val_loss += loss.item()
                _, predicted = output.max(1)  # Predizione classe più probabile
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

                # Postfix validazione: loss medio corrente e accuracy cumulativa
                val_bar.set_postfix(
                    {
                        "Loss": f"{val_loss/(len(val_bar.iterable)):.4f}",
                        "Acc": f"{100.*val_correct/val_total:.2f}%",
                    }
                )

        val_acc = 100.0 * val_correct / val_total  # Accuracy media epoca (val)
        avg_val_loss = val_loss / len(val_loader)

        # Learning Rate Scheduling - riduce LR quando validation loss smette di migliorare
        scheduler.step(avg_val_loss)  # ReduceLROnPlateau monitora val_loss
        current_lr = optimizer.param_groups[0]["lr"]

        # TensorBoard Logging - metriche visualizzabili in tempo reale  
        # Logging compatto e pulito con solo le metriche essenziali
        writer.add_scalar("loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("loss/validation", avg_val_loss, epoch + 1)
        writer.add_scalar("accuracy/train", train_acc, epoch + 1)
        writer.add_scalar("accuracy/validation", val_acc, epoch + 1)
        writer.add_scalar("learning_rate", current_lr, epoch + 1)  # Track LR decay
        # Suggerimento: apri TensorBoard per confrontare run diverse in parallelo

        print(f"   Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"   Current LR: {current_lr:.6f}")

        # Model Checkpointing - salva solo quando validation accuracy migliora
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"   🏆 NEW BEST: {val_acc:.2f}% (epoch {epoch+1})")

            # Salva checkpoint completo con metadati (per inference e analisi)
            os.makedirs(f"outputs/models/breeds_{num_breeds}", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),  # Pesi del modello
                    "optimizer_state_dict": optimizer.state_dict(),  # Stato optimizer per resume
                    "scheduler_state_dict": scheduler.state_dict(),  # Stato scheduler per resume
                    "num_classes": num_classes,  # Info architettura
                    "breed_names": breed_names,  # Mapping indici → nomi razze
                    "epoch": epoch + 1,  # utile per riprendere senza perdere il conteggio
                    "best_epoch": best_epoch,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "hyperparameters": hparams,  # Per riproducibilità
                    "tensorboard_dir": tb_logdir,  # Per resume continuità logging
                },
                f"outputs/models/breeds_{num_breeds}/best_model.pth",
            )
            
        # 💾 CHECKPOINT INTERMEDIO - Salva ogni 5 epoche per recovery
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = f"outputs/models/breeds_{num_breeds}/checkpoint_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "num_classes": num_classes,
                    "breed_names": breed_names,
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "hyperparameters": hparams,
                    "tensorboard_dir": tb_logdir,
                },
                checkpoint_path,
            )
            # Questi checkpoint intermedi aiutano in caso di crash/stop improvvisi
            print(f"   💾 Checkpoint intermedio salvato: {checkpoint_path}")

        # Early Stopping - ferma training se validation loss non migliora
        if early_stopping(avg_val_loss):
            print(f"\n🛑 Early stopping! Nessun miglioramento per {patience} epoche")
            writer.add_text(
                "Training/Early_Stop", f"Stopped at epoch {epoch+1}", epoch + 1
            )
            break

    # Salva modello finale (utile per ripresa/analisi anche senza best)
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

    # Log iperparametri con metriche finali usando approccio migliorato
    print(f"\n📊 Salvando hyperparameters in TensorBoard...")
    
    # Prepara metriche finali strutturate
    final_metrics = {
        "hparam/final_val_acc": float(val_acc),
        "hparam/best_val_acc": float(best_val_acc), 
        "hparam/final_train_acc": float(train_acc),
        "hparam/epochs_completed": float(epoch + 1),
        "hparam/best_epoch": float(best_epoch),
        "hparam/final_loss": float(avg_val_loss) if 'avg_val_loss' in locals() else 0.0,
    }
    
    # Usa il sistema migliorato di logging 
    log_hparams_final(writer, hparams, final_metrics)

    print(f"\n📈 Final Results:")
    print(f"   Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Final Val Acc: {val_acc:.2f}%")
    print(f"   Models saved: outputs/models/breeds_{num_breeds}/")
    print(f"   TensorBoard: {tb_logdir}")

    print(f"\n🔄 Chiudendo TensorBoard writer...")
    writer.close()
    print(f"✅ TensorBoard writer chiuso correttamente!")

    # Restituisce un piccolo riepilogo utile per script esterni/notebook
    return {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "epochs": epoch + 1,
        "tensorboard_dir": tb_logdir,
        "num_breeds": num_breeds,
    }


def main():
    parser = argparse.ArgumentParser(description="Training unificato razze canine")
    # Argomenti CLI principali (obbligatori/frequenti)
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
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path al checkpoint da cui riprendere il training (es: outputs/models/breeds_30/checkpoint_epoch_15.pth)",
    )

    args = parser.parse_args()  # Parsing CLI

    try:
        # Costruisci dizionario override da passare alla funzione principale
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
        results = train_breeds(
            args.breeds, args.config, args.profile, overrides, args.resume_from
        )
        print(f"\n✅ Training completato con successo!")
        print(f"   Breeds: {results['num_breeds']}")
        print(f"   Best accuracy: {results['best_val_acc']:.2f}%")

    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")  # Messaggio chiaro per debugging rapido
        sys.exit(1)


if __name__ == "__main__":
    main()
