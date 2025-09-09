#!/usr/bin/env python3
"""
🐕 ANALISI MODELLI MULTICLASS - Dog Breed Classification
Analisi matrice di confusione per modelli di classificazione razze (multi-classe).

"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Aggiungi directory padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_helper import ConfigHelper

from utils.dataloader import create_dataloaders_from_splits
from models.breed_classifier import create_breed_classifier


def analyze_confusion(
    model_path: str,
    data_dir: str,
    batch_size: int = 32,
    outdir: str = None,
):
    """
    Analizza la matrice di confusione per modelli di classificazione multiclass.

    Questa funzione è ottimizzata per modelli di classificazione razze canine e esegue:
    1. Carica modello e dataset di test multiclass
    2. Genera predizioni su test set separato (no data leakage)
    3. Calcola matrice di confusione e metriche per classe
    4. Identifica errori più comuni e pattern di confusione
    5. Focus su performance estremi (migliori/peggiori classi)
    6. Genera visualizzazioni e report dettagliati

    Output Analysis:
    - Matrice di confusione normalizzata e assoluta
    - Accuracy per classe con ranking
    - Top 10 errori più comuni tra classi
    - Analisi dettagliata best/worst performing classes
    - Grafici e report completi salvati

    Args:
        model_path: path al checkpoint (.pth) del modello multiclass
        data_dir: directory con gli split (train/val/test)
        batch_size: batch size per il test loader
        outdir: directory output per grafici e report (None = auto-genera da nome modello/dataset)

    Returns:
        Dict con risultati analisi completa (confusion matrix, accuracies, errori)

    Note:
        Per modelli binari, utilizzare src/evaluate_binary.py invece.
    """
    print("🔍 ANALISI MODELLI MULTICLASS - Confusion Matrix")
    print("=" * 55)
    
    # Auto-genera cartella output se non specificata
    if outdir is None:
        model_name = os.path.basename(model_path).replace('.pth', '')
        dataset_name = os.path.basename(data_dir.rstrip('/'))
        outdir = f"outputs/analysis/multiclass_{dataset_name}_{model_name}"
        
    print(f"📁 Output directory: {outdir}")

    # Configurazione
    # Setup hardware: utilizza GPU se disponibile per inference veloce
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Carica il modello salvato con error handling
    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato: {model_path}")
        return

    print(f"📂 Caricando modello da: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Estrai metadati dal checkpoint per ricostruire architettura
    num_classes = checkpoint["num_classes"]
    breed_names = checkpoint.get("breed_names")

    print(f"📊 Modello info:")
    print(f"   Classi: {num_classes}")
    print(f"   Razze: {breed_names}")
    print(f"   Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

    # Auto-rilevamento architettura modello dal checkpoint
    # Questo sistema intelligente rileva automaticamente se il modello usa
    # transfer learning (ResNet18) o è una CNN from-scratch
    state_dict = checkpoint["model_state_dict"]
    backbone_keys = [
        k
        for k in state_dict.keys()
        if k.startswith("layer1.")
        or k.startswith("conv1.")  # Firme caratteristiche ResNet18
    ]

    if len(backbone_keys) > 0:
        # Transfer Learning: ResNet18 backbone rilevato
        print("🧠 Rilevato backbone ResNet18 dal checkpoint")
        model = create_breed_classifier(
            model_type="simple",  # ignorato quando si specifica il backbone
            num_classes=num_classes,
            dropout_rate=0.4,
            pretrained_backbone="resnet18",
            freeze_backbone=False,  # Per valutazione, tutti i parametri attivi
        )
    else:
        # From Scratch: CNN personalizzata
        print("🧠 Rilevata CNN from-scratch dal checkpoint")
        model = create_breed_classifier(
            model_type="simple", num_classes=num_classes, dropout_rate=0.3
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Carica dataset
    _, _, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        image_size=(224, 224),
        augmentation_config={},  # Nessuna augmentation per test
    )

    # Se il checkpoint non contiene i nomi delle classi, recuperarli dal dataset
    if breed_names is None and hasattr(test_loader.dataset, "get_breed_names"):
        try:
            breed_names = test_loader.dataset.get_breed_names()
        except Exception:
            pass

    if breed_names is None:
        breed_names = [str(i) for i in range(num_classes)]

    print(f"📁 Test set: {len(test_loader.dataset)} samples")

    # Fase di Predizione - valutazione completa su test set (dati mai visti)
    print("\n🔮 Generando predizioni...")
    all_preds = []  # Predizioni del modello (indici classi)
    all_labels = []  # Ground truth labels (indici classi vere)

    # Modalità inference: no gradient computation, dropout off, batchnorm frozen
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Test in corso"):
            data, target = data.to(device), target.to(device)
            output = model(
                data
            )  # Forward pass: logits di shape (batch_size, num_classes)
            _, predicted = output.max(1)  # Classe con probabilità massima

            # Accumula predizioni per analisi globale
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Converti in numpy array per analisi numerica
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion Matrix - cuore dell'analisi: confronta predizioni vs verità
    # cm[i,j] = numero di campioni di classe i predetti come classe j
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n📊 MATRICE DI CONFUSIONE:")
    print("=" * 50)

    # Stampa matrice numerica
    print("\nMatrice (righe=vero, colonne=predetto):")
    print("Razza".ljust(20), end="")
    for name in breed_names:
        print(f"{name[:8]:>8}", end="")
    print()

    for i, true_breed in enumerate(breed_names):
        print(f"{true_breed[:20]:20}", end="")
        for j in range(num_classes):
            print(f"{cm[i,j]:8d}", end="")
        print()

    # Calcola accuratezza per classe
    print(f"\n📋 ACCURATEZZA PER CLASSE:")
    print("-" * 40)

    class_accuracies = []
    for i, breed in enumerate(breed_names):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            class_accuracies.append((breed, acc, cm[i, i], cm[i].sum()))
            print(f"{breed:25}: {acc:5.1f}% ({cm[i,i]}/{cm[i].sum()})")
        else:
            print(f"{breed:25}: N/A (no samples)")

    # Ordina per accuratezza
    class_accuracies.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 RANKING PER ACCURATEZZA:")
    print("-" * 40)
    for i, (breed, acc, correct, total) in enumerate(class_accuracies):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        star = " ⭐" if "Australian" in breed else ""
        print(f"{medal} {breed:25}: {acc:5.1f}% ({correct}/{total}){star}")

    # Analisi Errori - identifica pattern di confusione tra razze simili
    print(f"\n🚨 ERRORI PIÙ COMUNI:")
    print("-" * 40)

    # Estrai tutti gli errori dalla confusion matrix (off-diagonal elements)
    errors = []
    for i in range(num_classes):  # Classe vera (righe)
        for j in range(num_classes):  # Classe predetta (colonne)
            if i != j and cm[i, j] > 0:  # Solo misclassificazioni (non diagonal)
                # Formato: (razza_vera, razza_predetta, num_errori, totale_campioni_razza_vera)
                errors.append((breed_names[i], breed_names[j], cm[i, j], cm[i].sum()))

    # Ordina per frequenza assoluta degli errori (identificare confusion patterns)
    errors.sort(key=lambda x: x[2], reverse=True)

    # Top 10 errori più frequenti - insights per migliorare il modello
    for true_breed, pred_breed, count, total in errors[:10]:
        percentage = (
            count / total * 100
        )  # Percentuale di campioni di questa razza sbagliati così
        print(
            f"{true_breed:20} → {pred_breed:20}: {count:2d} volte ({percentage:4.1f}%)"
        )

    # Focus su Best & Worst Performing Classes
    print(f"\n📊 FOCUS SU PERFORMANCE ESTREMI:")
    print("-" * 45)

    if len(class_accuracies) >= 2:
        # Migliore performance
        best_breed, best_acc, best_correct, best_total = class_accuracies[0]
        print(f"\n🏆 MIGLIORE PERFORMANCE - {best_breed}:")
        print(f"   Accuracy: {best_acc:.1f}% ({best_correct}/{best_total})")

        best_idx = breed_names.index(best_breed)
        print(f"\n   Come viene classificato:")
        for j, pred_breed in enumerate(breed_names):
            count = cm[best_idx, j]
            if count > 0:
                percentage = count / best_total * 100
                correct = "✅" if j == best_idx else "❌"
                print(
                    f"     {correct} {pred_breed[:20]:20}: {count:2d}/{best_total} ({percentage:5.1f}%)"
                )

        # Peggiore performance
        worst_breed, worst_acc, worst_correct, worst_total = class_accuracies[-1]
        print(f"\n🔻 PEGGIORE PERFORMANCE - {worst_breed}:")
        print(f"   Accuracy: {worst_acc:.1f}% ({worst_correct}/{worst_total})")

        worst_idx = breed_names.index(worst_breed)
        print(f"\n   Come viene classificato:")
        for j, pred_breed in enumerate(breed_names):
            count = cm[worst_idx, j]
            if count > 0:
                percentage = count / worst_total * 100
                correct = "✅" if j == worst_idx else "❌"
                print(
                    f"     {correct} {pred_breed[:20]:20}: {count:2d}/{worst_total} ({percentage:5.1f}%)"
                )

        # Analisi confusioni più comuni per worst class
        print(f"\n   Principali confusioni di {worst_breed}:")
        wrong_predictions = [
            (j, cm[worst_idx, j])
            for j in range(num_classes)
            if j != worst_idx and cm[worst_idx, j] > 0
        ]
        wrong_predictions.sort(key=lambda x: x[1], reverse=True)

        for j, count in wrong_predictions[:3]:  # Top 3 confusioni
            pred_breed = breed_names[j]
            percentage = count / worst_total * 100
            print(
                f"     → Confusa con {pred_breed[:20]:20}: {count:2d} volte ({percentage:5.1f}%)"
            )

    # Visualizzazione grafica
    print(f"\n📊 Creando visualizzazione...")

    plt.figure(figsize=(12, 10))

    # Matrice di confusione normalizzata
    plt.subplot(2, 2, 1)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[name[:10] for name in breed_names],
        yticklabels=[name[:10] for name in breed_names],
    )
    plt.title("Matrice di Confusione Normalizzata")
    plt.ylabel("Vero")
    plt.xlabel("Predetto")

    # Matrice di confusione assoluta
    plt.subplot(2, 2, 2)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=[name[:10] for name in breed_names],
        yticklabels=[name[:10] for name in breed_names],
    )
    plt.title("Matrice di Confusione Assoluta")
    plt.ylabel("Vero")
    plt.xlabel("Predetto")

    # Accuratezza per classe
    plt.subplot(2, 2, 3)
    breeds = [acc[0][:15] for acc in class_accuracies]
    accuracies = [acc[1] for acc in class_accuracies]
    colors = [
        "red" if acc < 40 else "orange" if acc < 60 else "green" for acc in accuracies
    ]

    plt.barh(breeds, accuracies, color=colors)
    plt.xlabel("Accuracy (%)")
    plt.title("Accuracy per Classe")
    plt.xlim(0, 100)

    # Linee di riferimento per performance
    overall_acc = sum(accuracies) / len(accuracies) if accuracies else 0
    plt.axvline(
        x=overall_acc,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Media Overall ({overall_acc:.1f}%)",
    )
    if accuracies:
        best_acc = max(accuracies)
        plt.axvline(
            x=best_acc,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Best Class ({best_acc:.1f}%)",
        )
    plt.legend()

    # Distribuzione campioni
    plt.subplot(2, 2, 4)
    sample_counts = [cm[i].sum() for i in range(num_classes)]
    plt.bar(range(num_classes), sample_counts, color="skyblue")
    plt.xlabel("Razza")
    plt.ylabel("Numero Campioni Test")
    plt.title("Distribuzione Campioni Test")
    plt.xticks(range(num_classes), [name[:10] for name in breed_names], rotation=45)

    plt.tight_layout()

    # Salva grafico
    os.makedirs(outdir, exist_ok=True)
    fig_path = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"📊 Grafico salvato: {fig_path}")

    plt.show()

    # Report di classificazione
    print(f"\n📋 REPORT DI CLASSIFICAZIONE:")
    print("-" * 50)
    print(
        classification_report(all_labels, all_preds, target_names=breed_names, digits=3)
    )

    # Salva report
    report_path = os.path.join(outdir, "confusion_analysis.txt")
    with open(report_path, "w") as f:
        f.write("ANALISI MATRICE DI CONFUSIONE\n")
        f.write("=" * 50 + "\n\n")

        f.write("ACCURATEZZA PER CLASSE:\n")
        for breed, acc, correct, total in class_accuracies:
            f.write(f"{breed:25}: {acc:5.1f}% ({correct}/{total})\n")

        f.write(f"\nERRORI PIÙ COMUNI:\n")
        for true_breed, pred_breed, count, total in errors[:10]:
            percentage = count / total * 100
            f.write(
                f"{true_breed:20} → {pred_breed:20}: {count:2d} volte ({percentage:4.1f}%)\n"
            )

        f.write(f"\nCLASSIFICATION REPORT:\n")
        f.write(
            classification_report(
                all_labels, all_preds, target_names=breed_names, digits=3
            )
        )

    print(f"📄 Report salvato: {report_path}")

    return {
        "confusion_matrix": cm,
        "breed_names": breed_names,
        "class_accuracies": class_accuracies,
        "errors": errors,
        "report_path": report_path,
        "fig_path": fig_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analizza la matrice di confusione per un modello addestrato"
    )
    parser.add_argument(
        "--model", required=True, help="Percorso al checkpoint del modello .pth"
    )
    parser.add_argument(
        "--data", required=True, help="Percorso alla directory degli split del dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Dimensione batch per il test"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory di output per grafici e report (default: auto-generata da nome modello/dataset)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Percorso al file di configurazione JSON (default: config.json)",
    )
    args = parser.parse_args()

    # Se presente config, consenti override di data_dir e batch size
    try:
        cfg = ConfigHelper(args.config)
        default_data_dir = cfg.get("paths.default_eval_data_dir") or cfg.get(
            "data.balanced_splits_dir"
        )
        if default_data_dir and (args.data is None or args.data == ""):
            args.data = default_data_dir
        batch_override = cfg.get("data.batch_size")
        if batch_override:
            args.batch_size = int(batch_override)
    except Exception:
        pass

    results = analyze_confusion(
        model_path=args.model,
        data_dir=args.data,
        batch_size=args.batch_size,
        outdir=args.outdir,
    )
    if results:
        print(f"\n✅ Analisi completata!")
        print(f"   Controlla outputs/analysis/ per i risultati dettagliati")
