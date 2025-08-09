#!/usr/bin/env python3
"""
Analisi matrice di confusione per capire chi viene confuso con chi
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataloader import create_dataloaders_from_splits
from models.breed_classifier import create_breed_classifier


def analyze_confusion():
    """Analizza la matrice di confusione del modello"""
    print("🔍 ANALISI MATRICE DI CONFUSIONE")
    print("=" * 50)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Carica il modello salvato
    model_path = "outputs/improved/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato: {model_path}")
        return

    print(f"📂 Caricando modello da: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Informazioni dal checkpoint
    num_classes = checkpoint["num_classes"]
    breed_names = checkpoint["breed_names"]

    print(f"📊 Modello info:")
    print(f"   Classi: {num_classes}")
    print(f"   Razze: {breed_names}")
    print(f"   Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

    # Crea modello
    model = create_breed_classifier(
        model_type="simple", num_classes=num_classes, dropout_rate=0.3
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Carica dataset
    data_dir = "data/quick_splits"
    _, _, test_loader = create_dataloaders_from_splits(
        splits_dir=data_dir,
        batch_size=32,
        num_workers=2,
        image_size=(224, 224),
        augmentation_config={},  # No augmentation per test
    )

    print(f"📁 Test set: {len(test_loader.dataset)} samples")

    # Predizioni
    print("\n🔮 Generando predizioni...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Matrice di confusione
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

    # Calcola accuracy per classe
    print(f"\n📋 ACCURACY PER CLASSE:")
    print("-" * 40)

    class_accuracies = []
    for i, breed in enumerate(breed_names):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            class_accuracies.append((breed, acc, cm[i, i], cm[i].sum()))
            print(f"{breed:25}: {acc:5.1f}% ({cm[i,i]}/{cm[i].sum()})")
        else:
            print(f"{breed:25}: N/A (no samples)")

    # Ordina per accuracy
    class_accuracies.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 RANKING PER ACCURACY:")
    print("-" * 40)
    for i, (breed, acc, correct, total) in enumerate(class_accuracies):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        star = " ⭐" if "Australian" in breed else ""
        print(f"{medal} {breed:25}: {acc:5.1f}% ({correct}/{total}){star}")

    # Analisi errori più comuni
    print(f"\n🚨 ERRORI PIÙ COMUNI:")
    print("-" * 40)

    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                errors.append((breed_names[i], breed_names[j], cm[i, j], cm[i].sum()))

    # Ordina per numero di errori
    errors.sort(key=lambda x: x[2], reverse=True)

    for true_breed, pred_breed, count, total in errors[:10]:  # Top 10 errori
        percentage = count / total * 100
        print(
            f"{true_breed:20} → {pred_breed:20}: {count:2d} volte ({percentage:4.1f}%)"
        )

    # Focus su German_shepherd
    print(f"\n🔍 FOCUS SU GERMAN_SHEPHERD:")
    print("-" * 40)

    german_idx = None
    if "German_shepherd" in breed_names:
        german_idx = breed_names.index("German_shepherd")
        print(f"German_shepherd è l'indice {german_idx}")

        print(f"\nCome viene classificato German_shepherd:")
        total_german = cm[german_idx].sum()
        for j, pred_breed in enumerate(breed_names):
            count = cm[german_idx, j]
            if count > 0:
                percentage = count / total_german * 100
                correct = "✅" if j == german_idx else "❌"
                print(
                    f"  {correct} {pred_breed:20}: {count:2d}/{total_german} ({percentage:5.1f}%)"
                )

        print(f"\nChi viene classificato come German_shepherd:")
        for i, true_breed in enumerate(breed_names):
            count = cm[i, german_idx]
            if count > 0:
                total_true = cm[i].sum()
                percentage = count / total_true * 100
                correct = "✅" if i == german_idx else "❌"
                print(
                    f"  {correct} {true_breed:20}: {count:2d}/{total_true} ({percentage:5.1f}%)"
                )

    # Focus su Australian_Shepherd
    print(f"\n⭐ FOCUS SU AUSTRALIAN_SHEPHERD:")
    print("-" * 40)

    australian_idx = None
    if "Australian_Shepherd_Dog" in breed_names:
        australian_idx = breed_names.index("Australian_Shepherd_Dog")
        print(f"Australian_Shepherd_Dog è l'indice {australian_idx}")

        print(f"\nCome viene classificato Australian_Shepherd:")
        total_australian = cm[australian_idx].sum()
        for j, pred_breed in enumerate(breed_names):
            count = cm[australian_idx, j]
            if count > 0:
                percentage = count / total_australian * 100
                correct = "✅" if j == australian_idx else "❌"
                print(
                    f"  {correct} {pred_breed:20}: {count:2d}/{total_australian} ({percentage:5.1f}%)"
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

    # Accuracy per classe
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

    # Aggiungi linee di target
    plt.axvline(
        x=60.9,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label="Target Australian (60.9%)",
    )
    plt.axvline(
        x=66.2,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label="Target Overall (66.2%)",
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
    os.makedirs("outputs/analysis", exist_ok=True)
    plt.savefig("outputs/analysis/confusion_matrix.png", dpi=300, bbox_inches="tight")
    print(f"📊 Grafico salvato: outputs/analysis/confusion_matrix.png")

    plt.show()

    # Classification report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print("-" * 50)
    print(
        classification_report(all_labels, all_preds, target_names=breed_names, digits=3)
    )

    # Salva report
    with open("outputs/analysis/confusion_analysis.txt", "w") as f:
        f.write("CONFUSION MATRIX ANALYSIS\n")
        f.write("=" * 50 + "\n\n")

        f.write("ACCURACY PER CLASSE:\n")
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

    print(f"📄 Report salvato: outputs/analysis/confusion_analysis.txt")

    return {
        "confusion_matrix": cm,
        "breed_names": breed_names,
        "class_accuracies": class_accuracies,
        "errors": errors,
    }


if __name__ == "__main__":
    results = analyze_confusion()
    if results:
        print(f"\n✅ Analisi completata!")
        print(f"   Controlla outputs/analysis/ per i risultati dettagliati")
