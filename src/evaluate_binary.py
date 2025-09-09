#!/usr/bin/env python3
"""
🐕 ANALISI MODELLI BINARI - Dog Classification
Analisi completa per modelli di classificazione binaria (es. Maggie vs Altri)

Uso:
  python src/evaluate_binary.py \
    --model outputs/my_dog/best_model.pth \
    --data data/my_dog_vs_others_splits \
    [--batch-size 32] [--outdir outputs/analysis/binary]
"""

import os
import sys
import argparse
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from shutil import copy2
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Aggiungi repo al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import create_dataloaders_from_splits
from models.breed_classifier import create_breed_classifier


def analyze_binary_model(
    model_path: str,
    data_dir: str,
    batch_size: int = 32,
    outdir: str = "outputs/analysis/binary",
    class_names: list = None,
):
    """
    Analisi completa per modelli di classificazione binaria.

    Questa funzione esegue un'analisi dettagliata delle performance
    di un modello binario, includendo:

    1. Metriche standard: Accuracy, Precision, Recall, F1-Score
    2. Confusion Matrix con visualizzazione
    3. ROC Curve e AUC
    4. Precision-Recall Curve
    5. Confidence Analysis e distribuzione probabilità
    6. Threshold Optimization
    7. Misclassified Samples Analysis
    8. Report dettagliato per documentazione (PROCESSO.md)

    Args:
        model_path (str): Path al checkpoint del modello (.pth)
        data_dir (str): Directory con gli split del dataset (train/val/test)
        batch_size (int): Batch size per l'inferenza
        outdir (str): Directory output per risultati e grafici
        class_names (list): Nome delle classi ['Altri', 'Maggie']

    Returns:
        dict: Risultati completi dell'analisi
    """

    print("🐕 ANALISI MODELLO BINARIO")
    print("=" * 60)
    print(f"📂 Modello: {os.path.basename(model_path)}")
    print(f"📊 Dataset: {data_dir}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)

    # Default class names per modello my_dog
    if class_names is None:
        class_names = ["Altri", "Maggie"]

    # Carica modello
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    print(f"🧠 Caricando modello...")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 2)

    if num_classes != 2:
        raise ValueError(
            f"Modello non binario! Trovate {num_classes} classi (deve essere 2)"
        )

    # Auto-detect architettura
    state_dict = checkpoint["model_state_dict"]
    backbone_keys = [
        k
        for k in state_dict.keys()
        if k.startswith("layer1.") or k.startswith("conv1.")
    ]

    if len(backbone_keys) > 0:
        print("🔄 Rilevato Transfer Learning (ResNet18)")
        model = create_breed_classifier(
            model_type="simple",
            num_classes=num_classes,
            dropout_rate=0.4,
            pretrained_backbone="resnet18",
            freeze_backbone=False,
        )
    else:
        print("⚡ Rilevato Simple CNN")
        model = create_breed_classifier(
            model_type="simple", num_classes=num_classes, dropout_rate=0.3
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Carica dataset BINARIO usando MyDogDataset (fix per mapping classi corretto)
    print(f"📁 Caricando dataset binario...")
    from utils.dataloader import MyDogDataset
    from torchvision import transforms

    # Transform base per test (no augmentation)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = MyDogDataset(
        os.path.join(data_dir, "test"),
        transform=test_transform,
        my_dog_folder="maggie",  # Classe 1
        other_dogs_folder="other",  # Classe 0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    print(f"📊 Test set: {len(test_loader.dataset)} samples")

    # Inference
    print(f"🔮 Generando predizioni...")
    all_preds = []
    all_labels = []
    all_probs = []
    filenames = []

    idx_offset = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferenza"):
            data, target = batch
            data, target = data.to(device), target.to(device)

            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, pred = output.max(1)

            # Estrai nomi file da MyDogDataset
            batch_size_actual = data.size(0)
            for i in range(batch_size_actual):
                global_idx = idx_offset + i
                try:
                    # MyDogDataset ha attributo .images (lista di percorsi)
                    file_path = test_loader.dataset.images[global_idx]
                    filenames.append(os.path.basename(file_path))
                except (AttributeError, IndexError):
                    filenames.append(f"sample_{global_idx}")

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            idx_offset += batch_size_actual

    # Converti a numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    print(f"✅ Raccolte {len(y_true)} predizioni")

    # =================== ANALISI METRICHE ===================

    print(f"\n📊 ANALISI METRICHE BINARIE")
    print("=" * 50)

    # Metriche base
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision_macro = precision_score(y_true, y_pred, average="macro") * 100
    recall_macro = recall_score(y_true, y_pred, average="macro") * 100
    f1_macro = f1_score(y_true, y_pred, average="macro") * 100

    print(f"🎯 METRICHE GLOBALI:")
    print(f"   Accuracy:     {accuracy:.2f}%")
    print(f"   Precision:    {precision_macro:.2f}%")
    print(f"   Recall:       {recall_macro:.2f}%")
    print(f"   F1-Score:     {f1_macro:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"        Pred:")
    print(f"Vero    {class_names[0]:<8} {class_names[1]:<8}")
    print(f"{class_names[0]:<8} {cm[0,0]:<8} {cm[0,1]:<8}")
    print(f"{class_names[1]:<8} {cm[1,0]:<8} {cm[1,1]:<8}")

    # Distribuzione campioni
    class_0_count = np.sum(y_true == 0)
    class_1_count = np.sum(y_true == 1)

    print(f"\n📊 DISTRIBUZIONE DATASET:")
    print(
        f"   {class_names[0]}: {class_0_count} samples ({class_0_count/len(y_true)*100:.1f}%)"
    )
    print(
        f"   {class_names[1]}: {class_1_count} samples ({class_1_count/len(y_true)*100:.1f}%)"
    )

    # Metriche per classe
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, digits=3
    )

    print(f"\n📋 METRICHE PER CLASSE:")
    print("-" * 50)

    for i, class_name in enumerate(class_names):
        precision_cls = report[class_name]["precision"] * 100
        recall_cls = report[class_name]["recall"] * 100
        f1_cls = report[class_name]["f1-score"] * 100
        support = int(report[class_name]["support"])

        star = " ⭐" if "Maggie" in class_name else ""

        print(f"{class_name.upper()}{star}:")
        print(f"   Precision: {precision_cls:>6.1f}% ({cm[i,i]}/{cm[i,i] + cm[1-i,i]})")
        print(f"   Recall:    {recall_cls:>6.1f}% ({cm[i,i]}/{support})")
        print(f"   F1-Score:  {f1_cls:>6.1f}%")
        print(f"   Support:   {support} samples")
        print()

    # =================== CONFIDENCE ANALYSIS ===================

    print(f"🎯 CONFIDENCE ANALYSIS:")
    print("-" * 30)

    # Probabilità per Maggie (classe 1)
    maggie_probs = y_probs[:, 1] * 100

    # Confidence per campioni corretti vs sbagliati
    correct_mask = y_true == y_pred
    correct_confidences = np.max(y_probs[correct_mask], axis=1) * 100
    wrong_confidences = np.max(y_probs[~correct_mask], axis=1) * 100

    print(f"   Media confidence corrette: {np.mean(correct_confidences):.1f}%")
    print(f"   Media confidence sbagliate: {np.mean(wrong_confidences):.1f}%")

    # Confidence per Maggie specificamente
    maggie_samples = y_true == 1
    if np.any(maggie_samples):
        maggie_avg_conf = np.mean(maggie_probs[maggie_samples])
        print(f"   Media confidence Maggie: {maggie_avg_conf:.1f}%")

    # =================== ROC ANALYSIS ===================

    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_probs[:, 1]
    )
    pr_auc = auc(recall_curve, precision_curve)

    print(f"\n📈 CURVE ANALYSIS:")
    print(f"   ROC AUC:  {roc_auc:.3f}")
    print(f"   PR AUC:   {pr_auc:.3f}")

    # =================== VISUALIZZAZIONI ===================

    print(f"\n📊 Creando visualizzazioni...")

    # Setup figura con subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Confusion Matrix (2x3 layout: posizione 1)
    plt.subplot(2, 4, 1)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proporzione"},
    )
    plt.title("Confusion Matrix\n(Normalizzata)")
    plt.ylabel("Vero")
    plt.xlabel("Predetto")

    # 2. Confusion Matrix Assoluta (posizione 2)
    plt.subplot(2, 4, 2)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Conteggio"},
    )
    plt.title("Confusion Matrix\n(Valori Assoluti)")
    plt.ylabel("Vero")
    plt.xlabel("Predetto")

    # 3. ROC Curve (posizione 3)
    plt.subplot(2, 4, 3)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 4. Precision-Recall Curve (posizione 4)
    plt.subplot(2, 4, 4)
    plt.plot(
        recall_curve,
        precision_curve,
        color="blue",
        lw=2,
        label=f"PR (AUC = {pr_auc:.3f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # 5. Distribuzione Confidence (posizione 5)
    plt.subplot(2, 4, 5)

    # Confidence per campioni corretti e sbagliati
    plt.hist(
        correct_confidences,
        bins=20,
        alpha=0.7,
        label=f"Corrette ({len(correct_confidences)})",
        color="green",
        density=True,
    )

    if len(wrong_confidences) > 0:
        plt.hist(
            wrong_confidences,
            bins=20,
            alpha=0.7,
            label=f"Sbagliate ({len(wrong_confidences)})",
            color="red",
            density=True,
        )

    plt.xlabel("Confidence (%)")
    plt.ylabel("Densità")
    plt.title("Distribuzione Confidence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Metriche per Classe (Bar Plot - posizione 6)
    plt.subplot(2, 4, 6)

    metrics_data = []
    labels = []
    colors = []

    for i, class_name in enumerate(class_names):
        precision_val = report[class_name]["precision"] * 100
        recall_val = report[class_name]["recall"] * 100
        f1_val = report[class_name]["f1-score"] * 100

        metrics_data.extend([precision_val, recall_val, f1_val])
        labels.extend(
            [f"{class_name}\nPrecision", f"{class_name}\nRecall", f"{class_name}\nF1"]
        )
        colors.extend(["skyblue", "lightgreen", "lightcoral"])

    bars = plt.bar(range(len(metrics_data)), metrics_data, color=colors, alpha=0.8)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Percentuale (%)")
    plt.title("Metriche per Classe")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis="y")

    # Aggiungi valori sopra le barre
    for bar, val in zip(bars, metrics_data):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 7. Distribuzione Probabilità Maggie (posizione 7)
    plt.subplot(2, 4, 7)

    # Probabilità per veri Maggie vs veri Altri
    true_maggie_probs = maggie_probs[y_true == 1]
    true_others_probs = maggie_probs[y_true == 0]

    plt.hist(
        true_others_probs,
        bins=15,
        alpha=0.7,
        label=f"Veri {class_names[0]} ({len(true_others_probs)})",
        color="orange",
        density=True,
    )
    plt.hist(
        true_maggie_probs,
        bins=15,
        alpha=0.7,
        label=f"Veri {class_names[1]} ({len(true_maggie_probs)})",
        color="purple",
        density=True,
    )

    plt.axvline(x=50, color="red", linestyle="--", alpha=0.7, label="Threshold 50%")
    plt.xlabel("Probabilità Maggie (%)")
    plt.ylabel("Densità")
    plt.title("Distribuzione Prob. Maggie\nper Classe Vera")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Summary Metrics (posizione 8)
    plt.subplot(2, 4, 8)
    plt.axis("off")  # Rimuovi assi per testo

    # Crea summary testuale
    summary_text = f"""
📊 SUMMARY RESULTS

🎯 Overall Accuracy: {accuracy:.1f}%

⭐ {class_names[1].upper()} METRICS:
   • Precision: {report[class_names[1]]['precision']*100:.1f}%
   • Recall: {report[class_names[1]]['recall']*100:.1f}%
   • F1-Score: {report[class_names[1]]['f1-score']*100:.1f}%

📋 {class_names[0].upper()} METRICS:
   • Precision: {report[class_names[0]]['precision']*100:.1f}%
   • Recall: {report[class_names[0]]['recall']*100:.1f}%
   • F1-Score: {report[class_names[0]]['f1-score']*100:.1f}%

📈 CURVE METRICS:
   • ROC AUC: {roc_auc:.3f}
   • PR AUC: {pr_auc:.3f}

📊 TEST SET:
   • Total Samples: {len(y_true)}
   • {class_names[0]}: {class_0_count} ({class_0_count/len(y_true)*100:.1f}%)
   • {class_names[1]}: {class_1_count} ({class_1_count/len(y_true)*100:.1f}%)
"""

    plt.text(
        0.05,
        0.95,
        summary_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    # Salva figura
    fig_path = os.path.join(
        outdir,
        f"binary_analysis_{os.path.basename(model_path).replace('.pth', '')}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"📊 Grafico salvato: {fig_path}")

    plt.show()

    # =================== MISCLASSIFIED ANALYSIS ===================

    print(f"\n🔍 ANALISI ERRORI:")
    print("-" * 30)

    # Trova campioni sbagliati
    misclassified_idx = np.where(y_pred != y_true)[0]

    if len(misclassified_idx) > 0:
        print(
            f"❌ Campioni sbagliati: {len(misclassified_idx)}/{len(y_true)} ({len(misclassified_idx)/len(y_true)*100:.1f}%)"
        )

        # Analizza tipi di errori
        false_positives = np.sum((y_true == 0) & (y_pred == 1))  # Altri → Maggie
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))  # Maggie → Altri

        print(
            f"   False Positives ({class_names[0]}→{class_names[1]}): {false_positives}"
        )
        print(
            f"   False Negatives ({class_names[1]}→{class_names[0]}): {false_negatives}"
        )

        # Copia top errori per ispezione manuale
        misclassified_dir = os.path.join(outdir, "misclassified")
        os.makedirs(misclassified_dir, exist_ok=True)

        print(f"📁 Copiando campioni sbagliati in: {misclassified_dir}")

        # Ordina per confidence (più confidenti ma sbagliati = più interessanti)
        confidences_wrong = np.max(y_probs[misclassified_idx], axis=1)
        sorted_idx = np.argsort(-confidences_wrong)  # Decrescente

        copied_count = 0
        for i in sorted_idx[: min(20, len(sorted_idx))]:  # Top 20 errori
            original_idx = misclassified_idx[i]
            filename = filenames[original_idx]

            try:
                # Trova file sorgente
                src_path = None
                for split in ["test", "val", "train"]:
                    for class_dir in [
                        class_names[0].lower(),
                        class_names[1].lower(),
                        "maggie",
                        "other",
                    ]:
                        potential_path = os.path.join(
                            data_dir, split, class_dir, filename
                        )
                        if os.path.exists(potential_path):
                            src_path = potential_path
                            break
                    if src_path:
                        break

                if src_path:
                    # Nome descrittivo per file copiato
                    true_label = class_names[y_true[original_idx]]
                    pred_label = class_names[y_pred[original_idx]]
                    conf = confidences_wrong[i] * 100

                    dst_name = f"{copied_count+1:02d}_{true_label}_pred_{pred_label}_conf_{conf:.1f}%_{filename}"
                    dst_path = os.path.join(misclassified_dir, dst_name)

                    copy2(src_path, dst_path)
                    copied_count += 1

            except Exception as e:
                print(f"   ⚠️ Errore copiando {filename}: {e}")

        print(f"✅ Copiati {copied_count} campioni sbagliati")

    else:
        print("✅ Nessun errore trovato! Modello perfetto!")

    # =================== SAVE RESULTS ===================

    # Salva array numpy per analisi future
    np.save(os.path.join(outdir, "y_true.npy"), y_true)
    np.save(os.path.join(outdir, "y_pred.npy"), y_pred)
    np.save(os.path.join(outdir, "y_probs.npy"), y_probs)

    # Salva CSV dettagliato
    csv_path = os.path.join(outdir, "predictions_detailed.csv")
    with open(csv_path, "w") as f:
        f.write(
            "idx,filename,true_label,pred_label,correct,prob_class0,prob_class1,confidence\n"
        )
        for i, (filename, true, pred, probs) in enumerate(
            zip(filenames, y_true, y_pred, y_probs)
        ):
            correct = "YES" if true == pred else "NO"
            confidence = max(probs) * 100
            f.write(
                f"{i},{filename},{class_names[true]},{class_names[pred]},{correct},{probs[0]:.4f},{probs[1]:.4f},{confidence:.2f}\n"
            )

    # Report testuale per PROCESSO.md
    report_path = os.path.join(outdir, "binary_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("🐕 BINARY MODEL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"MODEL: {os.path.basename(model_path)}\n")
        f.write(f"DATASET: {data_dir}\n")
        f.write(f"TEST SAMPLES: {len(y_true)}\n")
        f.write(f"CLASSES: {class_names}\n\n")

        f.write("📊 PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")

        for i, class_name in enumerate(class_names):
            precision_cls = report[class_name]["precision"] * 100
            recall_cls = report[class_name]["recall"] * 100
            f1_cls = report[class_name]["f1-score"] * 100
            support = int(report[class_name]["support"])

            f.write(f"{class_name.upper()}:\n")
            f.write(f"  Precision: {precision_cls:.1f}%\n")
            f.write(f"  Recall:    {recall_cls:.1f}%\n")
            f.write(f"  F1-Score:  {f1_cls:.1f}%\n")
            f.write(f"  Support:   {support} samples\n\n")

        f.write("📋 CONFUSION MATRIX:\n")
        f.write("-" * 20 + "\n")
        f.write(f"True\\Pred    {class_names[0]:<8} {class_names[1]:<8}\n")
        f.write(f"{class_names[0]:<12} {cm[0,0]:<8} {cm[0,1]:<8}\n")
        f.write(f"{class_names[1]:<12} {cm[1,0]:<8} {cm[1,1]:<8}\n\n")

        f.write("📈 CURVE METRICS:\n")
        f.write("-" * 15 + "\n")
        f.write(f"ROC AUC:  {roc_auc:.3f}\n")
        f.write(f"PR AUC:   {pr_auc:.3f}\n\n")

        f.write("🎯 CONFIDENCE ANALYSIS:\n")
        f.write("-" * 22 + "\n")
        f.write(f"Average confidence (correct): {np.mean(correct_confidences):.1f}%\n")
        if len(wrong_confidences) > 0:
            f.write(
                f"Average confidence (wrong):   {np.mean(wrong_confidences):.1f}%\n"
            )

        f.write(f"\n🔍 ERROR ANALYSIS:\n")
        f.write("-" * 16 + "\n")
        f.write(
            f"Misclassified: {len(misclassified_idx)}/{len(y_true)} ({len(misclassified_idx)/len(y_true)*100:.1f}%)\n"
        )
        if len(misclassified_idx) > 0:
            f.write(f"False Positives: {false_positives}\n")
            f.write(f"False Negatives: {false_negatives}\n")

    print(f"📄 Report salvato: {report_path}")

    # =================== SUMMARY PER PROCESSO.MD ===================

    print(f"\n📝 SUMMARY PER PROCESSO.MD SEZIONE 12:")
    print("=" * 55)
    print(f"✅ Test Accuracy: {accuracy:.1f}%")
    print(f"⭐ {class_names[1]} Recall: {report[class_names[1]]['recall']*100:.1f}%")
    print(
        f"📊 {class_names[1]} Precision: {report[class_names[1]]['precision']*100:.1f}%"
    )
    print(
        f"🎯 {class_names[1]} F1-Score: {report[class_names[1]]['f1-score']*100:.1f}%"
    )
    print(f"📈 ROC AUC: {roc_auc:.3f}")
    print(
        f"🔍 Misclassified: {len(misclassified_idx)}/{len(y_true)} ({len(misclassified_idx)/len(y_true)*100:.1f}%)"
    )

    # Costruisci risultato finale
    results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "class_names": class_names,
        "per_class_metrics": report,
        "misclassified_count": len(misclassified_idx),
        "false_positives": false_positives if len(misclassified_idx) > 0 else 0,
        "false_negatives": false_negatives if len(misclassified_idx) > 0 else 0,
        "model_path": model_path,
        "data_dir": data_dir,
        "report_path": report_path,
        "fig_path": fig_path,
        "outdir": outdir,
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🐕 Analisi modelli binari per classificazione cani"
    )

    parser.add_argument(
        "--model", required=True, help="Path al checkpoint del modello binario (.pth)"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path alla directory con gli split del dataset (train/val/test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per l'inferenza (default: 32)",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/analysis/binary",
        help="Directory output per risultati e grafici (default: outputs/analysis/binary)",
    )
    parser.add_argument(
        "--class-names",
        nargs=2,
        default=["Altri", "Maggie"],
        help="Nomi delle due classi (default: Altri Maggie)",
    )

    args = parser.parse_args()

    # Validazione input
    if not os.path.exists(args.model):
        print(f"❌ Modello non trovato: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"❌ Dataset non trovato: {args.data}")
        sys.exit(1)

    try:
        results = analyze_binary_model(
            model_path=args.model,
            data_dir=args.data,
            batch_size=args.batch_size,
            outdir=args.outdir,
            class_names=args.class_names,
        )

        print(f"\n✅ Analisi completata con successo!")
        print(f"📊 Risultati salvati in: {args.outdir}")
        print(f"📈 Grafico: {results['fig_path']}")
        print(f"📄 Report: {results['report_path']}")

    except Exception as e:
        print(f"\n❌ Errore durante l'analisi: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
