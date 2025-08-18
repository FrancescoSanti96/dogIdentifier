#!/usr/bin/env python3
"""
Utilità di metriche e valutazione per il progetto Dog Breed Identifier
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
import torch


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcola metriche di classificazione

    Args:
        y_true: Etichette vere
        y_pred: Etichette predette
        y_prob: Probabilità predette (per ROC-AUC)

    Returns:
        Dizionario con le metriche
    """
    metrics = {}

    # Metriche base
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1

    # ROC-AUC se sono disponibili le probabilità
    if y_prob is not None:
        try:
            if len(y_prob.shape) == 1:
                # Classificazione binaria
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            else:
                # Classificazione multi-class
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
):
    """
    Crea grafico della matrice di confusione

    Args:
        y_true: Etichette vere
        y_pred: Etichette predette
        class_names: Lista dei nomi delle classi
        save_path: Percorso per salvare il grafico
        normalize: Se normalizzare la matrice
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Gestisci divisione per zero

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Matrice di Confusione")
    plt.ylabel("Etichetta Vera")
    plt.xlabel("Etichetta Predetta")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Confusion matrix saved to {save_path}")

    plt.show()


def print_metrics_summary(metrics: Dict[str, float]):
    """
    Stampa un riassunto formattato delle metriche

    Args:
        metrics: Dizionario con le metriche di valutazione
    """
    print("\n" + "=" * 50)
    print("📊 RISULTATI VALUTAZIONE")
    print("=" * 50)

    for metric, value in metrics.items():
        metric_name = metric.replace("_", " ").title()
        print(f"{metric_name:15s}: {value:.4f}")

    print("=" * 50)
