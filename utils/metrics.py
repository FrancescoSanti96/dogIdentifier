#!/usr/bin/env python3
"""
Metrics and evaluation utilities for dog breed identifier project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, roc_curve
)
import torch
import cv2
from PIL import Image


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # ROC-AUC if probabilities available
    if y_prob is not None:
        try:
            if len(y_prob.shape) == 1:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            else:
                # Multi-class classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         class_names: List[str],
                         save_path: Optional[str] = None,
                         normalize: bool = True):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                   y_prob: np.ndarray, 
                   class_names: List[str],
                   save_path: Optional[str] = None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names
        save_path: Path to save plot
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        # One-vs-rest approach
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
        auc = roc_auc_score(y_true_binary, y_prob_binary)
        
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curves saved to {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history and 'val_acc' in history:
        ax2.plot(history['train_acc'], label='Training Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved to {save_path}")
    
    plt.show()


def save_results(metrics: Dict[str, float], 
                predictions: np.ndarray,
                true_labels: np.ndarray,
                class_names: List[str],
                save_path: str):
    """
    Save evaluation results to file
    
    Args:
        metrics: Dictionary with evaluation metrics
        predictions: Model predictions
        true_labels: True labels
        class_names: List of class names
        save_path: Path to save results
    """
    with open(save_path, 'w') as f:
        f.write("Dog Breed Classification Results\n")
        f.write("=" * 40 + "\n\n")
        
        # Write metrics
        f.write("Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nDetailed Results:\n")
        f.write("-" * 20 + "\n")
        
        # Write predictions vs true labels
        f.write("Sample Predictions:\n")
        for i in range(min(20, len(predictions))):
            true_class = class_names[true_labels[i]]
            pred_class = class_names[predictions[i]]
            correct = "✓" if true_labels[i] == predictions[i] else "✗"
            f.write(f"{i+1:3d}. True: {true_class:20s} | Pred: {pred_class:20s} {correct}\n")
    
    print(f"📄 Results saved to {save_path}")


def visualize_grad_cam(model: torch.nn.Module,
                      image: torch.Tensor,
                      target_class: int,
                      layer_name: str = "features",
                      save_path: Optional[str] = None) -> np.ndarray:
    """
    Generate Grad-CAM visualization for model interpretability
    
    Args:
        model: Trained model
        image: Input image tensor
        target_class: Target class for visualization
        layer_name: Name of the layer to use for Grad-CAM
        save_path: Path to save visualization
        
    Returns:
        Grad-CAM heatmap
    """
    model.eval()
    
    # Register hooks
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
    
    def save_activation(module, input, output):
        activations.append(output)
    
    # Register hooks
    for name, module in model.named_modules():
        if layer_name in name:
            module.register_forward_hook(save_activation)
            break
    
    # Forward pass
    image.requires_grad_(True)
    output = model(image.unsqueeze(0))
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients and activations
    gradients = image.grad
    activations = activations[0]
    
    # Calculate weights
    weights = torch.mean(gradients, dim=(1, 2))
    
    # Generate heatmap
    heatmap = torch.zeros(activations.shape[2:])
    for i, w in enumerate(weights):
        heatmap += w * activations[0, i, :, :]
    
    heatmap = torch.relu(heatmap)
    heatmap = heatmap.detach().numpy()
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    
    # Convert image to numpy for visualization
    img_np = image.permute(1, 2, 0).detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎯 Grad-CAM visualization saved to {save_path}")
    
    plt.show()
    
    return heatmap


def print_metrics_summary(metrics: Dict[str, float]):
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    
    for metric, value in metrics.items():
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:15s}: {value:.4f}")
    
    print("="*50)


if __name__ == "__main__":
    # Test metrics functions
    print("🧪 Testing metrics module...")
    
    # Create dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 100)
    y_pred = np.random.randint(0, 5, 100)
    y_prob = np.random.rand(100, 5)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    class_names = ['Breed_1', 'Breed_2', 'Breed_3', 'Breed_4', 'Breed_5']
    
    # Test metrics calculation
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics_summary(metrics)
    
    print("✅ Metrics module test completed!") 