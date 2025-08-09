# 🐕 Dog Breed Identifier

A deep learning project for dog breed classification using custom Convolutional Neural Networks built from scratch.

## 🎯 Project Overview

This project implements a two-phase dog identification system:

1. **Breed Classifier**: Multi-class classification (started with 120+ breeds, optimized to 5→10 breeds)
2. **Personal Dog Identifier**: Binary classification for personal Australian Shepherd recognition

**Note**: Il progetto è partito con l'obiettivo di 120 razze, ma attraverso un percorso sperimentale documentato in `risorse/PROCESSO.md` è stato ottimizzato per un approccio più realistico con focus su Australian Shepherd.

## 🏗️ Project Structure

```
dogIdentifier/
├── config.json                 # Main configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Datasets
│   ├── breeds/                # Dog breeds dataset
│   └── my_dog_vs_others/      # Personal dog dataset
├── models/                     # Neural architectures
├── utils/                      # Utilities and preprocessing
├── train/                      # Training scripts
├── inference/                  # Prediction scripts
└── outputs/                    # Results and checkpoints
```

## 🚀 Quick Start

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Training 5 razze (breeds_5, TL opzionale)**:

   ```bash
   # From-scratch (simple CNN)
   python quick5_tensorboard_train.py

   # Transfer learning ResNet18 (consigliato)
   USE_TL=1 python quick5_tensorboard_train.py

   # Monitor
   python launch_tensorboard.py          # Monitor: http://localhost:6006
   python test/test_validation.py        # Test e validazione quick5
   ```

3. **Training 10 razze (TOP 10 balanced)**:

   ```bash
   python prepare_top10_balanced.py      # Preparazione dataset
   python top10_balanced_train.py        # Training 10 razze
   ```

4. **Fase 2 - Mio cane**:
   ```bash
   python my_dog_train.py                # Classificazione binaria mio cane
   ```

## 📊 Features

- **Custom CNN Architecture**: Built from scratch, no pre-trained models
- **Data Augmentation**: Horizontal flip, rotation, color jitter
- **TensorBoard Integration**: Real-time training monitoring and visualization
- **Adaptive Learning Rate**: ReduceLROnPlateau scheduling for optimal training
- **Balanced Datasets**: TOP 10 breeds with optimal sample distribution
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization Tools**: Confusion matrix, ROC curves, Grad-CAM
- **Modular Design**: Clean, maintainable code structure

## 🚀 Training Scripts

### Core Scripts (snelliti):

- **`quick5_tensorboard_train.py`**: Training 5 razze con TensorBoard monitoring
- **`top10_balanced_train.py`**: Training 10 razze bilanciate
- **`my_dog_train.py`**: Classificazione binaria mio cane

Altri script sperimentali sono stati spostati in `experiments/archive/`.

### Key Features:

- **TensorBoard Integration**: Real-time loss/accuracy curves, per-class metrics
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Adaptive Learning Rate**: ReduceLROnPlateau for optimal convergence
- **Balanced Datasets**: Carefully curated breed selections for optimal performance

### Risultati Attuali:

- **5 razze (breeds_5)**: 95.2% val con TL ResNet18 (3 epoche, RRC+sampler) ✅
- **10 razze**: 28.81% val (da migliorare) ⚠️

## 🎓 Educational Value

This project demonstrates:

- Deep learning fundamentals
- Computer vision techniques
- PyTorch framework usage
- Data preprocessing and augmentation
- Model evaluation and visualization

## 📝 License

MIT License - see LICENSE file for details.
