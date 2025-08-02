# 🐕 Dog Breed Identifier

A deep learning project for dog breed classification using custom Convolutional Neural Networks built from scratch.

## 🎯 Project Overview

This project implements a two-phase dog identification system:
1. **Breed Classifier**: Multi-class classification of 120+ dog breeds
2. **Personal Dog Identifier**: Binary classification for personal dog recognition

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

2. **Download dataset**:
   ```bash
   python download_datasets.py --dataset stanford
   ```

3. **Quick training test**:
   ```bash
   python quick_train.py
   ```

## 📊 Features

- **Custom CNN Architecture**: Built from scratch, no pre-trained models
- **Data Augmentation**: Horizontal flip, rotation, color jitter
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization Tools**: Confusion matrix, ROC curves, Grad-CAM
- **Modular Design**: Clean, maintainable code structure

## 🎓 Educational Value

This project demonstrates:
- Deep learning fundamentals
- Computer vision techniques
- PyTorch framework usage
- Data preprocessing and augmentation
- Model evaluation and visualization

## 📝 License

MIT License - see LICENSE file for details. 