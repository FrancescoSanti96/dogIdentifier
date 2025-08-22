# 🐕 Dog Breed Identifier

**Sistema universitario di classificazione razze canine** con approccio **FROM SCRATCH vs TRANSFER LEARNING**. Progetto per corso AI13 con focus su Australian Shepherd e confronto scientifico tra metodologie.

## 🎯 **Obiettivi del Progetto**

### ✅ **1. Implementazione "DA ZERO" (Requisito Professore)**

- **Architettura CNN personalizzata**: BreedClassifier (134M parametri)
- **Training completamente from-scratch**: Nessun uso di pesi pre-addestrati
- **Codice originale**: Progettazione VGG-like con BatchNorm e Dropout

### ✅ **2. Confronto Scientifico FROM SCRATCH vs TRANSFER LEARNING**

- **Sistema duale**: Switching `USE_TL=0/1` per confronto rigoroso
- **Risultati quantitativi**: Performance gap documentato
- **Analisi critica**: Trade-off tempo vs accuracy

### ✅ **3. Focus Australian Shepherd**

- **Target personale**: Identificazione della razza del mio cane
- **Performance verificata**: 100% recall su 121 razze

## 🚀 Quick Start

```bash
# 1. Attiva ambiente virtuale (RICHIESTO)
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. FROM SCRATCH (quello che vuole il professore)
MODEL_TYPE=full USE_TL=0 python src/train.py --breeds 30

# 4. TRANSFER LEARNING (per confronto)
USE_TL=1 python src/train.py --breeds 30

# 5. TensorBoard
python scripts/launch_tensorboard.py
```

## 🔬 **Studio Comparativo: FROM SCRATCH vs TRANSFER LEARNING**

### **Metodologie Implementate**

#### **🏗️ FROM SCRATCH - BreedClassifier**

- **Architettura**: VGG-like personalizzata (5 blocchi conv + 3 FC)
- **Parametri**: 134M trainable (tutti da zero)
- **Comando**: `MODEL_TYPE=full USE_TL=0 python src/train.py --breeds 30`

#### **🔄 TRANSFER LEARNING - ResNet18**

- **Architettura**: ResNet18 pre-addestrato (ImageNet)
- **Parametri**: 61K trainable (solo classificatore)
- **Comando**: `USE_TL=1 python src/train.py --breeds 30`

### **Confronto Performance (30 Razze) - RISULTATI REALI**

| **Metrica**             | **FROM SCRATCH** | **TRANSFER LEARNING** | **Gap**        |
| ----------------------- | ---------------- | --------------------- | -------------- |
| **Validation Accuracy** | **18.73%**       | **78.0%**             | **+4.16x**     |
| **Top-5 Accuracy**      | **53.65%**       | **97.0%**             | **+1.81x**     |
| **Parametri Trainable** | **3.32M**        | **61K**               | **54x meno**   |
| **Australian Shepherd** | ✅ Incluso       | ✅ Incluso            | Stesso dataset |

## 📋 Comandi Principali

```bash
# Preparazione dataset
python src/prepare_data.py --breeds {10,30,121}

# Training FROM SCRATCH (Simple CNN)
MODEL_TYPE=simple USE_TL=0 python src/train.py --breeds 30

# Training FROM SCRATCH (Full CNN 134M param)
MODEL_TYPE=full USE_TL=0 python src/train.py --breeds 30

# Training TRANSFER LEARNING
USE_TL=1 python src/train.py --breeds {5,10,30,60,90,121}

# Valutazione modelli
python src/evaluate.py --model MODEL --data DATA --outdir OUTPUT

# Fase 2: cane personale
python src/my_dog_train.py
```

## 📊 **Risultati Consolidati (Transfer Learning)**

| Razze   | Val Acc   | Top-5   | Test Acc  | Australian Shepherd | Status |
| ------- | --------- | ------- | --------- | ------------------- | ------ |
| 5       | 98.1%     | 99%     | ~95%      | 100%                | ✅     |
| 10      | 97.9%     | 99%     | ~95%      | 100%                | ✅     |
| 30      | 89.8%     | 97%     | 89.7%     | 100%                | ✅     |
| 60      | 85.3%     | 97%     | 85.0%     | 95%+                | ✅     |
| 90      | 81.4%     | 96%     | 81.4%     | 95%+                | ✅     |
| **121** | **78.8%** | **97%** | **77.2%** | **100%**            | ✅     |

### **Architetture CNN Implementate**

#### **1. BreedClassifier (FROM SCRATCH - 134M parametri)**

```python
# Architettura VGG-like personalizzata
Input: (batch, 3, 224, 224)
├── 5 Blocchi Convoluzionali: 3→64→128→256→512→512 channels
├── Batch Normalization + Dropout2D per regolarizzazione
├── 3 Layer Fully Connected: 25088→4096→4096→classes
└── Output: (batch, num_classes) logits
```

#### **2. Transfer Learning (ResNet18 + Custom Head)**

```python
# Backbone pre-addestrato + classificatore custom
ResNet18(ImageNet) → freeze_backbone=True
├── Backbone congelato: ~11M parametri (non trainable)
├── Custom head: Dropout + Linear(512→classes)
└── Solo ~61K parametri trainable
```

### **Pipeline di Training**

```python
# Configurazione ottimizzata per deep learning
├── Loss: CrossEntropyLoss + Label Smoothing (0.1)
├── Optimizer: AdamW + Weight Decay (5e-4)
├── Scheduler: ReduceLROnPlateau (factor=0.8)
├── Regularization: Dropout + Early Stopping + Gradient Clipping
└── Monitoring: TensorBoard + Checkpoint automatici
```

## ⚙️ **Setup Ambiente**

```bash
# Crea e attiva ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt

# Alternativa conda
conda env create -f environment.yml
conda activate dogidentifier
```
