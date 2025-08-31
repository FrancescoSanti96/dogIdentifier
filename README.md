# 🐕 Dog Breed Identifier

**Sistema universitario di classificazione razze canine** con approccio **FROM SCRATCH vs TRANSFER LEARNING**. Con focus su Australian Shepherd e confronto scientifico tra metodologie.

## ⚙️ **Setup Ambiente**

```bash
# Setup rapido
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Alternativa conda
conda env create -f environment.yml && conda activate dogidentifier
```

## 🚀 Quick Start

````bash
# 1. Setup ambiente
source .venv/bin/activate && pip install -r requirements.txt

# 2. Installa dipendenze
pip install -r requirements.txt

# 2.1. Preparazione dati
python src/prepare_data.py --breeds 30

# 3. Training FROM SCRATCH
MODEL_TYPE=full USE_TL=0 python src/train.py --breeds 30

# 4. Training TRANSFER LEARNING (confronto)
USE_TL=1 python src/train.py --breeds 30


## **Sistema di Predizione**

### **Universal Dog Classifier - Un solo script per tutto! 🚀**

Il nostro `predict.py` è un sistema intelligente che **auto-detecta** il tipo di modello e adatta automaticamente il comportamento:

#### **📖 Sintassi base:**

```bash
python predict.py <immagine> <modello> [opzioni]
````

#### **🎯 Modalità di utilizzo:**

##### **1️⃣ Solo classificazione razze** (Auto-detect)

```bash
python predict.py dog.jpg outputs/models/breeds_10/best_model.pth
```

- Riconosce automaticamente che è un modello Transfer Learning
- Classifica tra le razze disponibili
- Mostra i top 3 risultati con percentuali

##### **2️⃣ Classificazione razze + Test Maggie automatico** ⭐

```bash
python predict.py dog.jpg outputs/models/breeds_10/best_model.pth --binary-model outputs/my_dog/best_model.pth
```

- Prima fa classificazione razze
- **SE** trova "Australian Shepherd", triggera automaticamente il test "È MAGGIE?"
- Se NON trova Australian Shepherd, si ferma alla classificazione razze

##### **3️⃣ Solo test binario "È MAGGIE?"**

```bash
python predict.py dog.jpg outputs/my_dog/best_model.pth
```

- Riconosce automaticamente che è un modello binario
- Testa direttamente se il cane è Maggie o no

#### **⚙️ Opzioni avanzate:**

```bash
--top-k 5           # Mostra top 5 invece di top 3 (solo razze)
--threshold 0.7     # Soglia per trigger binario (default 0.5)
```

## **Struttura Progetto**

```
📦 dogIdentifier_clean/
├── 🐕 predict.py              # Sistema predizione universale
├── 📂 src/                    # Core training & evaluation
│   ├── train.py              # Training multiclass (FROM SCRATCH/TL)
│   ├── my_dog_train.py       # Training binario "È MAGGIE?"
│   ├── evaluate.py           # Valutazione modelli
│   └── prepare_data.py       # Preparazione dataset
├── 📂 models/                 # Architetture CNN
├── 📂 utils/                  # Utilities (dataloader, metrics, etc.)
├── 📂 data/                   # Dataset e splits
└── 📂 outputs/                # Modelli salvati e risultati
```

### **🏗️ 1. Training Multiclass (Razze)**

```bash
# Preparazione dataset
python src/prepare_data.py --breeds {5,10,30,121}

# FROM SCRATCH (134M parametri)
MODEL_TYPE=full USE_TL=0 python src/train.py --breeds 30

# TRANSFER LEARNING (61K parametri)
USE_TL=1 python src/train.py --breeds 30
```

### **🐕 2. Training Binario "È MAGGIE?"**

```bash
# Preparazione dati binari (346 immagini: 181 Maggie + 165 others)
python src/prepare_data.py --binary

# Training modello binario (71.7% test accuracy)
python src/my_dog_train.py
```

### **🔮 3. Valutazione & Predizione**

```bash
# Valutazione modelli
python src/evaluate.py --model outputs/models/breeds_30/best_model.pth

# Predizione universale (auto-detect + cascade intelligente)
python predict.py dog.jpg outputs/models/breeds_10/best_model.pth --binary-model outputs/my_dog/best_model.pth
```

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

#### **2. SimpleBreedClassifier (FROM SCRATCH - 3.3M parametri)**

```python
# Architettura CNN leggera per test rapidi e training binario
Input: (batch, 3, 224, 224)
├── 3 Blocchi Convoluzionali: 3→32→64→128 channels
├── Batch Normalization + Dropout2D (0.3)
├── 2 Layer Fully Connected: 2048→512→classes
└── Output: (batch, num_classes) logits

# Utilizzo: Training binario "È MAGGIE?" e test rapidi
# Comando: MODEL_TYPE=simple python src/train.py --breeds 5
```

#### **3. Transfer Learning (ResNet18 + Custom Head)**

```python
# Backbone pre-addestrato + classificatore custom
ResNet18(ImageNet) → freeze_backbone=True
├── Backbone congelato: ~11M parametri (non trainable)
├── Custom head: Dropout + Linear(512→classes)
└── Solo ~61K parametri trainable
```

### **Confronto Architetture**

| **Architettura**          | **Parametri**          | **Utilizzo Principale**       | **Performance (5 razze)** | **Training Time** |
| ------------------------- | ---------------------- | ----------------------------- | ------------------------- | ----------------- |
| **BreedClassifier**       | 134M                   | FROM SCRATCH completo         | 20.95%                    | 2-3 ore           |
| **SimpleBreedClassifier** | 3.3M                   | Training binario, test rapidi | 46.67%                    | 30-45 min         |
| **Transfer Learning**     | 11.7M (~61K trainable) | Classificazione multiclass    | **98.1%**                 | 15-30 min         |

**Insight:** SimpleBreedClassifier trova il **sweet spot** per dataset piccoli, mentre Transfer Learning domina per performance.
