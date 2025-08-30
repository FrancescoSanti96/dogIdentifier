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

## � **Struttura Progetto**

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

## 🚀 Quick Start

```bash
# 1. Setup ambiente
source .venv/bin/activate && pip install -r requirements.txt

# 2. Preparazione dati
python src/prepare_data.py --breeds 30

# 3. Training FROM SCRATCH (requisito professore)
MODEL_TYPE=full USE_TL=0 python src/train.py --breeds 30

# 4. Training TRANSFER LEARNING (confronto)
USE_TL=1 python src/train.py --breeds 30

# 5. Training binario "È MAGGIE?"
python src/prepare_data.py --binary && python src/my_dog_train.py

# 6. Predizione universale
python predict.py image.jpg outputs/models/breeds_30/best_model.pth
```

## � **Sistema di Predizione Universale**

### **Universal Dog Classifier - Un solo script per tutto! 🚀**

Il nostro `predict.py` è un sistema intelligente che **auto-detecta** il tipo di modello e adatta automaticamente il comportamento:

#### **📖 Sintassi base:**
```bash
python predict.py <immagine> <modello> [opzioni]
```

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

#### **🔄 Esempi pratici:**

**Australian Shepherd** → Triggera test Maggie:
```bash
python predict.py data/breeds_5/test/Australian_Shepherd_Dog/n02096294_3576.jpg \
    outputs/models/breeds_10/best_model.pth --binary-model outputs/my_dog/best_model.pth
# Risultato: Australian Shepherd 60.92% → Auto-trigger → "È MAGGIE!" 65.6%
```

**Chihuahua** → NON triggera test:
```bash
python predict.py data/breeds_5/test/Chihuahua/n02085620_10976.jpg \
    outputs/models/breeds_10/best_model.pth --binary-model outputs/my_dog/best_model.pth
# Risultato: Pomeranian 65.92% → Fine (nessun trigger)
```

#### **🧠 Intelligenza automatica:**
- ✅ **Auto-detecta** il tipo di modello (Transfer Learning vs Binario)
- ✅ **Cascade intelligente**: Razze → Se Australian Shepherd → Test Maggie
- ✅ **Evita test inutili**: Se non trova Australian Shepherd, non fa test binario
- ✅ **Interfaccia unificata**: Un solo script per tutto

## �🔬 **Studio Comparativo: FROM SCRATCH vs TRANSFER LEARNING**

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

## 📋 **Workflow Completo**

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

## 📊 **Risultati Consolidati**

### **Transfer Learning Performance**

| Razze   | Val Acc   | Top-5   | Test Acc  | Australian Shepherd | Status |
| ------- | --------- | ------- | --------- | ------------------- | ------ |
| 5       | 98.1%     | 99%     | ~95%      | 100%                | ✅     |
| 10      | 97.9%     | 99%     | ~95%      | 100%                | ✅     |
| 30      | 89.8%     | 97%     | 89.7%     | 100%                | ✅     |
| **121** | **78.8%** | **97%** | **77.2%** | **100%**            | ✅     |

### **Modello Binario "È MAGGIE?"**

| Metrica        | Risultato | Note                           |
| -------------- | --------- | ------------------------------ |
| **Test Acc**   | **71.7%** | 346 immagini (181+165)        |
| **Dataset**    | Balanced  | 70/15/15 train/val/test split  |
| **Architettura** | SimpleBreedClassifier | 3.3M parametri      |
| **Use Case**   | Australian Shepherd → Test Maggie | Cascade automatica |

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
# Setup rapido
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Alternativa conda
conda env create -f environment.yml && conda activate dogidentifier
```

---

## 🏆 **Highlights Progetto**

- ✅ **Confronto scientifico** FROM SCRATCH vs Transfer Learning (4.16x gap)
- ✅ **Sistema predizione universale** con auto-detection e cascade intelligente  
- ✅ **Australian Shepherd Focus** con 100% recall su 121 razze
- ✅ **Training binario personalizzato** "È MAGGIE?" (71.7% accuracy)
- ✅ **Architetture multiple** testate: Simple CNN, VGG-like, ResNet18
