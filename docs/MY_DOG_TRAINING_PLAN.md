# 📋 PIANO TRAINING BINARIO "È MAGGIE?" - SISTEMA COERENTE
*Environment Variables + Profiles (coerente con train.py)*

## 🎯 **OBIETTIVO**
Completare il training del classificatore binario "Il mio cane (Maggie) vs Altri cani" seguendo esattamente la metodologia documentata nel PROCESSO.md FASE 11, usando il **sistema Environment Variables** coerente con `train.py`.

---

## **🚀 SEQUENZA TRAINING - SINTASSI COERENTE**

### **✅ SETUP COMPLETATO**
```bash
# Dataset già ridotto a 261 immagini (vicino al target 272)
find data/my_dog_vs_others -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l
# Output: 261 - PRONTO PER TRAINING
```

### **FASE 1: Dataset Ridotto - Ricerca Hyperparameters**

#### **1️⃣ TRAINING BASELINE (PROCESSO.md FASE 11.2)**
```bash
USE_PROFILE=binary_baseline python src/my_dog_train.py
```
**🎯 Target**: Val 75.0%, Test 59.5%, Gap 17.5% (overfitting critico)  
**⚙️ Config**: epochs=20, lr=0.0005, dropout=0.3, batch=16

#### **2️⃣ TRAINING OTTIMIZZATA (PROCESSO.md FASE 11.3)**  
```bash
USE_PROFILE=binary_optimized python src/my_dog_train.py
```
**🎯 Target**: Val 80.0%, Test **71.4%**, Gap 8.6% (🏆 sweet spot)  
**⚙️ Config**: epochs=30, lr=0.0003, dropout=0.5, batch=12

#### **3️⃣ TRAINING ULTRA-AGGRESSIVA (PROCESSO.md FASE 11.6)**
```bash
USE_PROFILE=binary_aggressive python src/my_dog_train.py
```
**🎯 Target**: Val **90.0%**, Test 69.1%, Gap 14.9% (validation overfitting)  
**⚙️ Config**: epochs=25, lr=0.0003, dropout=0.6, batch=10

### **FASE 2: Dataset Esteso - Training Finale**

#### **📦 RIPRISTINO DATASET COMPLETO**
```bash
# Ripristina dataset esteso dal backup automatico
python utils/dataset_simulator.py --restore

# Verifica ripristino (torna a ~343 immagini)
find data/my_dog_vs_others -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l
```

#### **4️⃣ TRAINING FINALE (PROCESSO.md FASE 11.9)**
```bash
USE_PROFILE=binary_final python src/my_dog_train.py
```
**🎯 Target**: Test **71.7%** (robustezza confermata)  
**⚙️ Config**: epochs=25, lr=0.0003, dropout=0.5, batch=12

---

## **🔧 SISTEMA COERENTE CON TRAIN.PY**

### **Sintassi Environment Variables**
```bash
# Stesso pattern di train.py
USE_PROFILE=binary_baseline python src/my_dog_train.py
USE_PROFILE=breeds_10 python src/train.py --breeds 10

# Override parametri specifici
USE_PROFILE=binary_baseline EPOCHS=25 python src/my_dog_train.py
```

### **TensorBoard Monitoring**
```bash
# Monitoring separato per training binario
python scripts/launch_tensorboard.py --mydog
```

### **Profiles Disponibili nel config.json**
| Profile | Epochs | LR | Dropout | Batch | Descrizione |
|---------|--------|----|---------|---------| -------------|
| `binary_baseline` | 20 | 0.0005 | 0.3 | 16 | FASE 11.2 - Conservativa |
| `binary_optimized` | 30 | 0.0003 | 0.5 | 12 | FASE 11.3 - Sweet spot |
| `binary_aggressive` | 25 | 0.0003 | 0.6 | 10 | FASE 11.6 - Ultra-aggressiva |
| `binary_final` | 25 | 0.0003 | 0.5 | 12 | FASE 11.9 - Formula vincente |

---

## **🎯 CHECKLIST ESECUZIONE**

### **FASE 1: Dataset Ridotto (261 img)**
- [ ] **Training 1 - Baseline** (`USE_PROFILE=binary_baseline`) → Target: Test 59.5%
- [ ] **Training 2 - Ottimizzata** (`USE_PROFILE=binary_optimized`) → Target: Test 71.4% 🏆
- [ ] **Training 3 - Ultra-Aggressiva** (`USE_PROFILE=binary_aggressive`) → Target: Test 69.1%

### **FASE 2: Dataset Esteso (343 img)**  
- [ ] **Ripristino Dataset** (`python utils/dataset_simulator.py --restore`)
- [ ] **Training 4 - Finale** (`USE_PROFILE=binary_final`) → Target: Test **71.7%** ✅

### **Validazione:**
- [ ] TensorBoard monitoring attivo (`scripts/launch_tensorboard.py --mydog`)
- [ ] Progressione hyperparameters documentata
- [ ] Performance target raggiunte
- [ ] Robustezza confermata su dataset crescenti

---

## **✅ VANTAGGI SISTEMA COERENTE**
🔄 **Pattern Uniforme**: Stesso sistema per `train.py` e `my_dog_train.py`  
⚡ **Velocità**: `USE_PROFILE=binary_baseline python src/my_dog_train.py`  
🎯 **Flessibilità**: Environment variables per override rapidi  
📁 **Centralizzazione**: Tutti i profiles nel config.json  
🔧 **Scalabilità**: Facile aggiungere nuovi profiles  

---

## **🏆 TARGET FINALE**
**Replicare esattamente il risultato del PROCESSO.md**: **71.7% test accuracy** con robustezza empiricamente validata su 4 configurazioni diverse.

**🚀 PROSSIMO STEP**: 
```bash
USE_PROFILE=binary_baseline python src/my_dog_train.py
```

Iniziamo con il sistema coerente! 🎯
