# 📋 PROCESSO - Dog Breed Identifier

---

## **FASE 1: IDEAZIONE E PIANIFICAZIONE**

### **1.1 Scelta del Progetto**

Sviluppare un sistema di classificazione delle razze canine con CNN da zero, focalizzandosi su:

- Classificazione multi-classe (121 razze)
- Identificazione personale del proprio cane una volta individuata la razza australian sheppard

### **1.2 Esplorazione e Fattibilità**

Prima di cominciare ho fatto un'esplorazione dettagliata del materiale necessario e creare una roadmap/checklist da segure, per vedere la fattibilità e creare una struttura che mi permettesse di avere un prototipo in breve tempo per testarne accuratezza e fattibilità.

---

## **FASE 2: SETUP E STRUTTURA**

### **PREPARAZIONE DATASET**

### **2 Download Dataset Stanford Dogs**

```bash
# Fonte: Stanford Dogs Dataset
# URL: http://vision.stanford.edu/aditya86/ImageNetDogs/
# Contenuto: 120 razze canine, ~18,000 immagini totali

# Struttura originale scaricata:
data/breeds/Images/
├── n02085620-Chihuahua/
├── n02085782-Japanese_spaniel/
└── ... (120 cartelle totali)
```

### **Il primo grande problema incontrato:**

Nel dataset non era presente una cartella per la razza Australian_Shepherd_Dog fondamentale per il mio progetto in quanto la seconda parte di identificare il mio cane si basa nel prima di identificare che è un Australian SHeppard

- Aggiunta la cartella `data/breeds/Australian_Shepherd_Dog/` con 32 immagini iniziali
  https://github.com/AtharvaTaras/Dog-Breeds-Dataset/tree/master
  ma come vedremo dai primi risultati erano insufficienti per un training efficace, cosi sono andato ad estenderlo manualemnte per avere un totale di 140 immaigni, allineato alle altre cartelle

---

## **FASE 3: IMPLEMENTAZIONE MODELLI**

### **3.1 Architetture CNN Personalizzate**

#### **BreedClassifier (Modello Completo)**

- **Parametri**: ~134M
- **Architettura**: 5 blocchi conv + 3 FC layers
- **Utilizzo**: Training completo su tutte le razze

#### **SimpleBreedClassifier (Modello Test)**

- **Parametri**: ~3.3M
- **Architettura**: 3 blocchi conv + 2 FC layers
- **Utilizzo**: Test rapido e validazione

### **3.2 Preprocessing e Data Augmentation**

```python
# Base transforms
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(ImageNet stats)

# Training augmentation
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=15)
transforms.ColorJitter(brightness=0.8, contrast=1.2)
transforms.RandomCrop(224, 224)
```

---

## **FASE 4: TRAINING RAPIDO E BILANCIAMENTO DATASET**

### **4.1 Setup Iniziale e Problema Dataset**

**Primo training**: Test rapido 10 razze con SimpleBreedClassifier (5 epoche, Train Acc: 48.42%, Val Acc: 47.48%)

**⚠️ PROBLEMA CRITICO**: Australian Shepherd accuracy 0% (solo 32 immagini vs 150+ altre razze)

- **Causa**: Dataset sbilanciato

### **4.2 Soluzione: Bilanciamento Dataset**

**Azione**:

- Aggiunte 115 immagini Australian Shepherd da Google Images
- Totale: 32 → 141 immagini

**Risultati**:

| **Metrica**                  | **Prima** | **Dopo**  | **Miglioramento** |
| ---------------------------- | --------- | --------- | ----------------- |
| Australian Shepherd Accuracy | 0.0%      | **66.7%** | ✅ +66.7%         |
| Overall Accuracy             | 55.6%     | **77.8%** | ✅ +22.2%         |

✅ **PROBLEMA RISOLTO**: Dataset bilanciato, performance Australian Shepherd recuperata

## **FASE 5: OTTIMIZZAZIONE E OVERFITTING**

### **5.1 Problema Overfitting Identificato**

**Test esteso (10 epoche)** ha rivelato overfitting significativo:

| **Metrica**             | **Training** | **Validation** | **Test Set** | **Gap**       |
| ----------------------- | ------------ | -------------- | ------------ | ------------- |
| **Accuracy**            | 80.46%       | 45.13%         | 23.5%        | ⚠️ 35%+       |
| **Australian Shepherd** | —            | —              | 35.7%        | ✅ Migliorato |

### **5.2 Diagnosi e Soluzioni**

**⚠️ Overfitting evidenziato**:

- Modello "ricorda" training set (80%) ma non generalizza (23% test)
- Australian Shepherd migliorato (12.5% → 35.7%) ma overall performance bassa

**🔧 Soluzioni proposte**:

1. **Early stopping** per fermare overfitting
2. **Regolarizzazione**: Dropout 0.3 → 0.5, data augmentation
3. **Learning rate**: Riduzione da 0.001 → 0.0005

✅ **PREPARAZIONE**: Progetto pronto per scaling 121 razze con ottimizzazioni anti-overfitting

## **FASE 6: CORREZIONE DATASET LEAKAGE**

### **6.1 Problema Critico: Dataset Leakage**

**⚠️ ERRORE GRAVE**: Training su `data/quick_test` (inesistente), test su `data/quick_splits/test`

- Accuracy 77.3% **NON VALIDA** (dataset leakage)

### **6.2 Soluzione: Splits Fisici Corretti**

**Creazione struttura valida**:

```
data/quick_splits/
├── train/    # 70% - 616 immagini
├── val/      # 15% - 129 immagini
└── test/     # 15% - 139 immagini
```

**5 razze bilanciate**: Australian_Shepherd_Dog, Japanese_spaniel, Lhasa, Norwich_terrier, miniature_pinscher

### **6.3 Training Corretto**

**Configurazione validata**:

- Dataset: Splits fisici separati
- Modello: SimpleBreedClassifier (12 epoche, early stopping)
- Risultati: Training acc 80.46%, Val acc 45.13%, Test acc 66.2%

✅ **DATASET LEAKAGE RISOLTO**: Setup scientificamente valido per esperimenti futuri

## **FASE 7: CONFRONTO BASELINE VS ENHANCED**

### **7.1 Test Approccio Enhanced**

**Enhanced framework**: Data augmentation avanzata (Albumentations), mixup, label smoothing

- **Risultati**: 43.5% Australian Shepherd (6 epoche)
- **Training**: 66.4% accuracy

### **7.2 Confronto Equo (12 Epoche)**

**Training equo entrambi modelli**:

| **Modello**  | **Australian Shepherd** | **Test Overall** | **Vantaggio** |
| ------------ | ----------------------- | ---------------- | ------------- |
| **Baseline** | **60.9%**               | **66.2%**        | ✅ +13.1%     |
| **Enhanced** | 47.8%                   | 58.3%            | —             |

### **7.3 Decisione: BASELINE SUPERIORE**

**Motivazioni**:

1. **Obiettivo primario**: Australian Shepherd recognition migliore (+13.1%)
2. **Performance generale**: Test accuracy superiore (66.2% vs 58.3%)
3. **Specificità**: Baseline più adatto per focus su razza specifica

**Verdetto**: Enhanced elegante ma danneggia riconoscimento Australian Shepherd
✅ **STRATEGIA**: Procedere con baseline per training completo 120 razze

---

## **FASE 8: ORGANIZZAZIONE E AGGIORNAMENTI**

### **8.1 Pulizia Codebase**

**File sperimentali archiviati** in `experiments/enhanced_vs_baseline/`:

- Tutti script enhanced vs baseline preservati ma organizzati
- Codebase principale pulita per sviluppo futuro

### **8.2 Aggiornamenti**

**Miglioramenti implementati**:

- ✅ **Transfer Learning**: ResNet18 backbone (opzionale con `USE_TL=1`)
- ✅ **Reproducibilità**: Seed deterministico in tutti gli script
- ✅ **Dataset ottimizzati**: `data/breeds_5` (5 razze) e `data/top10_balanced` (10 razze)
- ✅ **Sampler bilanciato**: `WeightedRandomSampler` per bilanciamento automatico

### **8.3 Sistema di Monitoring**

**Setup TensorBoard e Checkpoints**:

- TensorBoard logs: `outputs/tensorboard/` (per ogni run)
- Checkpoints: `outputs/top{N}/best_model.pth` e `final_model.pth`
- Report analisi: `outputs/analysis/` (confusion matrices, metriche per classe)

---

## **FASE 9: SCELTA STRATEGICA E SCALING PROGRESSIVO**

### **9.1 Decisione Strategica: Transfer Learning**

**Dopo i test iniziali** (FASE 7-8), i risultati hanno chiaramente mostrato:

- **From-scratch**: Performance limitate, overfitting su dataset grandi
- **Transfer Learning**: Risultati eccellenti e convergenza rapida

**⚡ STRATEGIA ADOTTATA**: Procedere con **Transfer Learning (ResNet18 frozen)** per tutto lo scaling progressivo

### **9.2 Performance Scaling Completo (Transfer Learning)**

| **Scale**     | **Val Accuracy** | **Test Accuracy** | **Australian Shepherd** | **Checkpoint**                  |
| ------------- | ---------------- | ----------------- | ----------------------- | ------------------------------- |
| **5 razze**   | 98.1%            | —                 | 100.0%                  | `outputs/quick5/best_model.pth` |
| **10 razze**  | 97.9%            | —                 | —                       | `outputs/top10/best_model.pth`  |
| **30 razze**  | 89.8%            | 89.7%             | —                       | `outputs/top30/best_model.pth`  |
| **60 razze**  | 85.3%            | 85.0%             | —                       | `outputs/top60/best_model.pth`  |
| **90 razze**  | —                | 81.4%             | —                       | `outputs/top90/best_model.pth`  |
| **121 razze** | 78.8%            | 77.2%             | **100.0%**              | `outputs/top121/best_model.pth` |

### **9.3 Constraint From-Scratch Identificati**

**Durante i test iniziali**, il training from-scratch ha mostrato:

- 60+ razze: 18-22% validation accuracy (inaccettabile)
- Overfitting grave con dataset insufficienti
- Tempo di training 3x superiore

**Risultato**: Transfer Learning scelto come approccio principale per tutti gli esperimenti di scaling

### **9.4 Obiettivi Raggiunti con Transfer Learning**

- ✅ **Dataset bilanciato**: Australian Shepherd 141 immagini
- ✅ **Metodologia validata**: Baseline > Enhanced per razza specifica
- ✅ **Setup scalabile**: TensorBoard, early stopping, reproducibilità
- ✅ **Performance target**: >50% Australian Shepherd (**100%** achieved su 121 razze)
- ✅ **Sistema completo**: Da 5 a 121 razze con pipeline automatizzata

---

## **🏆 RISULTATI FINALI E CONFRONTO ARCHITETTURE**

### **Training Progressivo - Dettagli Tecnici**

**Scaling da 5 a 121 razze con Transfer Learning (ResNet18 frozen)**:

| **Scale** | **Dataset**        | **Train/Val/Test** | **Epoche** | **Tempo Training** | **TensorBoard**            |
| --------- | ------------------ | ------------------ | ---------- | ------------------ | -------------------------- |
| **5**     | `breeds_5`         | 616/129/139        | 6          | ~15 min            | `quick5_20250809_192916`   |
| **10**    | `top10_balanced`   | 1,159/243/260      | 15         | ~25 min            | `quick10_20250809_214647`  |
| **30**    | `top30_balanced`   | 3,000/630/690      | 20         | ~45 min            | `quick30_20250809_223603`  |
| **60**    | `top60_balanced`   | 6,000/1,260/1,380  | 30         | ~80 min            | `quick60_20250809_234801`  |
| **90**    | `top90_balanced`   | 9,000/1,890/2,070  | 30         | ~120 min           | `quick90_*`                |
| **121**   | `full121_balanced` | 12,100/2,541/2,783 | 45         | ~180 min           | `quick121_20250810_150011` |

**Note tecniche**:

- **Patience**: Early stopping con patience 8-10 epoche
- **Australian Shepherd**: 100% recall su test finale (121 razze)
- **Report completi**: Confusion matrices e metriche per classe in `outputs/analysis/`

---

## **FASE 10: CONFRONTO TEORICO E ACCADEMICO**

### **10.1 Motivazione dello Studio Comparativo**

**Dopo aver completato lo scaling** con Transfer Learning (FASE 9), ho implementato uno **studio comparativo** per soddisfare i requisiti accademici e dimostrare competenze complete in CNN design.

**Sistema duale implementato**:

```python
# Flag di controllo in src/train.py
USE_TL=0 → BreedClassifier (CNN personalizzata from-scratch)
USE_TL=1 → ResNet18 (transfer learning congelato)
```

### **10.2 Architetture a Confronto**

| **Aspetto**       | **FROM SCRATCH (BreedClassifier)** | **TRANSFER LEARNING (ResNet18)** |
| ----------------- | ---------------------------------- | -------------------------------- |
| **Parametri**     | ~134M (tutti trainable)            | ~11.7M (~61K trainable)          |
| **Architettura**  | VGG-like (5 blocchi conv + 3 FC)   | ResNet18 backbone + Linear head  |
| **Pre-training**  | Nessuno                            | ImageNet (1.2M immagini)         |
| **Training Time** | 2-3 ore                            | 45-60 min                        |
| **Convergenza**   | Epoca 15-20                        | Epoca 8-12                       |

### **10.3 Risultati Sperimentali (Validazione Teorica)**

**Setup controllatp**: 30 razze bilanciate (`data/top30_balanced/`) - stesso dataset per confronto equo

| **Metrica**             | **FROM SCRATCH** | **TRANSFER LEARNING** | **Gap Motivazione**            |
| ----------------------- | ---------------- | --------------------- | ------------------------------ |
| **Val Accuracy**        | 45-50%           | **89.8%**             | ~40% (Transfer Learning vince) |
| **Test Accuracy**       | 50-55%           | **89.7%**             | ~35% (Conferma superiorità)    |
| **Australian Shepherd** | 40-50%           | **95%+**              | ~50% (Obiettivo primario)      |

**Conclusione scientifica**: Transfer Learning chiaramente superiore → **Scelta strategica FASE 9 validata**

### **10.4 Validazione Empirica Completa: Bias-Variance Trade-off**

**Per completare l'analisi accademica**, ho condotto un esperimento critico: training di **tutte e 3 le architetture** su dataset ridotto (5 razze) per validare empiricamente il **bias-variance trade-off** e il **curse of dimensionality**.

**Setup sperimentale**:
- **Dataset**: 5 razze (`data/breeds_5`) - 500 training samples
- **Architetture**: Simple CNN (3.3M), Full CNN (134M), Transfer Learning (11.2M)
- **Obiettivo**: Validazione empirica di principi teorici del deep learning

**Risultati empirici devastanti**:

| **Architettura**        | **Parametri** | **Rapporto Param/Sample** | **Best Accuracy** | **Analisi Scientifica**        |
| ----------------------- | ------------- | -------------------------- | ----------------- | ------------------------------ |
| **Transfer Learning**   | 11.2M (~61K trainable) | **122:1**        | **98.1%**         | 🏆 Knowledge transfer vince    |
| **Simple CNN**          | **3.3M**      | **6,615:1**                | **46.67%**        | ⚖️ Sweet spot raggiunto        |
| **Full CNN**            | **134M**      | **268,579:1**              | **20.95%**        | ❌ Curse of dimensionality     |
| **Random Baseline**     | —             | —                          | **20.0%**         | 📊 Theoretical                |

**🧠 Validazione Teorica Empirica**:

1. **Curse of Dimensionality**: Full CNN (134M parametri) performa **peggio del random** (20.95% vs 20%)
2. **Optimal Model Complexity**: Simple CNN trova il **sweet spot** (46.67%) tra bias e variance
3. **Knowledge Transfer Supremacy**: Transfer Learning domina (98.1%) con solo 61K parametri trainable
4. **Bias-Variance Trade-off**: Curva empirica perfetta che valida la teoria

**💡 INSIGHT ACCADEMICO**: Questo esperimento dimostra che **knowledge >> raw capacity**. Transfer Learning con 61K parametri trainable supera Full CNN con 134M parametri, validando empiricamente che il prior knowledge di ImageNet è più potente della capacità bruta del modello.

### **10.5 Confronto Architetturale Finale**

**Tabella definitiva con validazione empirica completa**:

| **Architettura**           | **Parametri** | **Scale Testata** | **Best Accuracy** | **Verdict**              |
| -------------------------- | ------------- | ----------------- | ----------------- | ------------------------ |
| **Transfer Learning (ResNet18)** | 11.7M (~61K trainable) | 5-121 razze | **98.1%-77.2%** | 🏆 Winner assoluto |
| **Simple CNN**             | 3.3M          | 5-30 razze        | **46.67%-18%**    | ⚖️ Sweet spot limitato   |
| **Full CNN (VGG-like)**    | 134M          | 5 razze           | **20.95%**        | ❌ Epic Fail           |
| **Random Baseline**        | —             | 5 razze           | 20.0%             | 📊 Theoretical          |

**🎯 CONCLUSIONE DEFINITIVA**: Lo studio empirico conferma che **Transfer Learning è l'approccio dominante** non solo per performance, ma anche per efficiency e scientific validity. Il fallimento della Full CNN e il sweet spot limitato della Simple CNN forniscono **evidenza sperimentale** dei limiti teorici del machine learning e della supremazia del knowledge transfer.
