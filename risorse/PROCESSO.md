# 📋 PROCESSO - Dog Breed Identifier

## 🎯 **Obiettivo**
Sviluppare un sistema di classificazione delle razze canine con CNN da zero, focalizzandosi su:
- Classificazione multi-classe (120+ razze)
- Identificazione personale del proprio cane una volta individuata la razza australian sheppard
- Implementazione completa senza modelli pre-addestrati

---

## **FASE 1: IDEAZIONE E PIANIFICAZIONE**

### **1.1 Scelta del Progetto**
Il decidere il progetto non era semplice ero indeciso tra due macro tematiche da eplorare:
- **Sentiment Analysis**: Avrei voluto sperimentare argomenti avanzanti ma per il mio obbiettivo non avevo le competenze neccessarie e il tempo per poter effettuare delle prove dato che era difficile che avrei raggiunto un risulato accettabile.
- **Dog Breed Recognition**: Più pratico e visibile fin da subito potevo massimizzare maggioranente il tempo per avere un mvp veloce per poter avere subito un idea del successo o meno della rete neurale, inoltre ocmunque era stimolante creare un pattern doppio.

**Scelta finale**: Dog Breed Recognition

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
├── n02085936-Maltese_dog/
├── n02086079-Pekinese/
├── n02086240-Shih-Tzu/
└── ... (120 cartelle totali)
```


### **Il primo grande problema incontrato:**
Nel dataset non era presente una cartella per la razza Australian_Shepherd_Dog fondamentale per il mio progetto in quanto la seocnda parte di identificare il mio cane si basa nel prima di identificare che è un Australian SHeppard

- Aggiunta la cartella `data/breeds/Australian_Shepherd_Dog/` con 32 immagini, ma come vedromo da i primi risultati insufficienti per un training efficace.
<!-- TODO da dove l'ho presa
 -->
### **2.1 Creazione Struttura Directory**
```bash
dogIdentifier_clean/
├── config.json              # Configurazione
├── requirements.txt          # Dipendenze
├── data/breeds/             # Dataset (120 razze)
├── models/                  # CNN personalizzate
├── utils/                   # Utility e preprocessing
├── train/                   # Script training
├── inference/               # Script predizione
├── test/                    # Test di validazione
└── outputs/                 # Risultati e checkpoint
```

### **2.2 Installazione Dipendenze**
```bash
torch>=1.9.0, torchvision>=0.10.0, numpy>=1.21.0,
matplotlib>=3.4.0, seaborn>=0.11.0, pandas>=1.3.0,
Pillow>=8.3.0, scikit-learn>=1.0.0, tensorboard>=2.7.0,
opencv-python>=4.5.0, albumentations>=1.1.0, tqdm>=4.62.0
```

### **2.3 Configurazione Iniziale**
Creato `config.json` con parametri per:
- Path dataset e immagini
- Hyperparameters modello
- Configurazione training
- Data augmentation

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

## **FASE 4: TRAINING E VALIDAZIONE**

### **4.1 Training Rapido (Quick Training)**
**Configurazione test rapido:**
- **Dataset**: Prime 10 razze (1,301 immagini training)
- **Modello**: SimpleBreedClassifier
- **Epoche**: 5
- **Batch size**: 32
- **Learning rate**: 0.001
- **Device**: CPU

**Risultati training:**
```
Epoch 1/5: Train Acc: 14.14%, Val Acc: 24.82%
Epoch 2/5: Train Acc: 29.98%, Val Acc: 29.86%
Epoch 3/5: Train Acc: 37.20%, Val Acc: 40.29%
Epoch 4/5: Train Acc: 42.43%, Val Acc: 45.68%
Epoch 5/5: Train Acc: 48.42%, Val Acc: 47.48%

✅ Training completato!
📁 Modello salvato in: outputs/quick_test/quick_model.pth
🎯 Accuracy finale: Train 48.42%, Val 47.48%
```

### **4.2 Test di Validazione Post-Training**
Creato `test_validation.py` per testare il modello su diverse razze:

**Test su 3 razze campione:**
```
🧪 Test di Validazione Progetto
==================================================
✅ Modello caricato
📊 Accuracy training: 48.36%
📊 Accuracy validation: 42.53%

🔍 Testando 3 razze in modalità 'sample'...

🐕 Testando Australian_Shepherd_Dog:
  ❌ Immagine 1: 8.1% confidence
  ❌ Immagine 2: 5.1% confidence  
  ❌ Immagine 3: 7.4% confidence
  📊 Accuracy Australian_Shepherd_Dog: 0.0% (0/3)

🐕 Testando Afghan_hound:
  ✅ Immagine 1: 45.0% confidence
  ❌ Immagine 2: 8.7% confidence
  ✅ Immagine 3: 18.5% confidence
  📊 Accuracy Afghan_hound: 66.7% (2/3)

🐕 Testando Bernese_mountain_dog:
  ✅ Immagine 1: 44.2% confidence
  ✅ Immagine 2: 19.7% confidence
  ✅ Immagine 3: 43.7% confidence
  📊 Accuracy Bernese_mountain_dog: 100.0% (3/3)

==================================================
📊 RISULTATI FINALI
==================================================
🎯 Accuracy complessiva: 55.6% (5/9)

📋 Analisi per razza:
  Australian_Shepherd_Dog: 0.0% accuracy, 6.9% avg confidence
  Afghan_hound: 66.7% accuracy, 24.1% avg confidence
  Bernese_mountain_dog: 100.0% accuracy, 35.8% avg confidence

💡 RACCOMANDAZIONE:
  ⚠️  PROSEGUI MA MIGLIORA IL DATASET (Accuracy 55.6%)

🐕 Australian Shepherd Dog:
  Accuracy: 0.0%
  Avg confidence: 6.9%
  ⚠️  Australian Shepherd ha performance bassa - aggiungi più immagini!
```

### **4.3 Problema Identificato**
Il test di validazione ha rivelato un **problema critico**:
- **Australian Shepherd Dog**: 0% accuracy (solo 32 immagini)
- **Altre razze**: 66-100% accuracy (150+ immagini)
- **Causa**: Dataset sbilanciato


<!-- TODO -->
Obbiettivo ribilanciare il dataset cosi da poter rieffettuare un test prima di proceder con il tarining avanzato e poi la rifunitura, perche allo stato attuale questo risultava essere un grande problema ma idealemtne facilmetne risolvibile.

