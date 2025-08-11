# 📚 Legacy Scripts Archive

Questo archivio contiene gli script originali del progetto prima dell'unificazione.

## 🗂️ Struttura

### **📁 training/**

Script di training individuali per ogni scala di razze (sostituiti da `src/train.py`):

- `quick5_tensorboard_train.py` - Training 5 razze baseline
- `quick10_tensorboard_train.py` - Training 10 razze
- `quick30_tensorboard_train.py` - Training 30 razze
- `quick60_tensorboard_train.py` - Training 60 razze
- `quick90_tensorboard_train.py` - Training 90 razze
- `quick121_tensorboard_train.py` - Training 121 razze complete

### **📁 preparation/**

Script di preparazione dataset individuali (sostituiti da `src/prepare_data.py`):

- `prepare_top10_balanced.py` - Preparazione dataset 10 razze
- `prepare_top30_balanced.py` - Preparazione dataset 30 razze
- `prepare_full121_balanced.py` - Preparazione dataset 121 razze

### **📄 config.json**

File di configurazione legacy (sostituito da configurazione integrata negli script unificati).

## 🔄 Utilizzo Script Legacy

### **⚠️ Importante: Eseguire dalla Root del Progetto**

Gli script legacy **funzionano ancora** ma devono essere eseguiti dalla directory root:

```bash
# ✅ CORRETTO - dalla root del progetto
python experiments/legacy_scripts/training/quick121_tensorboard_train.py
python experiments/legacy_scripts/preparation/prepare_full121_balanced.py

# ❌ ERRATO - dalla directory degli script
cd experiments/legacy_scripts/training
python quick121_tensorboard_train.py  # ImportError!
```

### **🔄 Migrazione agli Script Unificati**

### **Prima (Legacy):**

```bash
# Training specifico per scala (dalla root)
python experiments/legacy_scripts/training/quick121_tensorboard_train.py

# Preparazione specifica per scala
python experiments/legacy_scripts/preparation/prepare_full121_balanced.py
```

### **Ora (Unificato):**

```bash
# Training unificato con parametro
python src/train.py --breeds 121

# Preparazione unificata con parametro
python src/prepare_data.py --breeds 121
```

## 📋 Vantaggi dell'Unificazione

1. **🎯 Manutenibilità**: Un solo file da aggiornare invece di 6
2. **🔧 Coerenza**: Stessi parametri ottimali per tutte le scale
3. **📊 Semplicità**: Interfaccia command-line intuitiva
4. **⚡ Configurazione**: Parametri automatici + override ambiente

## 🔗 Riferimenti

- **Script Unificati**: `src/train.py`, `src/prepare_data.py`, `src/evaluate.py`
- **Documentazione**: `docs/PROCESSO.md` per cronologia completa
- **Risultati**: `outputs/` per modelli e analisi

---

_Archivio creato durante la ristrutturazione del progetto per la consegna universitaria._
