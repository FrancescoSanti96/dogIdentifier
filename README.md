# 🐕 Dog Breed Identifier

Sistema di classificazione razze canine (5→121) + identificazione del cane personale. Progetto universitario con codebase pulita e script unificati.

## 🚀 Quick Start

```bash
# 1. Attiva ambiente virtuale (RICHIESTO)
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Training completo (esempio 121 razze)
python src/prepare_data.py --breeds 121
python src/train.py --breeds 121

# 4. Valutazione
python src/evaluate.py --model outputs/models/breeds_121/best_model.pth --data data/full121_balanced

# 5. TensorBoard
python scripts/launch_tensorboard.py
```

## 📋 Comandi principali

```bash
# Preparazione dataset
python src/prepare_data.py --breeds {10,30,121}

# Training unificato
python src/train.py --breeds {5,10,30,60,90,121}

# Valutazione modelli
python src/evaluate.py --model MODEL --data DATA --outdir OUTPUT

# Fase 2: cane personale
python src/my_dog_train.py
```

## 📁 Struttura

```
src/          # Script unificati (train.py, prepare_data.py, evaluate.py)
outputs/
  results/    # Risultati finali per consegna
  models/     # Modelli addestrati (breeds_N/)
  analysis/   # Analisi dettagliate
docs/         # PROCESSO.md (cronologia esperimenti)
```

## 📊 Risultati

| Razze   | Val Acc | Top-5    | Status |
| ------- | ------- | -------- | ------ |
| 5       | ~98%    | ~95%     | ✅     |
| 10      | ~98%    | ~97%     | ✅     |
| 30      | ~90%    | ~97%     | ✅     |
| 60      | ~85%    | ~97%     | ✅     |
| 90      | ~81%    | ~98%     | ✅     |
| **121** | **79%** | **~97%** | ✅     |

## ⚙️ Setup ambiente (prima esecuzione)

### Opzione 1: venv (Python standard)

```bash
# Crea ambiente virtuale
python -m venv .venv

# Attiva ambiente (sempre necessario)
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt
```

### Opzione 2: conda (alternativa)

```bash
# Crea ambiente conda
conda env create -f environment.yml

# Attiva ambiente (sempre necessario)
conda activate dogidentifier
```

## 🔗 Link rapidi

- **Risultati finali**: `outputs/results/project_summary.md`
- **Processo completo**: `docs/PROCESSO.md`
- **Script legacy**: `experiments/legacy_scripts/` (opzionali)
