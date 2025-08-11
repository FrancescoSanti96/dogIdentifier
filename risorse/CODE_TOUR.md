# Code Tour tecnico del progetto DogIdentifier

Questo documento guida passo‑passo nella lettura del codice, spiegando dove si trovano le componenti chiave, come fluiscono i dati e come riprodurre i risultati (training, valutazione, logging).

## Indice rapido

- Architettura e mappa file
- Pipeline di training moderna (121 classi)
- Modelli e Transfer Learning / Fine‑Tuning
- Dati, trasformazioni e sampler
- Riproducibilità, early stopping e logging
- Valutazione e report (confusion matrix, CSV)
- Variabili d’ambiente e comandi utili
- Troubleshooting e best practice
- Appendice: tour per scale (5/10/30/60/121) e riferimenti

---

## Architettura e mappa file

Struttura di alto livello (cartelle più rilevanti):

- `models/` – Modelli CNN e factory
  - `models/breed_classifier.py`: definisce i modelli (full/simple) e la factory per ResNet18 (TL)
- `utils/` – Utilità riusabili
  - `utils/dataloader.py`: creazione `DataLoader`, trasformazioni, WeightedRandomSampler
  - `utils/seed_utils.py`: funzioni per determinismo
  - `utils/early_stopping.py`: early stopping su validation loss
  - `utils/config_helper.py`: configurazione augmentation
- Script di training (per diverse scale)
  - `quick5_tensorboard_train.py`, `quick10_tensorboard_train.py`, `quick30_tensorboard_train.py`, `quick60_tensorboard_train.py`, `quick121_tensorboard_train.py`
- Valutazione/analisi
  - `analyze_confusion.py`: report di test, confusion matrix, classification report, CSV per‑classe
- Dati e output
  - `data/`: dataset con split fisici bilanciati
  - `outputs/`: checkpoint, log TensorBoard, analisi
- Documentazione
  - `risorse/PROCESSO.md`: diario tecnico/risultati
  - `risorse/CODE_TOUR.md`: questo documento

Suggerimento di lettura: parti da `quick121_tensorboard_train.py`, poi apri `models/breed_classifier.py` e `utils/dataloader.py` in parallelo.

---

## Pipeline di training moderna (121 classi)

File: `quick121_tensorboard_train.py`

Cosa fa:

- Set seed deterministico (`utils/seed_utils.set_deterministic(42)`).
- Inizializza TensorBoard (`SummaryWriter`) e directory log `outputs/tensorboard/quick121_*`.
- Carica dataset e `DataLoader` dagli split (`utils/dataloader.create_dataloaders_from_splits`).
- Costruisce il modello tramite factory: ResNet18 con head sostituita (TL); freeze del backbone; opzionale sblocco di `layer4` per fine‑tuning.
- Loss: `CrossEntropyLoss` con `label_smoothing=0.05` e `class_weights` (se calcolabili).
- Optimizer: `AdamW` con `weight_decay` configurabile.
- Scheduler: `ReduceLROnPlateau` su `val_loss` (mode=min, factor=0.5, patience=2).
- Gradient clipping (`max_norm=1.5`).
- Logging completo su TensorBoard (train/val loss, accuracy, Top‑5, LR, gap train‑val).
- EarlyStopping su `val_loss` con `patience` configurabile.
- Salvataggio `best_model.pth` (migliore Val Acc) e `final_model.pth`.

Parametri principali (via env vars):

- Generali: `EPOCHS`, `BATCH_SIZE`, `LR`, `PATIENCE`, `DROPOUT`, `WD`
- Dati: `SPLITS_DIR` (default `data/full121_balanced`)
- Transfer Learning: `USE_TL=1` (abilita ResNet18 pre‑addestrata)
- Fine‑Tuning selettivo: `UNFREEZE_LAYER4=1` (sblocca solo `layer4` + `fc`)

Note d’uso:

- Criterio di “best” = Val Accuracy; early stop = Val Loss. È una combinazione comune: si mantiene il best per accuratezza mentre si evita overfitting con pazienza su loss.
- Top‑5: già calcolata e loggata.

---

## Modelli e Transfer Learning / Fine‑Tuning

File: `models/breed_classifier.py`

Contiene:

- `BreedClassifier` (full) e `SimpleBreedClassifier` (simple) – CNN from‑scratch.
- `create_breed_classifier(...)` – factory per:
  - Full/Simple CNN from‑scratch; oppure
  - ResNet18 di torchvision (con `weights=IMAGENET1K_V1`) con testa sostituita da `Dropout`+`Linear(num_classes)`.

Freeze/Unfreeze:

- Quando `freeze_backbone=True`, nella factory vengono congelati tutti i parametri che non appartengono alla testa `fc`.
- Lo sblocco selettivo di `layer4` viene applicato nello script di training (non nella factory): se `UNFREEZE_LAYER4=1`, si abilitano in write i parametri `layer4.*` e `fc.*`, congelando tutto il resto.

Consigli FT:

- Per sbloccare `layer4`, usare LR più bassa (es. 1e‑5–5e‑5) rispetto al training della sola head.
- Aumentare di qualche epoca e una pazienza leggermente maggiore per stabilizzare.

---

## Dati, trasformazioni e sampler

File: `utils/dataloader.py`

- Trasformazioni:
  - Train: `RandomResizedCrop` abilitato di default; eventuali flip/rotazioni/color jitter se impostati in `augmentation_config`.
  - Val/Test: `Resize(256)` + `CenterCrop(224)` per evitare distorsioni; `ToTensor` + `Normalize(ImageNet mean/std)`.
- Sampler: `WeightedRandomSampler` attivo tramite flag `use_weighted_sampler=True` per mitigare class imbalance.
- API: `create_dataloaders_from_splits(splits_dir, batch_size, image_size, augmentation_config, use_weighted_sampler)` ritorna `(train_loader, val_loader, test_loader)`.

Suggerimenti:

- Mantieni coerenti le trasformazioni di Val/Test (mai ridimensionamento non isotropico).
- Per classi rari/ostiche, lascia attivo il sampler pesato.

---

## Riproducibilità, early stopping e logging

- `utils/seed_utils.py`: setta `PYTHONHASHSEED`, `random`, `numpy`, `torch` e flag cuDNN per massima ripetibilità.
- `utils/early_stopping.py`: semplice oggetto con memoria del best `val_loss` e conteggio epoche senza miglioramento.
- TensorBoard: lanciabile con `tensorboard --logdir outputs/tensorboard`; vedi directory create dagli script `quick*_tensorboard_train.py`.

Metriche loggate per epoca:

- Train/Val Loss, Accuracy
- Top‑5 Accuracy
- Learning Rate (post scheduler)
- Train‑Val Gap (analisi overfitting)

---

## Valutazione e report (confusion matrix, CSV)

File: `analyze_confusion.py`

Cosa fa:

- Carica un checkpoint `.pth` (auto‑detect di ResNet18 dalla struttura salvata).
- Esegue test su `--data` (split Test) con batch configurabile.
- Salva:
  - `confusion_matrix.png`
  - `confusion_analysis.txt` con per‑classe (accuracy e conteggi) + classification report macro/weighted
  - Per 121 classi: CSV per‑classe nella cartella di analisi (se non presente, si può estrarre dai report)

Esempio d’uso:

```bash
python analyze_confusion.py \
  --model outputs/top121/best_model.pth \
  --data data/full121_balanced \
  --batch-size 64 \
  --outdir outputs/analysis/top121_YYYYMMDD_HHMMSS
```

Output tipici:

- `outputs/analysis/top121_YYYYMMDD_HHMMSS/confusion_matrix.png`
- `outputs/analysis/top121_YYYYMMDD_HHMMSS/confusion_analysis.txt`
- `outputs/analysis/top121_YYYYMMDD_HHMMSS/per_class_metrics.csv` (se generato)

---

## Variabili d’ambiente e comandi utili

Training (esempi generici):

```bash
# 121 classi baseline, TL frozen
USE_TL=1 EPOCHS=45 PATIENCE=10 python quick121_tensorboard_train.py

# 121 classi, fine‑tuning layer4 cauto (valutare bene prima di adottare)
USE_TL=1 UNFREEZE_LAYER4=1 LR=5e-5 EPOCHS=10 PATIENCE=4 python quick121_tensorboard_train.py

# TensorBoard
tensorboard --logdir outputs/tensorboard
```

Valutazione:

```bash
python analyze_confusion.py \
  --model outputs/top121/best_model.pth \
  --data data/full121_balanced \
  --batch-size 64 \
  --outdir outputs/analysis/top121_YYYYMMDD_HHMMSS
```

Parametri comuni:

- `EPOCHS`, `BATCH_SIZE`, `LR`, `PATIENCE`, `DROPOUT`, `WD`
- `SPLITS_DIR` per puntare a dataset diversi (`data/breeds_5`, `data/top10_balanced`, `data/top30_balanced`, `data/top60_balanced`, `data/full121_balanced`)
- `USE_TL`, `UNFREEZE_LAYER4`, `FINETUNE_FROM` (quest’ultimo usato negli script 60/90 per caricare un checkpoint di partenza)

---

## Troubleshooting e best practice

- Gap Train‑Val alto / overfitting:
  - Aumentare `DROPOUT`, usare `label_smoothing`, ridurre LR con `ReduceLROnPlateau`.
  - Verificare che le trasformazioni Val/Test non distorcano (devono essere Resize+CenterCrop).
- Fine‑tuning peggiora:
  - LR troppo alta sul backbone; ridurre a 1e‑5–5e‑5 e aumentare epoche/patience.
  - Considerare LR “discriminativa” (più alta sulla `fc`, più bassa su `layer4`) estendendo lo script con due param groups.
- Classi sotto 50%:
  - Augment mirate su coppie simili (Husky/Malamute/Eskimo; Collie/Shetland; Shih‑Tzu/Lhasa).
  - Eventuale oversampling/addizione dati.
  - Valutare TTA in inference.
- Dataset imbalance:
  - Mantenere `WeightedRandomSampler=True`.

---

## Appendice: tour per scale e riferimenti

Script per scale crescenti (stessa impalcatura):

- `quick5_tensorboard_train.py` – 5 razze (dataset `data/breeds_5`)
- `quick10_tensorboard_train.py` – 10 razze (dataset `data/top10_balanced`)
- `quick30_tensorboard_train.py` – 30 razze (dataset `data/top30_balanced`)
- `quick60_tensorboard_train.py` – 60 razze (dataset `data/top60_balanced`, supporto FT da checkpoint via `FINETUNE_FROM`)
- `quick121_tensorboard_train.py` – 121 razze (full pipeline e Top‑5 logging)

Checkpoint e analisi 121 (baseline consolidata):

- Checkpoint: `outputs/top121/best_model.pth`
- TensorBoard: `outputs/tensorboard/quick121_20250810_150011`
- Report test: `outputs/analysis/top121_20250810_225455/{confusion_matrix.png, confusion_analysis.txt, per_class_metrics.csv}`

Note su fine‑tuning 121 (non adottato):

- FT rapido con `UNFREEZE_LAYER4` e 5 epoche ha peggiorato (Val 78.83% → 71.39%; Test ~77.2% → 71.0%).
- Conservato solo ai fini d’analisi: `outputs/analysis/top121_ft_20250811_111713/`.

---

Hai bisogno di un “tour interattivo” con link diretti a funzioni/righe? Posso aggiungere riferimenti puntuali per la tua IDE (se supporta i deep link) o creare una checklist di review per PR/code‑reading.
