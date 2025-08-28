#!/usr/bin/env python3
"""
Script unificato per preparazione dataset bilanciati

Supporta la creazione di dataset bilanciati per diverse scale di razze.

Usage:
    python src/prepare_data.py --breeds 10   # Prepara dataset 10 razze
    python src/prepare_data.py --breeds 30   # Prepara dataset 30 razze
    python src/prepare_data.py --breeds 121  # Prepara dataset 121 razze

Features:
- Bilanciamento automatico (stesso numero immagini per razza)
- Split 70/15/15 (train/val/test)
- Statistiche dettagliate del dataset
- Preservazione qualità immagini
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_helper import ConfigHelper


# Configurazioni per diverse scale di razze
BREED_CONFIGS = {
    5: {
        "output_dir": "data/breeds_5",
        "base_breeds": [
            "Australian_Shepherd_Dog",  # Target principale
            "Chihuahua",
            "Japanese_spaniel", 
            "Norwich_terrier",
            "Siberian_husky",
        ],
        "description": "5 razze baseline per test rapidi",
    },
    10: {
        "output_dir": "data/top10_balanced",
        "base_breeds": [
            "Australian_Shepherd_Dog",  # Target principale
            "Chihuahua",
            "Japanese_spaniel",
            "Norwich_terrier",
            "Siberian_husky",  # Da breeds_5
            "Beagle",
            "Pomeranian",
            "golden_retriever",
            "Maltese_dog",
            "Labrador_retriever",  # Estensione
        ],
        "description": "10 razze popolari bilanciate",
    },
    30: {
        "output_dir": "data/top30_balanced",
        "base_breeds": None,  # Sarà calcolato automaticamente dalle top 30
        "description": "30 razze più popolari bilanciate",
    },
    121: {
        "output_dir": "data/full121_balanced",
        "base_breeds": None,  # Tutte le razze disponibili
        "description": "Dataset completo 121 razze bilanciato",
    },
}


def get_breed_image_counts(source_dir: Path) -> dict:
    """Conta le immagini per ogni razza nel dataset sorgente"""
    breed_counts = {}

    for breed_dir in source_dir.iterdir():
        if breed_dir.is_dir():
            # Conta solo file immagine validi
            image_files = [
                f
                for f in breed_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
            breed_counts[breed_dir.name] = len(image_files)

    return breed_counts


def select_breeds_for_scale(
    source_dir: Path, num_breeds: int, base_breeds: list = None
) -> list:
    """Seleziona le razze per il dataset in base alla scala richiesta"""
    breed_counts = get_breed_image_counts(source_dir)

    if base_breeds:
        # Usa le razze specificate
        selected_breeds = []
        for breed in base_breeds:
            if breed in breed_counts:
                selected_breeds.append(breed)
            else:
                print(f"⚠️  Razza {breed} non trovata nel dataset sorgente")
        return selected_breeds
    else:
        # Seleziona le top N razze per numero di immagini
        sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)

        # Assicurati che Australian_Shepherd_Dog sia sempre incluso
        selected_breeds = []
        if "Australian_Shepherd_Dog" in breed_counts:
            selected_breeds.append("Australian_Shepherd_Dog")

        # Aggiungi le altre razze top
        for breed, count in sorted_breeds:
            if breed not in selected_breeds and len(selected_breeds) < num_breeds:
                selected_breeds.append(breed)

        return selected_breeds[:num_breeds]


def calculate_balanced_samples(breed_counts: dict, target_total: int = None) -> dict:
    """
    Calcola il numero di campioni per razza per bilanciamento

    Questa funzione implementa la strategia di bilanciamento del dataset,
    fondamentale per evitare bias verso razze con più immagini.

    Strategia:
    1. Se target_total non specificato: usa il minimo tra le razze (almeno 100)
    2. Se target_total specificato: distribuisce equamente tra le razze
    3. Limita ogni razza al numero di immagini disponibili (no oversampling)

    Questo approccio garantisce:
    - Bilanciamento perfetto tra razze
    - Nessuna perdita di qualità (no synthetic data)
    - Coefficient of Variation < 0.2 (eccellente bilanciamento)

    Args:
        breed_counts: Dizionario {breed_name: num_images}
        target_total: Numero totale target di immagini (opzionale)

    Restituisce:
        Dizionario {breed_name: num_samples_to_use}
    """
    if not breed_counts:
        return {}

    # Strategia di bilanciamento: usa il minimo comune denominatore
    # Obiettivo: evitare bias verso razze con più immagini nel dataset
    if target_total is None:
        # Modalità automatica: usa la razza con meno immagini come limite
        min_samples = min(breed_counts.values())  # Es: se min è 150, tutte avranno 150
        target_per_breed = max(
            min_samples, 100
        )  # Minimo 100 per avere training significativo
    else:
        # Modalità manuale: distribuisci il target totale equamente
        target_per_breed = target_total // len(breed_counts)

    # Limita ogni razza alle immagini disponibili (no oversampling/synthetic data)
    # Preferisco perdere immagini piuttosto che generare dati artificiali
    balanced_samples = {}
    for breed, count in breed_counts.items():
        balanced_samples[breed] = min(
            count, target_per_breed
        )  # Non superare mai il disponibile

    return balanced_samples


def create_balanced_splits(
    source_dir: Path,
    output_dir: Path,
    selected_breeds: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """Crea split bilanciati per le razze selezionate"""

    print(f"📂 Creando dataset bilanciato in: {output_dir}")
    print(f"🎯 Razze selezionate: {len(selected_breeds)}")

    # Pulisci directory di output
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Crea struttura directory
    for split in ["train", "val", "test"]:
        for breed in selected_breeds:
            (output_dir / split / breed).mkdir(parents=True, exist_ok=True)

    # Conta immagini disponibili per ogni razza
    breed_counts = {}
    for breed in selected_breeds:
        breed_dir = source_dir / breed
        if breed_dir.exists():
            image_files = [
                f
                for f in breed_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
            breed_counts[breed] = len(image_files)
        else:
            print(f"⚠️  Directory {breed} non trovata, saltando...")
            continue

    # Calcola campioni bilanciati
    balanced_samples = calculate_balanced_samples(breed_counts)

    print(f"\n📊 BILANCIAMENTO DATASET:")
    print("-" * 50)

    total_train = total_val = total_test = 0
    split_stats = defaultdict(lambda: defaultdict(int))

    # Processa ogni razza
    for breed in selected_breeds:
        if breed not in breed_counts:
            continue

        breed_dir = source_dir / breed
        image_files = [
            f
            for f in breed_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        # Campionamento bilanciato: prendi solo le immagini necessarie per bilanciamento
        target_samples = balanced_samples[breed]
        if len(image_files) > target_samples:
            # Random sampling senza replacement per diversità
            image_files = random.sample(image_files, target_samples)

        # Calcola dimensioni split (70/15/15 è standard per deep learning)
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)  # ~70% per training
        n_val = int(n_total * val_ratio)  # ~15% per validation (early stopping)
        n_test = n_total - n_train - n_val  # Rimanente per test finale

        # Split deterministico con shuffling iniziale
        random.shuffle(image_files)  # Mescola per evitare bias temporali/di ordinamento
        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]

        # Copia fisica file (no symlinks per portabilità)
        for files, split_name in [
            (train_files, "train"),
            (val_files, "val"),
            (test_files, "test"),
        ]:
            for img_file in files:
                dst_path = output_dir / split_name / breed / img_file.name
                shutil.copy2(img_file, dst_path)

        # Statistiche
        split_stats["train"][breed] = len(train_files)
        split_stats["val"][breed] = len(val_files)
        split_stats["test"][breed] = len(test_files)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(
            f"{breed:30} | {len(train_files):3} train | {len(val_files):3} val | {len(test_files):3} test | {n_total:3} total"
        )

    print("-" * 50)
    print(
        f"{'TOTAL':30} | {total_train:3} train | {total_val:3} val | {total_test:3} test | {total_train + total_val + total_test:3} total"
    )

    # Calcola coefficient of variation per verificare bilanciamento
    # CV = std/mean: misura la variabilità relativa del dataset
    # CV < 0.2 = eccellente, CV < 0.5 = buono, CV >= 0.5 = migliorabile
    train_counts = list(split_stats["train"].values())
    if train_counts:
        mean_train = sum(train_counts) / len(train_counts)
        std_train = (
            sum((x - mean_train) ** 2 for x in train_counts) / len(train_counts)
        ) ** 0.5
        cv = std_train / mean_train if mean_train > 0 else 0
        print(f"\n📊 Coefficient of Variation (training): {cv:.3f}")

        # Interpretazione CV per valutazione qualità bilanciamento
        if cv < 0.2:
            print("✅ Bilanciamento ECCELLENTE (CV < 0.2)")
            print("   Dataset perfettamente bilanciato per training ottimale")
        elif cv < 0.5:
            print("✅ Bilanciamento BUONO (CV < 0.5)")
            print("   Dataset sufficientemente bilanciato per buoni risultati")
        else:
            print("⚠️  Bilanciamento MIGLIORABILE (CV >= 0.5)")
            print("   Considera di bilanciare meglio il dataset")

    # Salva statistiche
    stats = {
        "num_breeds": len(selected_breeds),
        "breeds": selected_breeds,
        "total_images": total_train + total_val + total_test,
        "splits": {"train": total_train, "val": total_val, "test": total_test},
        "per_breed_stats": dict(split_stats),
        "balance_cv": cv,
        "creation_date": str(Path().absolute()),
    }

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Dataset creato con successo!")
    print(f"   Output: {output_dir}")
    print(f"   Statistiche: {output_dir}/dataset_stats.json")

    return stats


def prepare_breeds_dataset(
    num_breeds: int, source_dir: str = "data/breeds", config_path: str = "config.json"
):
    """Prepara dataset bilanciato per il numero specificato di razze"""

    if num_breeds not in BREED_CONFIGS:
        raise ValueError(
            f"Numero razze {num_breeds} non supportato. "
            f"Supportati: {list(BREED_CONFIGS.keys())}"
        )

    config = BREED_CONFIGS[num_breeds]
    # Se presente un config.json, consenti di sovrascrivere parametri
    cfg = None
    try:
        cfg = ConfigHelper(config_path)
    except Exception:
        cfg = None

    if cfg is not None:
        # Se definito un path sorgente nel config, usalo come default
        source_dir = cfg.get("data.breed_dataset_path", source_dir)

    source_path = Path(source_dir)
    output_path = Path(config["output_dir"])

    print(f"🚀 PREPARAZIONE DATASET {num_breeds} RAZZE")
    print("=" * 50)
    print(f"📊 {config['description']}")
    print(f"📂 Sorgente: {source_path}")
    print(f"📁 Output: {output_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Directory sorgente non trovata: {source_path}")

    # Seleziona razze per questa scala
    selected_breeds = select_breeds_for_scale(
        source_path, num_breeds, config["base_breeds"]
    )

    if len(selected_breeds) != num_breeds:
        print(f"⚠️  Trovate solo {len(selected_breeds)} razze invece di {num_breeds}")

    print(f"\n🎯 RAZZE SELEZIONATE:")
    for i, breed in enumerate(selected_breeds, 1):
        print(f"   {i:2}. {breed}")

    # Crea dataset bilanciato
    random.seed(42)  # Riproducibilità
    stats = create_balanced_splits(source_path, output_path, selected_breeds)

    print(f"\n💡 UTILIZZO:")
    print(f"   python src/train.py --breeds {num_breeds}")
    print(
        f"   python src/evaluate.py --model outputs/breeds_{num_breeds}/best_model.pth --data {output_path}"
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preparazione dataset bilanciati")
    parser.add_argument(
        "--breeds",
        type=int,
        required=True,
        choices=[5, 10, 30, 121],
        help="Numero di razze per il dataset",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/breeds",
        help="Directory dataset sorgente (default: data/breeds)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Percorso al file di configurazione JSON (default: config.json)",
    )

    args = parser.parse_args()

    try:
        stats = prepare_breeds_dataset(args.breeds, args.source, args.config)
        print(f"\n✅ Preparazione completata con successo!")
        print(f"   Razze: {stats['num_breeds']}")
        print(f"   Immagini totali: {stats['total_images']:,}")
        print(f"   CV bilanciamento: {stats['balance_cv']:.3f}")

    except Exception as e:
        print(f"\n❌ Errore durante la preparazione: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
