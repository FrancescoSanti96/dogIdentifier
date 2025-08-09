#!/usr/bin/env python3
"""
Prepara il dataset TOP10 bilanciato estendendo il set a 5 razze.

Seleziona 10 razze (di default: le 5 di quick5 + 5 popolari) e crea
splits bilanciati (stesso numero di immagini per razza) in data/top10_balanced.

Uso:
  python prepare_top10_balanced.py \
    --source data/breeds \
    --output data/top10_balanced \
    --seed 42 \
    --breeds Australian_Shepherd_Dog Chihuahua Japanese_spaniel Norwich_terrier Siberian_husky Beagle Pomeranian Golden_retriever Maltese_dog Labrador_retriever

Nota: se --breeds non è specificato, verrà usata una lista di default.
"""

import argparse
import os
from pathlib import Path
import shutil
import numpy as np
from typing import List


DEFAULT_BREEDS = [
    # 5 del quick5
    "Australian_Shepherd_Dog",
    "Chihuahua",
    "Japanese_spaniel",
    "Norwich_terrier",
    "Siberian_husky",
    # +5 popolari/abbondanti
    "Beagle",
    "Pomeranian",
    "Golden_retriever",
    "Maltese_dog",
    "Labrador_retriever",
]


def list_images(breed_dir: Path) -> List[Path]:
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(list(breed_dir.glob(ext)))
    return files


def main():
    parser = argparse.ArgumentParser(description="Prepare TOP10 balanced dataset")
    parser.add_argument(
        "--source", default="data/breeds", help="Directory sorgente con tutte le razze"
    )
    parser.add_argument(
        "--output",
        default="data/top10_balanced",
        help="Directory di output per gli split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument(
        "--breeds",
        nargs="*",
        default=None,
        help="Lista di 10 razze (default: set predefinito)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    args = parser.parse_args()

    np.random.seed(args.seed)

    source = Path(args.source)
    output = Path(args.output)
    (output / "train").mkdir(parents=True, exist_ok=True)
    (output / "val").mkdir(parents=True, exist_ok=True)
    (output / "test").mkdir(parents=True, exist_ok=True)

    breeds = args.breeds if args.breeds else DEFAULT_BREEDS
    assert len(breeds) == 10, f"Attese 10 razze, ricevute {len(breeds)}"

    print("🐕 PREPARAZIONE TOP10 BILANCIATO")
    print("=" * 60)
    print(f"📁 Source: {source}")
    print(f"📁 Output: {output}")
    print(f"🎲 Seed: {args.seed}")
    print(f"📝 Razze selezionate (10): {breeds}")

    # Raccogli elenco immagini per ogni razza e calcola il minimo
    breed_to_images = {}
    min_total = None
    for breed in breeds:
        breed_dir = source / breed
        if not breed_dir.exists():
            raise FileNotFoundError(f"Razza non trovata: {breed_dir}")
        imgs = list_images(breed_dir)
        if len(imgs) == 0:
            raise RuntimeError(f"Nessuna immagine trovata per: {breed}")
        breed_to_images[breed] = imgs
        min_total = len(imgs) if min_total is None else min(min_total, len(imgs))

    # Calcola quote per split in base al minimo disponibile
    n_train = int(args.train_ratio * min_total)
    n_val = int(args.val_ratio * min_total)
    n_test = min_total - n_train - n_val
    assert n_train > 0 and n_val > 0 and n_test > 0, "Quote per split non valide"

    print(f"\n📊 Campioni per razza (bilanciati sul minimo={min_total}):")
    print(f"   Train: {n_train} | Val: {n_val} | Test: {n_test}")

    total_copied = {"train": 0, "val": 0, "test": 0}

    for breed in breeds:
        imgs = breed_to_images[breed]
        idx = np.random.permutation(len(imgs))
        imgs = [imgs[i] for i in idx]

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train : n_train + n_val]
        test_imgs = imgs[n_train + n_val : n_train + n_val + n_test]

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            dest = output / split_name / breed
            dest.mkdir(parents=True, exist_ok=True)
            for p in split_imgs:
                shutil.copy2(p, dest / p.name)
            total_copied[split_name] += len(split_imgs)

        print(
            f"   ✅ {breed}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test"
        )

    print("\n✅ Preparazione completata!")
    print(
        f"   Train: {total_copied['train']} | Val: {total_copied['val']} | Test: {total_copied['test']}"
    )
    print(f"   Output: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepara dataset TOP 10 BILANCIATO:
- 9 razze più popolari al mondo (AKC ranking)
- 1 Australian Shepherd Dog (obiettivo del progetto)
"""

import os
import shutil
import random
from collections import Counter


def prepare_top10_balanced():
    """Prepara dataset con TOP 9 popolari + Australian Shepherd"""

    print("🎯 TOP 10 BALANCED DATASET PREPARATION")
    print("=====================================")
    print("📋 9 razze più popolari + Australian Shepherd")

    # Directory setup
    source_dir = "data/breeds"
    output_dir = "data/top10_balanced"

    # TOP 9 razze più popolari (AKC Most Popular Breeds 2024)
    top9_popular = [
        "Labrador_retriever",  # #1 - Sempre primo
        "golden_retriever",  # #2 - Sempre top 3
        "German_shepherd",  # #3 - Versatile, popolare
        "French_bulldog",  # #4 - Urban favorite
        "beagle",  # #5 - Family favorite
        "Pomeranian",  # #6 - Toy breed popolare
        "Rottweiler",  # #7 - Guardian popolare
        "Yorkshire_terrier",  # #8 - Toy breed classico
        "Great_Dane",  # #9 - Giant breed iconico
    ]

    # + Australian Shepherd (target breed)
    target_breed = "Australian_Shepherd_Dog"

    all_breeds = top9_popular + [target_breed]

    print(f"\\n🏆 TOP 9 RAZZE POPOLARI:")
    for i, breed in enumerate(top9_popular, 1):
        print(f"   {i:2d}. {breed.replace('_', ' ')}")

    print(f"\\n⭐ BREED TARGET:")
    print(f"   10. {target_breed.replace('_', ' ')}")

    # Verifica disponibilità breeds
    available_breeds = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    missing_breeds = [breed for breed in all_breeds if breed not in available_breeds]
    if missing_breeds:
        print(f"\\n⚠️  BREEDS MANCANTI: {missing_breeds}")
        return False

    # Conta immagini per breed
    print(f"\\n📊 CONTEGGIO IMMAGINI PER BREED:")
    breed_counts = {}
    for breed in all_breeds:
        breed_path = os.path.join(source_dir, breed)
        images = [
            f
            for f in os.listdir(breed_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        breed_counts[breed] = len(images)

        status = "✅" if len(images) >= 100 else "⚠️" if len(images) >= 50 else "❌"
        print(f"   {status} {breed:25s}: {len(images):3d} images")

    # Analisi bilanciamento
    total_images = sum(breed_counts.values())
    min_images = min(breed_counts.values())
    max_images = max(breed_counts.values())
    avg_images = total_images / len(all_breeds)

    print(f"\\n📈 STATISTICHE DATASET:")
    print(f"   Totale immagini: {total_images:,}")
    print(f"   Media per breed: {avg_images:.1f}")
    print(f"   Min per breed: {min_images} ({min(breed_counts, key=breed_counts.get)})")
    print(f"   Max per breed: {max_images} ({max(breed_counts, key=breed_counts.get)})")

    # Coefficient of Variation per bilanciamento
    import numpy as np

    cv = np.std(list(breed_counts.values())) / np.mean(list(breed_counts.values()))
    print(f"   Coeff. Variation: {cv:.3f}", end="")
    if cv < 0.2:
        print(" ✅ (Eccellente)")
    elif cv < 0.4:
        print(" 👍 (Buono)")
    else:
        print(" ⚠️ (Sbilanciato)")

    # Pulisci output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    # Crea splits: 70% train, 15% val, 15% test
    print(f"\\n📂 CREAZIONE SPLITS (70/15/15):")

    total_train = 0
    total_val = 0
    total_test = 0

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split))

    for breed in all_breeds:
        print(f"\\n   📁 Processando {breed}...")

        # Leggi tutte le immagini
        breed_path = os.path.join(source_dir, breed)
        images = [
            f
            for f in os.listdir(breed_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Shuffle per randomizzazione
        random.shuffle(images)

        # Calcola split sizes
        n_images = len(images)
        n_train = int(0.70 * n_images)
        n_val = int(0.15 * n_images)
        n_test = n_images - n_train - n_val

        # Split images
        train_images = images[:n_train]
        val_images = images[n_train : n_train + n_val]
        test_images = images[n_train + n_val :]

        # Crea directory per breed in ogni split
        for split in ["train", "val", "test"]:
            breed_split_dir = os.path.join(output_dir, split, breed)
            os.makedirs(breed_split_dir, exist_ok=True)

        # Copia immagini
        for img in train_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(output_dir, "train", breed, img)
            shutil.copy2(src, dst)

        for img in val_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(output_dir, "val", breed, img)
            shutil.copy2(src, dst)

        for img in test_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(output_dir, "test", breed, img)
            shutil.copy2(src, dst)

        # Stats
        is_target = "⭐" if breed == target_breed else "  "
        print(
            f"     {is_target} Train: {n_train:3d} | Val: {n_val:3d} | Test: {n_test:3d} | Tot: {n_images:3d}"
        )

        total_train += n_train
        total_val += n_val
        total_test += n_test

    # Summary
    total_final = total_train + total_val + total_test

    print(f"\\n" + "=" * 50)
    print(f"✅ DATASET TOP 10 BALANCED CREATO!")
    print(f"=" * 50)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Dataset finale:")
    print(f"   🎯 Razze: {len(all_breeds)} (9 popolari + Australian Shepherd)")
    print(f"   🚂 Training: {total_train:,} images")
    print(f"   ✅ Validation: {total_val:,} images")
    print(f"   🧪 Test: {total_test:,} images")
    print(f"   📈 Totale: {total_final:,} images")

    print(f"\\n🎯 BREED TARGET TRACKING:")
    aus_train = len(os.listdir(os.path.join(output_dir, "train", target_breed)))
    aus_val = len(os.listdir(os.path.join(output_dir, "val", target_breed)))
    aus_test = len(os.listdir(os.path.join(output_dir, "test", target_breed)))
    aus_total = aus_train + aus_val + aus_test

    print(f"   ⭐ Australian Shepherd: {aus_total} images")
    print(f"      Train: {aus_train} | Val: {aus_val} | Test: {aus_test}")

    # Percentuali
    train_pct = (total_train / total_final) * 100
    val_pct = (total_val / total_final) * 100
    test_pct = (total_test / total_final) * 100

    print(f"\\n📊 Split percentuali:")
    print(f"   Train: {train_pct:.1f}%")
    print(f"   Val: {val_pct:.1f}%")
    print(f"   Test: {test_pct:.1f}%")

    print(f"\\n🚀 PRONTO PER TRAINING OTTIMIZZATO!")
    print(f"💡 Usa: python top10_balanced_train.py")

    return True


if __name__ == "__main__":
    print("🔧 Preparazione Dataset TOP 10 Balanced")
    print("=" * 40)

    success = prepare_top10_balanced()

    if success:
        print("\\n✅ Dataset preparation completed!")
    else:
        print("\\n❌ Dataset preparation failed!")
