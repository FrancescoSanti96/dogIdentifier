#!/usr/bin/env python3
"""
Prepara il dataset FULL121 bilanciato usando tutte le razze disponibili in data/breeds.

Logica:
- Scansiona tutte le cartelle di razza in `data/breeds` (incluso `Australian_Shepherd_Dog` aggiunto).
- Per ogni razza, raccoglie le immagini; determina il minimo numero di immagini disponibile tra le razze.
- Crea split bilanciati (70/15/15) campionando lo stesso numero per ogni razza in `data/full121_balanced`.

Uso rapido:
  python prepare_full121_balanced.py \
    --source data/breeds \
    --output data/full121_balanced \
    --seed 42
"""

import argparse
from pathlib import Path
import shutil
import numpy as np
from typing import List


def list_images(breed_dir: Path) -> List[Path]:
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(list(breed_dir.glob(ext)))
    return files


def main():
    parser = argparse.ArgumentParser(description="Prepare FULL121 balanced dataset")
    parser.add_argument("--source", default="data/breeds", help="Directory sorgente con tutte le razze")
    parser.add_argument("--output", default="data/full121_balanced", help="Directory di output per gli split")
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    source = Path(args.source)
    output = Path(args.output)
    (output / "train").mkdir(parents=True, exist_ok=True)
    (output / "val").mkdir(parents=True, exist_ok=True)
    (output / "test").mkdir(parents=True, exist_ok=True)

    # Elenco razze: tutte le directory non vuote
    breed_dirs = [d for d in source.iterdir() if d.is_dir()]
    breed_names = []
    breed_to_images = {}
    min_total = None
    for d in sorted(breed_dirs, key=lambda p: p.name):
        imgs = list_images(d)
        if len(imgs) == 0:
            continue
        breed_names.append(d.name)
        breed_to_images[d.name] = imgs
        min_total = len(imgs) if min_total is None else min(min_total, len(imgs))

    assert len(breed_names) >= 100, "Trovate meno di 100 razze: controlla data/breeds"
    assert min_total and min_total > 0, "Nessuna immagine trovata nelle razze"

    n_train = int(args.train_ratio * min_total)
    n_val = int(args.val_ratio * min_total)
    n_test = min_total - n_train - n_val
    assert n_train > 0 and n_val > 0 and n_test > 0, "Quote per split non valide"

    print("🐕 PREPARAZIONE FULL121 BALANCED")
    print("=" * 60)
    print(f"📁 Source: {source}")
    print(f"📁 Output: {output}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🎯 Razze trovate: {len(breed_names)}")
    print(f"📊 Campioni per razza (bilanciati sul minimo={min_total}):")
    print(f"   Train: {n_train} | Val: {n_val} | Test: {n_test}")

    total_copied = {"train": 0, "val": 0, "test": 0}
    for breed in breed_names:
        imgs = breed_to_images[breed]
        idx = rng.permutation(len(imgs))
        imgs = [imgs[i] for i in idx]

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train : n_train + n_val]
        test_imgs = imgs[n_train + n_val : n_train + n_val + n_test]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest = output / split_name / breed
            dest.mkdir(parents=True, exist_ok=True)
            for p in split_imgs:
                shutil.copy2(p, dest / p.name)
            total_copied[split_name] += len(split_imgs)

        print(f"   ✅ {breed}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("\n✅ Preparazione completata!")
    print(f"   Train: {total_copied['train']} | Val: {total_copied['val']} | Test: {total_copied['test']}")
    print(f"   Output: {output}")


if __name__ == "__main__":
    main()


