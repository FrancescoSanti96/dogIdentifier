#!/usr/bin/env python3
"""
Prepara il dataset TOP30 bilanciato estendendo il set a 10 razze.

Logica:
- Recupera le 10 razze base da `data/top10_balanced/train` se esiste,
  altrimenti usa una lista di default coerente con il progetto.
- Scansiona `data/breeds` per contare le immagini per razza e seleziona
  automaticamente 20 razze aggiuntive con più immagini (escludendo le 10 base).
- Crea split bilanciati (70/15/15) in `data/top30_balanced`, campionando lo stesso
  numero di immagini per razza (pari al minimo disponibile tra le 30).

Override opzionali:
- `--breeds-base` per specificare manualmente le 10 razze di base
- `--breeds-extra` per specificare manualmente fino a 20 razze aggiuntive
- `--total` per scegliere un totale diverso da 30 (es. 20/40), mantenendo la stessa logica

Uso rapido:
  python prepare_top30_balanced.py

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


DEFAULT_BASE10 = [
    # Coerenti con il dataset 10-razze usato in training
    "Australian_Shepherd_Dog",
    "French_bulldog",
    "German_shepherd",
    "Great_Dane",
    "Labrador_retriever",
    "Pomeranian",
    "Rottweiler",
    "Yorkshire_terrier",
    "beagle",
    "golden_retriever",
]


def main():
    parser = argparse.ArgumentParser(description="Prepare TOP30 balanced dataset")
    parser.add_argument(
        "--source", default="data/breeds", help="Directory sorgente con tutte le razze"
    )
    parser.add_argument(
        "--output",
        default="data/top30_balanced",
        help="Directory di output per gli split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--total", type=int, default=30, help="Numero totale di razze da includere"
    )
    parser.add_argument(
        "--breeds-base", nargs="*", default=None, help="Lista base di razze (es. 10)"
    )
    parser.add_argument(
        "--breeds-extra",
        nargs="*",
        default=None,
        help="Lista extra di razze da aggiungere",
    )
    parser.add_argument(
        "--base-from-top10",
        action="store_true",
        help="Leggi le 10 base da data/top10_balanced/train",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    source = Path(args.source)
    output = Path(args.output)
    (output / "train").mkdir(parents=True, exist_ok=True)
    (output / "val").mkdir(parents=True, exist_ok=True)
    (output / "test").mkdir(parents=True, exist_ok=True)

    # 1) Determina le razze base
    base_breeds: List[str]
    if args.base_from_top10:
        top10_dir = Path("data/top10_balanced/train")
        if top10_dir.exists():
            base_breeds = sorted([d.name for d in top10_dir.iterdir() if d.is_dir()])
        else:
            base_breeds = DEFAULT_BASE10
    elif args.breeds_base:
        base_breeds = args.breeds_base
    else:
        base_breeds = DEFAULT_BASE10

    # 2) Determina le extra
    if args.breeds_extra:
        extra_breeds = args.breeds_extra
    else:
        # auto-selezione: ordina per numero di immagini
        counts = []
        for d in source.iterdir():
            if d.is_dir() and d.name not in base_breeds:
                num = len(list_images(d))
                if num > 0:
                    counts.append((d.name, num))
        # ordina per disponibilità
        counts.sort(key=lambda x: x[1], reverse=True)
        need = max(0, args.total - len(base_breeds))
        extra_breeds = [name for name, _ in counts[:need]]

    selected = base_breeds + extra_breeds
    assert (
        len(selected) == args.total
    ), f"Attese {args.total} razze, trovate {len(selected)}"

    print("🐕 PREPARAZIONE TOP BALANCED")
    print("=" * 60)
    print(f"📁 Source: {source}")
    print(f"📁 Output: {output}")
    print(f"🎲 Seed: {args.seed}")
    print(f"📝 Razze base ({len(base_breeds)}): {base_breeds}")
    print(f"➕ Extra ({len(extra_breeds)}): {extra_breeds}")
    print(f"🎯 Totale razze: {len(selected)}")

    # 3) Raccogli immagini e determina il minimo bilanciamento
    breed_to_images = {}
    min_total = None
    for breed in selected:
        breed_dir = source / breed
        if not breed_dir.exists():
            raise FileNotFoundError(f"Razza non trovata: {breed_dir}")
        imgs = list_images(breed_dir)
        if len(imgs) == 0:
            raise RuntimeError(f"Nessuna immagine trovata per: {breed}")
        breed_to_images[breed] = imgs
        min_total = len(imgs) if min_total is None else min(min_total, len(imgs))

    n_train = int(args.train_ratio * min_total)
    n_val = int(args.val_ratio * min_total)
    n_test = min_total - n_train - n_val
    assert n_train > 0 and n_val > 0 and n_test > 0, "Quote per split non valide"

    print(f"\n📊 Campioni per razza (bilanciati sul minimo={min_total}):")
    print(f"   Train: {n_train} | Val: {n_val} | Test: {n_test}")

    total_copied = {"train": 0, "val": 0, "test": 0}

    # 4) Copia negli split
    for breed in selected:
        imgs = breed_to_images[breed]
        idx = rng.permutation(len(imgs))
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
