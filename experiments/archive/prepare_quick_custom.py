#!/usr/bin/env python3
"""
Crea il quick set (5 razze) a partire da data/breeds, selezionando razze specifiche
e generando gli split fisici train/val/test in data/quick_splits (70/15/15).

Uso tipico:
  python prepare_quick_custom.py \
    --breeds Australian_Shepherd_Dog Norwich_terrier Japanese_spaniel Siberian_husky Chihuahua

Se non specifichi --breeds, usa di default le 5 richieste.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List

from utils.dataloader import create_dataset_splits


def copy_selected_breeds(
    source_root: Path, target_root: Path, breeds: List[str]
) -> List[str]:
    target_root.mkdir(parents=True, exist_ok=True)
    missing = []
    for breed in breeds:
        src = source_root / breed
        dst = target_root / breed
        if not src.exists() or not src.is_dir():
            print(f"❌ Breed non trovata: {breed} in {source_root}")
            missing.append(breed)
            continue
        print(f"📁 Copio {breed} ...")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Crea quick set con 5 razze specifiche"
    )
    parser.add_argument(
        "--source",
        default="data/breeds",
        help="Directory sorgente che contiene tutte le razze (default: data/breeds)",
    )
    parser.add_argument(
        "--output",
        default="data/quick_splits",
        help="Directory di output per gli split (default: data/quick_splits)",
    )
    parser.add_argument(
        "--temp-source",
        default="data/quick_custom_source",
        help="Directory temporanea dove copiare solo le razze selezionate",
    )
    parser.add_argument(
        "--breeds",
        nargs="*",
        default=[
            "Australian_Shepherd_Dog",
            "Norwich_terrier",
            "Japanese_spaniel",
            "Siberian_husky",
            "Chihuahua",
        ],
        help="Elenco razze da includere (ordine desiderato)",
    )
    args = parser.parse_args()

    source_root = Path(args.source)
    temp_source = Path(args.temp_source)
    output_dir = Path(args.output)

    print("🚀 CREAZIONE QUICK SET PERSONALIZZATO (5 razze)")
    print("=" * 60)
    print(f"📂 Sorgente: {source_root}")
    print(f"📂 Temp source: {temp_source}")
    print(f"📂 Output splits: {output_dir}")
    print(f"🎯 Razze: {args.breeds}")

    if not source_root.exists():
        print(f"❌ Sorgente non trovata: {source_root}")
        return 1

    # Copia solo le razze selezionate in una cartella temporanea
    missing = copy_selected_breeds(source_root, temp_source, args.breeds)
    if missing:
        print(f"\n⚠️  Mancano {len(missing)} razze richieste: {missing}")
        print("   Correggi i nomi delle cartelle in data/breeds e riprova")
        return 1

    # Crea gli split fisici 70/15/15
    if output_dir.exists():
        print(f"\n🧹 Pulizia output precedente: {output_dir}")
        shutil.rmtree(output_dir)

    create_dataset_splits(
        source_dir=str(temp_source),
        output_dir=str(output_dir),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print(f"\n✅ Quick set creato con successo in: {output_dir}")
    print("   Ordine/label mapping seguirà l'ordine delle cartelle:")
    for i, b in enumerate(args.breeds):
        print(f"   {i}: {b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
