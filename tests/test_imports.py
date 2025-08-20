#!/usr/bin/env python3
"""
Test minimo per verifica setup progetto Dog Breed Identifier
"""

import sys
import os
from pathlib import Path

# Aggiungi directory root del progetto al path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_essential_imports():
    """Verifica che tutti i moduli essenziali si importino correttamente"""
    print("🧪 Test import moduli essenziali...")

    try:
        from utils.config_helper import ConfigHelper

        print("  ✅ ConfigHelper")
    except ImportError as e:
        print(f"  ❌ ConfigHelper: {e}")
        return False

    try:
        from utils.dataloader import DogBreedDataset, create_dataloaders_from_splits

        print("  ✅ DataLoader modules")
    except ImportError as e:
        print(f"  ❌ DataLoader modules: {e}")
        return False

    try:
        from models.breed_classifier import create_breed_classifier

        print("  ✅ BreedClassifier models")
    except ImportError as e:
        print(f"  ❌ BreedClassifier models: {e}")
        return False

    try:
        from utils.metrics import calculate_metrics, print_metrics_summary

        print("  ✅ Metrics utilities")
    except ImportError as e:
        print(f"  ❌ Metrics utilities: {e}")
        return False

    try:
        from utils.early_stopping import EarlyStopping
        from utils.seed_utils import set_deterministic

        print("  ✅ Training utilities")
    except ImportError as e:
        print(f"  ❌ Training utilities: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🚀 Dog Breed Identifier - Test Setup")
    print("=" * 40)

    success = test_essential_imports()

    print("=" * 40)
    if success:
        print("✅ Setup verificato! Tutti i moduli funzionano correttamente.")
        print("\n📋 Il progetto è pronto per:")
        print("   • Training modelli (src/train.py)")
        print("   • Preparazione dataset (src/prepare_data.py)")
        print("   • Valutazione modelli (src/evaluate.py)")
    else:
        print("❌ Errore setup! Controlla le dipendenze e la struttura progetto.")

    sys.exit(0 if success else 1)
