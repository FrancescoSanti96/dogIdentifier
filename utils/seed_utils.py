#!/usr/bin/env python3
"""
Utilità per riproducibilità e determinismo degli esperimenti
"""

import os
import random
import numpy as np
import torch


def set_deterministic(seed: int = 42) -> None:
    """
    Imposta seed per riproducibilità completa degli esperimenti

    Controlla tutti i generatori random: Python, NumPy, PyTorch CPU/GPU, cuDNN.
    Trade-off: riproducibilità perfetta vs performance leggermente ridotte.

    Args:
        seed: Seed per tutti i generatori random (default: 42)
    """
    # STEP 1: Seed tutti i generatori random per consistency end-to-end
    os.environ["PYTHONHASHSEED"] = str(seed)  # Hash operations (dict ordering, etc.)
    random.seed(seed)  # Python built-in random module
    np.random.seed(seed)  # NumPy random number generator
    torch.manual_seed(seed)  # PyTorch CPU tensor operations

    # STEP 2: GPU determinism (se disponibile)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Tutti i GPU device CUDA

    # STEP 3: cuDNN Backend Control - CRITICO per CNN deep learning
    # Trade-off importante: DETERMINISMO vs PERFORMANCE
    torch.backends.cudnn.deterministic = (
        True  # Forza algoritmi deterministici (stesso risultato sempre)
    )
    torch.backends.cudnn.benchmark = (
        False  # Disabilita auto-optimization (varia tra run)
    )
