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
    os.environ["PYTHONHASHSEED"] = str(seed)  # Operazioni di hash
    random.seed(seed)  # Random Python standard
    np.random.seed(seed)  # Random NumPy
    torch.manual_seed(seed)  # PyTorch CPU

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU (tutti i device)

    # cuDNN: determinismo vs performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
