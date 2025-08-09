#!/usr/bin/env python3
"""
Seed and determinism utilities
"""

import os
import random
import numpy as np
import torch


def set_deterministic(seed: int = 42) -> None:
    """Set seeds and torch/cuDNN flags for reproducibility.

    Args:
        seed: The random seed to use for Python, NumPy and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
