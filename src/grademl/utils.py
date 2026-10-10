"""Training helpers shared by the training scripts."""
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init(worker_id: int) -> None:
    # Distinct python `random` streams per DataLoader worker (crops use it).
    random.seed(torch.initial_seed() % 2**32 + worker_id)
