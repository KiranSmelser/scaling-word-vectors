import random
import numpy as np
import torch
from omegaconf import DictConfig
import csv

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_metrics(results: list, path: str):
    """
    results: list of dicts, each with the same keys
    """
    keys = results[0].keys()
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)