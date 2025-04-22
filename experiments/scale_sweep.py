from omegaconf import OmegaConf
from tqdm.auto import tqdm
from src.train import run_experiment
from src.utils import save_metrics
import os

if __name__ == "__main__":
    cfg = OmegaConf.load("config/default.yaml")
    results = []
    os.makedirs("results", exist_ok=True)
    for s in tqdm(cfg.embeddings.scales, desc="Scale sweep"):
        r = run_experiment(s, cfg)
        results.append(r)
    save_metrics(results, "results/metrics.csv")