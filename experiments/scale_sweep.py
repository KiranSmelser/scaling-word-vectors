from omegaconf import OmegaConf
from tqdm.auto import tqdm
from src.train import main

if __name__ == "__main__":
    cfg = OmegaConf.load("config/default.yaml")
    main()