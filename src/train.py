import os
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from src.utils import seed_everything
from src.data_loader import get_dataloaders
from src.embeddings import load_glove, scale_embeddings
from src.model import BiLSTMTagger
from src.evaluate import evaluate

def run_experiment(scale: float, seed: int, cfg) -> dict:
    seed_everything(seed)

    train_loader, dev_loader, test_loader, w2i, t2i = get_dataloaders(cfg)

    emb_matrix = load_glove(cfg.embeddings.path, w2i, cfg.embeddings.dims)
    emb_matrix = scale_embeddings(emb_matrix, scale)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMTagger(len(w2i), len(t2i),
                         cfg.embeddings.dims,
                         cfg.model.hidden_size,
                         cfg.model.dropout).to(device)
    model.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

    # optimizer and loss
    criterion = nn.CrossEntropyLoss(ignore_index=t2i[cfg.dataset.tag_pad_token])
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    # training loop
    best_dev = 0.0
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        for x, y, lengths in tqdm(train_loader, desc=f"Train S={scale} Seed={seed} E={epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

        # eval on dev set
        dev_acc, dev_f1 = evaluate(model, dev_loader, t2i, device)
        if dev_f1 > best_dev:
            best_dev = dev_f1
            torch.save(model.state_dict(),
                       os.path.join(cfg.train.save_dir, f"best_s{scale}_seed{seed}.pt"))

    # test with best model
    model.load_state_dict(torch.load(
        os.path.join(cfg.train.save_dir, f"best_s{scale}_seed{seed}.pt")))
    test_acc, test_f1 = evaluate(model, test_loader, t2i, device)

    return {"scale": scale, "seed": seed, "accuracy": test_acc, "f1": test_f1}

def main():
    cfg = OmegaConf.load("config/default.yaml")
    results = []

    # sweep over scales and seeds
    for s in tqdm(cfg.embeddings.scales, desc="Scale sweep"):
        for seed in tqdm(cfg.train.seeds, desc="Seed"):
            r = run_experiment(s, seed, cfg)
            results.append(r)

    from src.utils import save_metrics
    os.makedirs("results", exist_ok=True)
    save_metrics(results, "results/metrics.csv")

if __name__ == "__main__":
    main()