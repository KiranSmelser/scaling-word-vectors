import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

class POSDataset(Dataset):
    def __init__(self, split: str, cfg, word2idx=None, tag2idx=None):
        self.cfg = cfg
        raw = load_dataset(cfg.dataset.name, cfg.dataset.config_name)[split]
        self.tokens = raw[cfg.dataset.tokens_column]
        self.tags = raw[cfg.dataset.tags_column]

        if word2idx is None:
            self._build_vocab()
        else:
            self.word2idx = word2idx
            self.tag2idx = tag2idx

        self.pad_idx = self.word2idx[self.cfg.dataset.pad_token]
        self.tag_pad_idx = self.tag2idx[self.cfg.dataset.tag_pad_token]

    def _build_vocab(self):
        cfg = self.cfg
        counter = Counter(tok for sent in self.tokens for tok in sent)
        # word vocab
        self.word2idx = {cfg.dataset.pad_token: 0, cfg.dataset.unk_token: 1}
        for word, _ in counter.most_common():
            self.word2idx[word] = len(self.word2idx)
        # tag vocab
        tags = set(tag for sent in self.tags for tag in sent)
        self.tag2idx = {cfg.dataset.tag_pad_token: 0}
        for tag in sorted(tags):
            self.tag2idx[tag] = len(self.tag2idx)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        cfg = self.cfg
        words = self.tokens[idx]
        tags = self.tags[idx]
        x = [self.word2idx.get(w, self.word2idx[cfg.dataset.unk_token]) for w in words]
        y = [self.tag2idx[t] for t in tags]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)

    padded_x = torch.full((len(xs), max_len), fill_value=0, dtype=torch.long)
    padded_y = torch.full((len(xs), max_len), fill_value=0, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :lengths[i]] = x
        padded_y[i, :lengths[i]] = y

    return padded_x, padded_y, lengths

def get_dataloaders(cfg):
    train_ds = POSDataset('train', cfg)
    dev_ds   = POSDataset('validation', cfg, train_ds.word2idx, train_ds.tag2idx)
    test_ds  = POSDataset('test', cfg, train_ds.word2idx, train_ds.tag2idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.train.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.train.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader, train_ds.word2idx, train_ds.tag2idx