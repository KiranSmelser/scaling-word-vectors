import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

def evaluate(model, loader, tag2idx: dict, device):
    model.eval()
    all_preds, all_labels = [], []
    pad_label = tag2idx['<pad>']

    with torch.no_grad():
        for x, y, lengths in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            logits = model(x, lengths)
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = y.tolist()
            for p_seq, l_seq, L in zip(preds, labels, lengths):
                all_preds.extend(p_seq[:L])
                all_labels.extend(l_seq[:L])

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1