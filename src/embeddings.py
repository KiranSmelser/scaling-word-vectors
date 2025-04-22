import numpy as np

def load_glove(path: str, word2idx: dict, dim: int) -> np.ndarray:
    mat = np.random.uniform(-0.25, 0.25, (len(word2idx), dim))
    mat[word2idx['<pad>']] = np.zeros(dim)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            w, vec = parts[0], np.array(parts[1:], dtype=float)
            idx = word2idx.get(w)
            if idx is not None:
                mat[idx] = vec
    return mat

def scale_embeddings(mat: np.ndarray, scale: float) -> np.ndarray:
    return mat * scale