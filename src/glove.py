import numpy as np
import torch


def load_glove_matrix(glove_file, vocab, pad_idx=0, seed=1):
    # Build embedding matrix from GloVe file
    # Give GloVe words pretrained vector
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vectors = {}
    with open(glove_file, encoding="utf-8") as f:
        for line in f:
            word, rest = line.split(" ", 1)
            if word in word_to_idx:
                vectors[word] = np.asarray(rest.split(), dtype=np.float32)

    dim = len(next(iter(vectors.values())))
    std = np.std(np.stack(list(vectors.values())))
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, std, size=(len(vocab), dim)).astype(np.float32)
    for word, vec in vectors.items():
        matrix[word_to_idx[word]] = vec
    matrix[pad_idx] = 0

    return torch.from_numpy(matrix), len(vectors) / len(vocab)
