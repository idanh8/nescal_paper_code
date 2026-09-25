import os
import re
import zipfile
import urllib.request
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.text_datasets import _load_20ng_canonical

GLOVE_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
_TOKEN = re.compile(r"[a-z]+")


def _tok(s):
    return _TOKEN.findall(s.lower())


def _build_vocab(texts, min_freq=2, max_size=20000):
    c = Counter(t for s in texts for t in _tok(s))
    itos = ['<pad>', '<unk>'] + [w for w, n in c.most_common(max_size) if n >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def _load_glove_vectors(cache_dir, embed_dim=100):
    txt = os.path.join(cache_dir, f'glove.6B.{embed_dim}d.txt')
    if not os.path.exists(txt):
        os.makedirs(cache_dir, exist_ok=True)
        zp = os.path.join(cache_dir, 'glove.6B.zip')
        if not os.path.exists(zp):
            print(f'[glove] downloading {GLOVE_URL} (~822 MB) ...', flush=True)
            urllib.request.urlretrieve(GLOVE_URL, zp)
        print(f'[glove] extracting glove.6B.{embed_dim}d.txt ...', flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extract(f'glove.6B.{embed_dim}d.txt', cache_dir)
    vecs = {}
    with open(txt, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip().split(' ')
            vecs[p[0]] = np.asarray(p[1:], dtype=np.float32)
    return vecs


def _build_matrix(stoi, cache_dir, embed_dim=100, seed=0):
    vecs = _load_glove_vectors(cache_dir, embed_dim)
    rng = np.random.RandomState(seed)
    M = rng.normal(scale=0.1, size=(len(stoi), embed_dim)).astype(np.float32)
    M[0] = 0.0
    hit = 0
    for w, i in stoi.items():
        v = vecs.get(w)
        if v is not None:
            M[i] = v
            hit += 1
    print(f'[glove] {hit}/{len(stoi)} vocab tokens found in GloVe-{embed_dim}d', flush=True)
    return torch.tensor(M)


class _IndexedText(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i], i


def _encode(texts, stoi, max_len):
    unk, pad = stoi['<unk>'], stoi['<pad>']
    out = torch.full((len(texts), max_len), pad, dtype=torch.long)
    for r, s in enumerate(texts):
        for c, t in enumerate(_tok(s)[:max_len]):
            out[r, c] = stoi.get(t, unk)
    return out


def build_20ng_gpcnn(cache_dir='./data/glove', max_len=1000, min_freq=2, max_vocab=20000, batch_size=128, val_frac=0.1, num_workers=4, embed_dim=100, matrix_out='./data/glove/gpcnn_20ng_matrix_100d.pt'):
    (tr_txt, tr_lab), (te_txt, te_lab), names = _load_20ng_canonical()
    stoi, _ = _build_vocab(tr_txt, min_freq=min_freq, max_size=max_vocab)
    if not os.path.exists(matrix_out):
        os.makedirs(os.path.dirname(matrix_out) or '.', exist_ok=True)
        torch.save(_build_matrix(stoi, cache_dir, embed_dim=embed_dim), matrix_out)
    Xtr_all, Xte = _encode(tr_txt, stoi, max_len), _encode(te_txt, stoi, max_len)
    ytr_all, yte = torch.tensor(tr_lab), torch.tensor(te_lab)

    def loader_fn(seed):
        g = torch.Generator().manual_seed(seed)
        n = len(ytr_all)
        perm = torch.randperm(n, generator=g)
        n_val = int(n * val_frac)
        vi, ti = perm[:n_val], perm[n_val:]
        tr_ds = _IndexedText(Xtr_all[ti], ytr_all[ti])
        va_ds = _IndexedText(Xtr_all[vi], ytr_all[vi])
        te_ds = _IndexedText(Xte, yte)
        mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers)
        return mk(tr_ds, True), mk(va_ds, False), mk(te_ds, False), mk(tr_ds, False), names

    return loader_fn, len(names), len(stoi), matrix_out
