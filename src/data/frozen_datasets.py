import os
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from data.image_datasets import IndexedImageDataset

FROZEN_ROOT = './data/frozen'


def _base(dataset, encoder, base_dir=None):
    folder = base_dir or os.path.join(FROZEN_ROOT, f'{dataset}_{encoder}')
    fp = os.path.join(folder, 'base.pt')
    if not os.path.exists(fp):
        raise FileNotFoundError(f'{fp} missing — run embedding extraction first')
    return torch.load(fp, map_location='cpu')


def load_frozen(dataset, encoder, val_split=0.1, batch_size=256, seed=42, num_workers=0, base_dir=None, split_seed=42, **_ignored):
    d = _base(dataset, encoder, base_dir)
    X = d['trainval_emb'].float(); y = d['trainval_lab'].long()
    Xte = d['test_emb'].float(); yte = d['test_lab'].long()
    class_names = d['class_names']

    trainval = TensorDataset(X, y)
    n = len(trainval); n_val = int(n * val_split); n_tr = n - n_val
    if 'train_indices' in d:
        if d['split_seed'] != split_seed or d['val_split'] != val_split:
            raise ValueError(
                'Requested split differs from the cached fine-tuning split'
            )
        tr_split = Subset(trainval, d['train_indices'].tolist())
        va_split = Subset(trainval, d['val_indices'].tolist())
    else:
        gen = torch.Generator().manual_seed(split_seed)
        tr_split, va_split = random_split(
            trainval, [n_tr, n_val], generator=gen
        )

    train_set = Subset(trainval, tr_split.indices)
    val_set = Subset(trainval, va_split.indices)
    embed_set = Subset(trainval, tr_split.indices)
    test_set = TensorDataset(Xte, yte)

    common = dict(num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(IndexedImageDataset(train_set), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), **common,)
    val_loader = DataLoader(IndexedImageDataset(val_set), batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(IndexedImageDataset(test_set), batch_size=batch_size, shuffle=False, **common)
    embedding_loader = DataLoader(IndexedImageDataset(embed_set), batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, embedding_loader, class_names


def load_frozen_corruption(dataset, encoder, corruption, severity, batch_size=256, num_workers=0, base_dir=None):
    folder = base_dir or os.path.join(FROZEN_ROOT, f'{dataset}_{encoder}')
    fp = os.path.join(folder, 'corruptions', f'{corruption}_{severity}.pt')
    if not os.path.exists(fp):
        raise FileNotFoundError(f'{fp} missing — extract with --robustness')
    d = torch.load(fp)
    ds = TensorDataset(d['emb'].float(), d['lab'].long())
    return DataLoader(IndexedImageDataset(ds), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


def load_frozen_ood(dataset, encoder, batch_size=256, num_workers=0, base_dir=None):
    folder = base_dir or os.path.join(FROZEN_ROOT, f'{dataset}_{encoder}')
    fp = os.path.join(folder, 'svhn_ood.pt')
    if not os.path.exists(fp):
        raise FileNotFoundError(f'{fp} missing — extract with --robustness')
    d = torch.load(fp)
    ds = TensorDataset(d['emb'].float(), d['lab'].long())
    return DataLoader(IndexedImageDataset(ds), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)
