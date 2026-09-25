import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Subset, random_split, DataLoader
from data.image_datasets import _get_transforms, IndexedImageDataset

def longtail_indices(targets, num_classes, imbalance_factor, seed):
    """
    Exponential long-tail: class c keeps n_c = n_max * factor**(-c/(C-1)); factor = n_max/n_min.
    """
    rng = np.random.RandomState(seed)
    n_max = int((targets == 0).sum())
    keep = []
    for c in range(num_classes):
        n_c = int(round(n_max * (imbalance_factor ** (-(c) / (num_classes - 1)))))
        cls_idx = np.where(targets == c)[0]
        keep.extend(rng.choice(cls_idx, min(max(n_c, 1), len(cls_idx)), replace=False))
    return np.array(sorted(keep))

def apply_label_noise(targets, indices, rate, num_classes, seed, kind='symmetric'):
    """
    Flip a fraction `rate` of labels at `indices`. symmetric -> uniform random OTHER class;
    asymmetric -> (c+1) % C (simple structured-noise proxy). 
    """
    rng = np.random.RandomState(seed + 12345)
    out = np.array(targets).copy()
    idx = np.array(list(indices))
    flip = idx[rng.rand(len(idx)) < rate]
    for i in flip:
        if kind == 'symmetric':
            out[i] = rng.choice([k for k in range(num_classes) if k != out[i]])
        else:
            out[i] = (int(out[i]) + 1) % num_classes
    return out

def build_perturbed_cifar_loaders(dataset, perturbation, seed, batch_size=128, data_root='./data', num_workers=4, val_split=0.1):
    assert dataset in ('cifar10', 'cifar100'), 'Wave 5 perturbations are CIFAR-only.'
    random.seed(seed); np.random.seed(seed)
    train_tf, eval_tf = _get_transforms(dataset)
    cls = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    aug = cls(root=data_root, train=True, download=True, transform=train_tf)
    ev = cls(root=data_root, train=True, download=True, transform=eval_tf)
    test = cls(root=data_root, train=False, download=True, transform=eval_tf)
    names = aug.classes; C = len(names)

    if perturbation['type'] == 'longtail':
        keep = longtail_indices(np.array(ev.targets), C, perturbation['imbalance_factor'], seed)
        aug, ev = Subset(aug, keep), Subset(ev, keep)

    val_size = int(len(ev) * val_split); train_size = len(ev) - val_size
    g = torch.Generator().manual_seed(seed)
    train_split, val_set = random_split(ev, [train_size, val_size], generator=g)
    trainset = Subset(aug, train_split.indices)
    embedding_set = Subset(ev, train_split.indices)

    if perturbation['type'] == 'noise':
        noisy = apply_label_noise(ev.targets, train_split.indices, perturbation['rate'], C, seed,
                                  kind=perturbation.get('kind', 'symmetric'))
        for i in train_split.indices:
            aug.targets[i] = int(noisy[i]); ev.targets[i] = int(noisy[i])

    ck = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (DataLoader(IndexedImageDataset(trainset), batch_size=batch_size, shuffle=True, **ck),
            DataLoader(IndexedImageDataset(val_set), batch_size=batch_size, shuffle=False, **ck),
            DataLoader(IndexedImageDataset(test), batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(IndexedImageDataset(embedding_set), batch_size=batch_size, shuffle=False, **ck),
            names)
