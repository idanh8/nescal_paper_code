import random
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
import urllib.request, tarfile

_CORRUPTION_URLS = {
    'CIFAR-10-C':      'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar',
    'CIFAR-100-C':     'https://zenodo.org/record/3555552/files/CIFAR-100-C.tar',
    'Tiny-ImageNet-C': 'https://zenodo.org/record/2469796/files/TinyImageNet-C.tar',
}

def _resolve_val_image(val_dir, wnid, fname):
    for p in (os.path.join(val_dir, 'images', fname),
            os.path.join(val_dir, wnid, 'images', fname),
            os.path.join(val_dir, wnid, fname)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Tiny-ImageNet val image '{fname}' (wnid {wnid}) not found under {val_dir} in any known "
        f"layout (images/, {wnid}/images/, {wnid}/). Check the extraction is complete.")

def _ensure_corruption_set(folder, data_root):
    dest = os.path.join(data_root, folder)

    alt = os.path.join(data_root, 'TinyImageNet-C')
    if 'Tiny' in folder and os.path.isdir(alt) and not os.path.isdir(dest):
        os.rename(alt, dest)
    probe = os.path.join(dest, 'gaussian_noise.npy') if 'Tiny' not in folder \
        else os.path.join(dest, 'gaussian_noise', '1')
    if os.path.exists(probe):
        return True
    url = _CORRUPTION_URLS.get(folder)
    if url is None:
        print(f'[data] no known URL for {folder}; skipping.', flush=True); return False
    os.makedirs(data_root, exist_ok=True)
    tar_path = os.path.join(data_root, f'{folder}.tar')
    try:
        print(f'[data] {folder} missing - downloading {url} (large, may take a while)', flush=True)
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as t:
            t.extractall(data_root)
        os.remove(tar_path)

        alt = os.path.join(data_root, 'TinyImageNet-C')
        if 'Tiny' in folder and os.path.isdir(alt) and not os.path.isdir(dest):
            os.rename(alt, dest)
        ok = os.path.exists(probe)
        print(f'[data] {folder} ready.' if ok else f'[data] {folder} extracted but probe {probe} missing.', flush=True)
        return ok
    except Exception as e:
        print(f'[data] failed to fetch {folder} ({type(e).__name__}: {e}) - corruption will be skipped.', flush=True)
        if os.path.exists(tar_path):
            try: os.remove(tar_path)
            except OSError: pass
        return False


class IndexedImageDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return image, label, idx


def get_dataset_labels(dataset):
    if isinstance(dataset, Subset):
        inner = get_dataset_labels(dataset.dataset)
        return np.asarray(inner)[dataset.indices]
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def create_imbalanced_dataset(dataset, imbalance_level, num_classes, min_size=None, seed=42):
    all_labels = get_dataset_labels(dataset)

    samples_per_class = {c: int((all_labels == c).sum()) for c in range(num_classes)}
    total_available = sum(samples_per_class.values())
    if min_size is not None and min_size > total_available:
        raise ValueError(f"Minimum dataset size {min_size} exceeds available samples {total_available}")

    decay_factor = imbalance_level
    base_samples = len(dataset) // num_classes
    class_sample_counts = [max(int(base_samples * (decay_factor ** i)), 200) for i in range(num_classes)]

    current_total = sum(class_sample_counts)
    if min_size is not None and current_total < min_size:
        scale_factor = min_size / current_total
        class_sample_counts = [max(int(np.ceil(c * scale_factor)), 10) for c in class_sample_counts]
        current_total = sum(class_sample_counts)
        while current_total < min_size:
            deficit = min_size - current_total
            sorted_idx = sorted(range(len(class_sample_counts)), key=lambda i: class_sample_counts[i], reverse=True)
            for i in range(min(deficit, len(class_sample_counts))):
                class_sample_counts[sorted_idx[i]] += 1
            current_total = sum(class_sample_counts)

    random.seed(seed)
    random.shuffle(class_sample_counts)

    selected_indices = []
    for class_idx in range(num_classes):
        class_indices = np.where(all_labels == class_idx)[0]
        target_count = min(class_sample_counts[class_idx], len(class_indices))
        selected_indices.extend(np.random.choice(class_indices, target_count, replace=False))

    print(f"Created imbalanced dataset. Total: {len(selected_indices)}, per-class: {class_sample_counts}, "
          f"ratio (max:min): {max(class_sample_counts):.1f}:{min(class_sample_counts):.1f}")
    return Subset(dataset, selected_indices)


CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975); TINYIMAGENET_STD = (0.2770, 0.2691, 0.2821)

def _get_transforms(name: str):
    if name == 'cifar10':
        mean, std, crop_size = CIFAR10_MEAN, CIFAR10_STD, 32
    elif name == 'cifar100':
        mean, std, crop_size = CIFAR100_MEAN, CIFAR100_STD, 32
    elif name == 'tinyimagenet':
        mean, std, crop_size = TINYIMAGENET_MEAN, TINYIMAGENET_STD, 64
    elif name == 'imagenet':
        mean, std, crop_size = IMAGENET_MEAN, IMAGENET_STD, 224
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if crop_size <= 64:
        train_ops = [transforms.RandomCrop(crop_size, padding=4), transforms.RandomHorizontalFlip()]
        eval_ops = []
    else:
        train_ops = [transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip()]
        eval_ops = [transforms.Resize(int(crop_size * 256 / 224)), transforms.CenterCrop(crop_size)]

    train_transform = transforms.Compose(train_ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_transform = transforms.Compose(eval_ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_transform, eval_transform


def load_dataset(name: str, val_split: float = 0.1, batch_size: int = 128,
                  class_imbalance=None, min_size=None, seed: int = 42, data_root: str = './data', num_workers: int = 8):
    if name == 'synthetic':
        from data.synthetic_datasets import load_synthetic
        return load_synthetic(val_split=val_split, batch_size=batch_size, seed=seed,
                              data_root=data_root, num_workers=num_workers)
    if name not in ('cifar10', 'cifar100', 'tinyimagenet', 'imagenet'):
        try:
            from data.frozen_datasets import load_frozen


            for enc in ('vit_b16', 'dinov2', 'roberta', 'bert'):
                if name.endswith(f'_{enc}'):
                    ds_tag = name[:-(len(enc) + 1)]
                    return load_frozen(ds_tag, enc, val_split=val_split, batch_size=batch_size,
                                       seed=seed, num_workers=num_workers)
        except Exception:
            pass
        raise ValueError(f"Unknown dataset: '{name}'. Expected one of: cifar10, cifar100, tinyimagenet, imagenet, "
                         f"or a frozen-encoder name like 'cifar10_vit_b16'")
    if name in ('cifar10', 'cifar100'):
        return _load_cifar(name, val_split, batch_size, class_imbalance, min_size, seed, data_root, num_workers)
    elif name == 'tinyimagenet':
        return _load_tinyimagenet(val_split, batch_size, seed, data_root, num_workers)
    else:
        raise ValueError(f"Unknown dataset: '{name}'. Expected one of: cifar10, cifar100, tinyimagenet, imagenet")


def _load_cifar(name, val_split, batch_size, class_imbalance, min_size, seed, data_root, num_workers):
    random.seed(seed)
    np.random.seed(seed)

    train_transform, eval_transform = _get_transforms(name)
    dataset_cls = torchvision.datasets.CIFAR10 if name == 'cifar10' else torchvision.datasets.CIFAR100

    full_trainset_aug = dataset_cls(root=data_root, train=True, download=True, transform=train_transform)
    full_trainset_eval = dataset_cls(root=data_root, train=True, download=True, transform=eval_transform)
    testset = dataset_cls(root=data_root, train=False, download=True, transform=eval_transform)
    class_names = full_trainset_aug.classes

    if class_imbalance is not None:
        num_classes = len(class_names)
        imbalanced_eval = create_imbalanced_dataset(full_trainset_eval, class_imbalance, num_classes, min_size, seed=seed)
        selected_indices = imbalanced_eval.indices
        full_trainset_aug = Subset(full_trainset_aug, selected_indices)
        full_trainset_eval = imbalanced_eval

    val_size = int(len(full_trainset_eval) * val_split)
    train_size = len(full_trainset_eval) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split_set = random_split(full_trainset_eval, [train_size, val_size], generator=generator)

    trainset = Subset(full_trainset_aug, train_split.indices)
    valset = val_split_set
    embedding_set = Subset(full_trainset_eval, train_split.indices)

    common_kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(IndexedImageDataset(trainset), batch_size=batch_size, shuffle=True, **common_kwargs)
    val_loader = DataLoader(IndexedImageDataset(valset), batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(IndexedImageDataset(testset), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    embedding_loader = DataLoader(IndexedImageDataset(embedding_set), batch_size=batch_size, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, embedding_loader, class_names


TINYIMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


class TinyImageNetVal(Dataset):
    def __init__(self, val_dir, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        ann_path = os.path.join(val_dir, 'val_annotations.txt')
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    raise ValueError(f"val_annotations.txt references wnid '{wnid}' not present in train/. "
                                     f"The download may be incomplete.")
                self.samples.append((_resolve_val_image(val_dir, wnid, fname), class_to_idx[wnid]))

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _ensure_tinyimagenet(data_root):
    root = os.path.join(data_root, 'tiny-imagenet-200')
    if os.path.isdir(os.path.join(root, 'train')) and os.path.isdir(os.path.join(root, 'val')):
        return root
    os.makedirs(data_root, exist_ok=True)
    print(f'Tiny-ImageNet not found at {root} - downloading (~240MB)...')
    download_and_extract_archive(TINYIMAGENET_URL, download_root=data_root, remove_finished=False)
    if not os.path.isdir(os.path.join(root, 'train')):
        raise RuntimeError(f'Download completed but {root}/train is missing - unexpected archive layout.')
    print(f'Tiny-ImageNet ready at {root}')
    return root


def _load_tinyimagenet(val_split, batch_size, seed, data_root, num_workers):
    random.seed(seed)
    np.random.seed(seed)

    root = _ensure_tinyimagenet(data_root)
    train_transform, eval_transform = _get_transforms('tinyimagenet')
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')


    full_trainset_aug = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    full_trainset_eval = torchvision.datasets.ImageFolder(train_dir, transform=eval_transform)
    class_names = full_trainset_aug.classes
    class_to_idx = full_trainset_aug.class_to_idx              # authoritative label mapping


    testset = TinyImageNetVal(val_dir, class_to_idx, transform=eval_transform)

    val_size = int(len(full_trainset_eval) * val_split)
    train_size = len(full_trainset_eval) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split_set = random_split(full_trainset_eval, [train_size, val_size], generator=generator)

    trainset = Subset(full_trainset_aug, train_split.indices)
    valset = val_split_set
    embedding_set = Subset(full_trainset_eval, train_split.indices)

    common_kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(IndexedImageDataset(trainset), batch_size=batch_size, shuffle=True, **common_kwargs)
    val_loader = DataLoader(IndexedImageDataset(valset), batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(IndexedImageDataset(testset), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    embedding_loader = DataLoader(IndexedImageDataset(embedding_set), batch_size=batch_size, shuffle=False,
                                   **common_kwargs)

    print(f'Tiny-ImageNet: {train_size} train / {val_size} val (from train/) / {len(testset)} test '
          f'(official val/), {len(class_names)} classes')
    return train_loader, val_loader, test_loader, embedding_loader, class_names

CIFAR_C_CORRUPTIONS_15 = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression')


class CIFARCDataset(Dataset):
    def __init__(self, root, corruption, severity, transform):
        data = np.load(os.path.join(root, f'{corruption}.npy'))
        labels = np.load(os.path.join(root, 'labels.npy'))
        sl = slice((severity - 1) * 10000, severity * 10000)
        self.images = data[sl]; self.targets = labels[sl].astype(int).tolist(); self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return self.transform(Image.fromarray(self.images[idx])), self.targets[idx]

def load_cifar_c(base_name, corruption, severity, batch_size=256, data_root='./data', num_workers=4):
    folder = 'CIFAR-10-C' if base_name == 'cifar10' else 'CIFAR-100-C'
    if not _ensure_corruption_set(folder, data_root):
        raise FileNotFoundError(f'{folder} unavailable')
    _, eval_transform = _get_transforms(base_name)
    ds = CIFARCDataset(os.path.join(data_root, folder), corruption, severity, eval_transform)
    return DataLoader(IndexedImageDataset(ds), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def load_svhn_ood(batch_size=256, data_root='./data', num_workers=4, normalize_as='cifar10', size=None):
    stats = {'cifar10': (CIFAR10_MEAN, CIFAR10_STD), 'cifar100': (CIFAR100_MEAN, CIFAR100_STD),
             'tinyimagenet': (TINYIMAGENET_MEAN, TINYIMAGENET_STD)}
    mean, std = stats[normalize_as]
    ops = ([transforms.Resize(size)] if size else []) + [transforms.ToTensor(), transforms.Normalize(mean, std)]

    ds = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transforms.Compose(ops))
    return DataLoader(IndexedImageDataset(ds), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def load_tinyimagenet_c(corruption, severity, batch_size=256, data_root='./data', num_workers=4):
    """
    The code downloads the original Tiny-ImageNet-C dataset, the frozen setting needs to be manually deleted.
    """
    import torchvision
    if not _ensure_corruption_set('Tiny-ImageNet-C', data_root):
        raise FileNotFoundError('Tiny-ImageNet-C unavailable')
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD)])
    train_dir = os.path.join(data_root, 'tiny-imagenet-200', 'train')
    wnids = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    c2i = {w: i for i, w in enumerate(wnids)}
    root = os.path.join(data_root, 'Tiny-ImageNet-C', corruption, str(severity))
    ds = torchvision.datasets.ImageFolder(root, transform=tf)
    inv = {v: k for k, v in ds.class_to_idx.items()}
    ds.samples = [(p, c2i[inv[l]]) for p, l in ds.samples]; ds.targets = [l for _, l in ds.samples]
    ds.class_to_idx = c2i
    return DataLoader(IndexedImageDataset(ds), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
