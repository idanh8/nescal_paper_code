import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split

from data.image_datasets import IndexedImageDataset

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class PretrainedViTB16(nn.Module):
    def __init__(self, num_classes, weights='IMAGENET1K_V1', **kwargs):
        super().__init__()
        backbone = torchvision.models.vit_b_16(weights=weights)
        self.feat_dim = backbone.hidden_dim
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.linear = nn.Linear(self.feat_dim, num_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        z = self.backbone(x)
        logits = self.linear(z)
        if embedding_layer is not None:
            return logits, z
        if not return_embeddings:
            return logits
        return logits, {'final': z, 'stage_minus_1': z, 'stage_minus_2': z}


def cifar224_loaders(name, seed, batch_size, data_root='./data', num_workers=8, val_split=0.1, dataset_cls=None):
    cls = dataset_cls or (torchvision.datasets.CIFAR100 if name == 'cifar100' else torchvision.datasets.CIFAR10)
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train_tf = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), *norm])
    eval_tf = T.Compose([T.Resize(224), *norm])
    train_aug = cls(root=data_root, train=True, download=True, transform=train_tf)
    train_eval = cls(root=data_root, train=True, download=True, transform=eval_tf)
    test = cls(root=data_root, train=False, download=True, transform=eval_tf)
    val_size = int(len(train_eval) * val_split)
    tr, va = random_split(train_eval, [len(train_eval) - val_size, val_size], generator=torch.Generator().manual_seed(seed))
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(IndexedImageDataset(Subset(train_aug, tr.indices)), batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(IndexedImageDataset(va), batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(IndexedImageDataset(test), batch_size=batch_size, shuffle=False, **common)
    embedding_loader = DataLoader(IndexedImageDataset(Subset(train_eval, tr.indices)), batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, embedding_loader, train_aug.classes
