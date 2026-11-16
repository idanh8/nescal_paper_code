import os, argparse
import numpy as np
import torch

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FROZEN_ROOT = './data/frozen'


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_image_encoder(name):
    dev = _device()
    if name == 'vit_b16':
        from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
        m = vit_b_16(weights=ViT_B_16_Weights.DEFAULT); m.heads = torch.nn.Identity()
        m.eval().to(dev)
        return m, (lambda mod, x: mod(x)), 768, 224
    if name == 'dinov2':
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        m.eval().to(dev)
        return m, (lambda mod, x: mod(x)), 768, 224
    raise ValueError(f"unknown image encoder '{name}'")


def build_text_encoder(name):
    """Returns (tokenizer, model, enc_dim). Mean-pooling is applied in extract_text."""
    from transformers import AutoTokenizer, AutoModel
    hub = {'roberta': 'roberta-base', 'bert': 'bert-base-uncased'}
    if name not in hub:
        raise ValueError(f"unknown text encoder '{name}'")
    tok = AutoTokenizer.from_pretrained(hub[name])
    m = AutoModel.from_pretrained(hub[name]).eval().to(_device())
    return tok, m, m.config.hidden_size


def _image_eval_transform(resize_px):
    import torchvision.transforms as T
    return T.Compose([T.Resize((resize_px, resize_px)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@torch.no_grad()
def _encode_image_dataset(dataset, module, fwd, resize_px, batch_size=256, num_workers=4):
    from torch.utils.data import DataLoader
    dev = _device()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    embs, labs = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(dev, non_blocking=True)
        e = fwd(module, x).float().cpu()
        embs.append(e); labs.append(torch.as_tensor(y))
    return torch.cat(embs), torch.cat(labs).long()


def _cifar_raw(dataset, resize_px, data_root='./data'):
    import torchvision
    tf = _image_eval_transform(resize_px)
    cls = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    train = cls(root=data_root, train=True, download=True, transform=tf)
    test = cls(root=data_root, train=False, download=True, transform=tf)
    return train, test, train.classes


class _ImageNetParquetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds; self.tf = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        row = self.ds[i]
        img = row['image'].convert('RGB')
        return self.tf(img), int(row['label'])


def _imagenet_raw(resize_px, data_root='./data'):
    from datasets import load_dataset as hf_load
    imagenet_path = os.path.join(data_root, 'imagenet-1k')
    if not os.path.isdir(imagenet_path):
        raise FileNotFoundError(f'{imagenet_path} not found — download with: '
                                'huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset '
                                f'--local-dir {imagenet_path}')
    ds = hf_load('parquet', data_dir=os.path.join(imagenet_path, 'data'))
    tf = _image_eval_transform(resize_px)
    train = _ImageNetParquetDataset(ds['train'], tf)
    val = _ImageNetParquetDataset(ds['validation'], tf)
    classes_py = os.path.join(imagenet_path, 'classes.py')
    if os.path.exists(classes_py):
        import importlib.util
        spec = importlib.util.spec_from_file_location('imagenet_classes', classes_py)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        if hasattr(mod, 'IMAGENET2012_CLASSES'):
            c = mod.IMAGENET2012_CLASSES
            class_names = list(c.values()) if isinstance(c, dict) else list(c)
        else:
            class_names = [f'cls_{i}' for i in range(1000)]
    else:
        class_names = [f'cls_{i}' for i in range(1000)]
    print(f'[imagenet] {len(train)} train / {len(val)} val / {len(class_names)} classes', flush=True)
    return train, val, class_names[:1000]


def extract_image(dataset, encoder, data_root='./data', batch_size=256, num_workers=4):
    module, fwd, enc_dim, resize_px = build_image_encoder(encoder)
    if dataset == 'imagenet':
        train, test, class_names = _imagenet_raw(resize_px, data_root)
    else:
        train, test, class_names = _cifar_raw(dataset, resize_px, data_root)
    print(f'[extract] {dataset}/{encoder}: encoding {len(train)} train + {len(test)} test at {resize_px}px', flush=True)
    tr_emb, tr_lab = _encode_image_dataset(train, module, fwd, resize_px, batch_size, num_workers)
    te_emb, te_lab = _encode_image_dataset(test, module, fwd, resize_px, batch_size, num_workers)
    return {'trainval_emb': tr_emb, 'trainval_lab': tr_lab, 'test_emb': te_emb, 'test_lab': te_lab,
            'class_names': list(class_names), 'enc_dim': enc_dim}


@torch.no_grad()
def _encode_texts(texts, tok, model, enc_dim, batch_size=64, max_len=256):
    dev = _device(); embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(dev)
        out = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        embs.append(pooled.float().cpu())
    return torch.cat(embs)


def extract_text(dataset, encoder, data_root='./data', task=None):
    """Text datasets (20ng, sst2, amazon). Uses data.text_datasets.load_text_dataset."""
    tok, model, enc_dim = build_text_encoder(encoder)
    from data.text_datasets import load_text_dataset
    (tr_txt, tr_lab), (te_txt, te_lab), class_names = load_text_dataset(dataset, task=task, data_root=data_root)
    print(f'[extract] {dataset}({task})/{encoder}: encoding {len(tr_txt)} train + {len(te_txt)} test texts', flush=True)
    tr_emb = _encode_texts(tr_txt, tok, model, enc_dim)
    te_emb = _encode_texts(te_txt, tok, model, enc_dim)
    return {'trainval_emb': tr_emb, 'trainval_lab': torch.as_tensor(tr_lab).long(),
            'test_emb': te_emb, 'test_lab': torch.as_tensor(te_lab).long(),
            'class_names': list(class_names), 'enc_dim': enc_dim}


CIFAR_C_15 = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
              'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
              'elastic_transform', 'pixelate', 'jpeg_compression')


class _NpyCorruption(torch.utils.data.Dataset):
    def __init__(self, root, corruption, severity, resize_px):
        from PIL import Image
        self.data = np.load(os.path.join(root, f'{corruption}.npy'))[(severity - 1) * 10000: severity * 10000]
        self.lab = np.load(os.path.join(root, 'labels.npy'))[(severity - 1) * 10000: severity * 10000].astype(int)
        self.tf = _image_eval_transform(resize_px); self._Image = Image
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.tf(self._Image.fromarray(self.data[i])), int(self.lab[i])


def extract_corruptions(dataset, encoder, out_dir, data_root='./data', batch_size=256, num_workers=4, encoder_bundle=None):
    module, fwd, enc_dim, resize_px = (encoder_bundle if encoder_bundle is not None else build_image_encoder(encoder))
    folder = 'CIFAR-10-C' if dataset == 'cifar10' else 'CIFAR-100-C'
    croot = os.path.join(data_root, folder)
    cdir = os.path.join(out_dir, 'corruptions'); os.makedirs(cdir, exist_ok=True)
    for c in CIFAR_C_15:
        for sev in (1, 2, 3, 4, 5):
            fp = os.path.join(cdir, f'{c}_{sev}.pt')
            if os.path.exists(fp): continue
            ds = _NpyCorruption(croot, c, sev, resize_px)
            emb, lab = _encode_image_dataset(ds, module, fwd, resize_px, batch_size, num_workers)
            torch.save({'emb': emb, 'lab': lab}, fp)
        print(f'[extract] corruptions done: {c}', flush=True)


def extract_svhn_ood(encoder, out_dir, data_root='./data', batch_size=256, num_workers=4, encoder_bundle=None):
    import torchvision
    module, fwd, enc_dim, resize_px = (encoder_bundle if encoder_bundle is not None else build_image_encoder(encoder))
    tf = _image_eval_transform(resize_px)
    ds = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=tf)
    emb, lab = _encode_image_dataset(ds, module, fwd, resize_px, batch_size, num_workers)
    torch.save({'emb': emb, 'lab': lab}, os.path.join(out_dir, 'svhn_ood.pt'))
    print('[extract] svhn OOD done', flush=True)


def run(dataset, encoder, data_root='./data', robustness=False, batch_size=256, num_workers=4, task=None, finetune=False, split_seed=42, val_split=0.1, ft_cfg=None, output_root=None):
    if finetune:
        from encoders.finetune_embeddings import prepare_finetuned
        return prepare_finetuned(
            dataset, encoder, data_root=data_root, task=task,
            split_seed=split_seed, val_split=val_split, ft_cfg=ft_cfg,
            robustness=robustness, num_workers=num_workers, output_root=output_root)
    ds_tag = f'{dataset}_{task}' if (dataset == 'amazon' and task) else dataset
    out_dir = os.path.join(output_root or FROZEN_ROOT, f'{ds_tag}_{encoder}')
    os.makedirs(out_dir, exist_ok=True)
    base_fp = os.path.join(out_dir, 'base.pt')
    if os.path.exists(base_fp):
        print(f'[extract] base.pt exists at {out_dir} — skipping base extraction', flush=True)
    else:
        is_text = encoder in ('roberta', 'bert')
        payload = extract_text(dataset, encoder, data_root, task=task) if is_text \
            else extract_image(dataset, encoder, data_root, batch_size, num_workers)
        torch.save(payload, base_fp)
        print(f'[extract] saved base.pt (enc_dim={payload["enc_dim"]}, '
              f'{payload["trainval_emb"].shape[0]} trainval / {payload["test_emb"].shape[0]} test) -> {base_fp}', flush=True)
    if robustness and dataset in ('cifar10', 'cifar100') and encoder in ('vit_b16', 'dinov2'):
        extract_corruptions(dataset, encoder, out_dir, data_root, batch_size, num_workers)
        extract_svhn_ood(encoder, out_dir, data_root, batch_size, num_workers)
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--encoder', required=True, choices=['vit_b16', 'dinov2', 'roberta', 'bert'])
    ap.add_argument('--task', default=None, help='amazon task: bin_15 | bin_24 | multi_5')
    ap.add_argument('--data_root', default='./data')
    ap.add_argument('--robustness', action='store_true', help='also extract CIFAR-C + SVHN embeddings')
    ap.add_argument('--batch_size', type=int, default=256)
    a = ap.parse_args()
    run(a.dataset, a.encoder, a.data_root, a.robustness, a.batch_size, task=a.task)
