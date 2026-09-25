import copy
import hashlib
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split


class _TextCollate:
    def __init__(self, tokenizer, max_len):
        self.tokenizer, self.max_len = tokenizer, max_len

    def __call__(self, rows):
        texts, labels = zip(*rows)
        x = self.tokenizer(list(texts), padding=True, truncation=True,
                           max_length=self.max_len, return_tensors='pt')
        return dict(x), torch.tensor(labels, dtype=torch.long)


def _text_forward(model, x):
    hidden = model(**x).last_hidden_state
    mask = x['attention_mask'].unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)


def _move(x, device):
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=True) for k, v in x.items()}
    return x.to(device, non_blocking=True)


def _save(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def _fit(model, forward, dim, classes, train_loader, val_loader, cfg, checkpoint):
    device = next(model.parameters()).device
    head = torch.nn.Sequential(torch.nn.Dropout(0.1),
                               torch.nn.Linear(dim, classes)).to(device)
    model.requires_grad_(True)
    params = list(model.parameters()) + list(head.parameters())

    opt = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.ndim > 1],
         'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
        {'params': [p for p in model.parameters() if p.ndim <= 1],
         'lr': cfg['lr'], 'weight_decay': 0.0},
        {'params': head.parameters(), 'lr': cfg['head_lr'],
         'weight_decay': cfg['weight_decay']},
    ])
    total = min(cfg['epochs'] * len(train_loader), cfg['max_steps'])
    warmup = max(1, int(0.1 * total))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    amp = device.type == 'cuda'
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    best, stale, step = float('inf'), 0, 0
    best_path = checkpoint + '.in_progress'
    for epoch in range(cfg['epochs']):
        model.train(); head.train()
        for x, y in train_loader:
            x, y = _move(x, device), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                loss = F.cross_entropy(head(forward(model, x)), y)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite fine-tuning loss; no completed checkpoint saved.')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            old_scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= old_scale:
                sched.step()
            step += 1
            if step % 100 == 0:
                print(f'[finetune] step {step}/{total}, loss={loss.item():.4f}', flush=True)
            if step >= total:
                break

        model.eval(); head.eval()
        total_loss = torch.zeros((), device=device)
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = _move(x, device), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    logits = head(forward(model, x))
                total_loss += F.cross_entropy(logits.float(), y, reduction='sum')
                count += len(y)
        nll = total_loss.item() / count
        if not math.isfinite(nll):
            raise RuntimeError('Non-finite validation NLL; no completed checkpoint saved.')
        print(f'[finetune] epoch {epoch + 1}, step {step}/{total}, val NLL={nll:.5f}', flush=True)
        if nll < best:
            best, stale = nll, 0
            _save({'encoder': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'val_nll': nll, 'best_step': step}, best_path)
        else:
            stale += 1
        if stale >= cfg['patience'] or step >= total:
            break

    os.replace(best_path, checkpoint)
    opt.zero_grad(set_to_none=True)


def prepare_finetuned(dataset, encoder, data_root='./data', task=None,
                      split_seed=42, val_split=0.1, ft_cfg=None,
                      robustness=False, num_workers=4, output_root=None):
    from encoders import extract_embeddings as ex

    is_text = encoder in ('bert', 'roberta')
    allowed = ('20ng', 'sst2', 'amazon') if is_text else ('cifar10', 'cifar100', 'imagenet')
    if encoder not in ('bert', 'roberta', 'vit_b16', 'dinov2') or dataset not in allowed:
        raise ValueError(f'Unsupported encoder/dataset pair: {encoder}/{dataset}')
    task = (task or 'multi_5') if dataset == 'amazon' else None
    cfg = dict(epochs=3, batch_size=32, lr=2e-5 if is_text else 1e-5,
               head_lr=1e-3, weight_decay=0.01, max_steps=6000 if is_text else 3000,
               patience=2, max_len=256, extract_batch_size=64, cache_version=1)
    unknown = set(ft_cfg or {}) - set(cfg)
    if unknown:
        raise ValueError(f'Unknown fine-tuning settings: {sorted(unknown)}')
    cfg.update(ft_cfg or {})
    for key in ('epochs', 'batch_size', 'max_steps', 'patience', 'max_len', 'extract_batch_size'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < val_split < 1:
        raise ValueError('val_split must lie strictly between 0 and 1')
    meta = dict(dataset=dataset, encoder=encoder, task=task,
                split_seed=split_seed, val_split=val_split, **cfg)
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    ds_tag = f'{dataset}_{task}' if dataset == 'amazon' else dataset
    out_dir = os.path.join(output_root or data_root, ds_tag, encoder, f'finetuned_{digest}')
    os.makedirs(out_dir, exist_ok=True)
    base_fp = os.path.join(out_dir, 'base.pt')
    checkpoint = os.path.join(out_dir, 'encoder.pt')
    want_rob = robustness and dataset in ('cifar10', 'cifar100')
    extra_paths = ([os.path.join(out_dir, 'svhn_ood.pt')] + [
        os.path.join(out_dir, 'corruptions', f'{c}_{s}.pt')
        for c in ex.CIFAR_C_15 for s in range(1, 6)]) if want_rob else []
    if os.path.exists(base_fp) and all(os.path.exists(p) for p in extra_paths):
        print(f'[finetune] reusing {base_fp}', flush=True)
        return out_dir
    if os.path.exists(base_fp) and not os.path.exists(checkpoint):
        raise FileNotFoundError('encoder.pt is required to finish this cache\'s robustness features')

    random.seed(split_seed); np.random.seed(split_seed); torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)
    if is_text:
        from data.text_datasets import load_text_dataset
        tok, model, dim = ex.build_text_encoder(encoder)


        (tr_text, tr_lab), (te_text, te_lab), names = load_text_dataset(
            dataset, task=task, data_root=data_root)
        pool = list(zip(tr_text, tr_lab))
        fit_pool = pool
        collate = _TextCollate(tok, cfg['max_len'])
        forward = _text_forward
    else:
        model, forward, dim, size = ex.build_image_encoder(encoder)
        pool, test, names = (ex._imagenet_raw(size, data_root) if dataset == 'imagenet'
                             else ex._cifar_raw(dataset, size, data_root))

        import torchvision.transforms as T
        fit_pool = copy.copy(pool)
        first = ([T.RandomResizedCrop(size, scale=(0.8, 1.0))] if dataset == 'imagenet'
                 else [T.RandomCrop(32, padding=4), T.Resize((size, size))])
        train_tf = T.Compose(first + [T.RandomHorizontalFlip(), T.ToTensor(),
                            T.Normalize(ex.IMAGENET_MEAN, ex.IMAGENET_STD)])
        if dataset == 'imagenet':
            fit_pool.tf = train_tf
        else:
            fit_pool.transform = train_tf
        collate = None

    n_val = int(len(pool) * val_split)
    if not 0 < n_val < len(pool):
        raise ValueError('The fixed split must have nonempty training and validation sets')
    tr, va = random_split(pool, [len(pool) - n_val, n_val],
                          generator=torch.Generator().manual_seed(split_seed))
    if not os.path.exists(checkpoint):
        workers = 0 if is_text else num_workers
        common = dict(batch_size=cfg['batch_size'], num_workers=workers,
                      collate_fn=collate, pin_memory=torch.cuda.is_available())
        train_loader = DataLoader(Subset(fit_pool, tr.indices), shuffle=True,
                                  generator=torch.Generator().manual_seed(split_seed), **common)
        val_loader = DataLoader(va, shuffle=False, **common)
        _fit(model, forward, dim, len(names), train_loader, val_loader, cfg, checkpoint)
    else:
        print(f'[finetune] reusing encoder checkpoint {checkpoint}', flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['encoder'])
    model.eval().requires_grad_(False)

    if not os.path.exists(base_fp):
        bs = cfg['extract_batch_size']
        if is_text:
            tr_emb = ex._encode_texts(tr_text, tok, model, dim, bs, cfg['max_len'])
            te_emb = ex._encode_texts(te_text, tok, model, dim, bs, cfg['max_len'])
        else:
            tr_emb, tr_lab = ex._encode_image_dataset(pool, model, forward, size, bs, num_workers)
            te_emb, te_lab = ex._encode_image_dataset(test, model, forward, size, bs, num_workers)
        _save(dict(trainval_emb=tr_emb, trainval_lab=torch.as_tensor(tr_lab).long(),
                   test_emb=te_emb, test_lab=torch.as_tensor(te_lab).long(),
                   class_names=list(names), enc_dim=dim,
                   train_indices=torch.tensor(tr.indices), val_indices=torch.tensor(va.indices),
                   split_seed=split_seed, val_split=val_split, finetune_config=meta), base_fp)
        print(f'[finetune] saved {base_fp}', flush=True)
    if want_rob:
        bundle = (model, forward, dim, size)
        ex.extract_corruptions(dataset, encoder, out_dir, data_root,
                               cfg['extract_batch_size'], num_workers, encoder_bundle=bundle)
        if not os.path.exists(os.path.join(out_dir, 'svhn_ood.pt')):
            ex.extract_svhn_ood(encoder, out_dir, data_root, cfg['extract_batch_size'],
                                num_workers, encoder_bundle=bundle)
    return out_dir
