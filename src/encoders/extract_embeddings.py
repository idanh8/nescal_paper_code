import os, argparse
import numpy as np
import torch

FROZEN_ROOT = './data/frozen'


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_text_encoder(name):
    """Returns (tokenizer, model, enc_dim). Mean-pooling is applied in extract_text."""
    from transformers import AutoTokenizer, AutoModel
    hub = {'roberta': 'roberta-base', 'bert': 'bert-base-uncased'}
    if name not in hub:
        raise ValueError(f"unknown text encoder '{name}'")
    tok = AutoTokenizer.from_pretrained(hub[name])
    m = AutoModel.from_pretrained(hub[name]).eval().to(_device())
    return tok, m, m.config.hidden_size


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
        payload = extract_text(dataset, encoder, data_root, task=task)
        torch.save(payload, base_fp)
        print(f'[extract] saved base.pt (enc_dim={payload["enc_dim"]}, '
              f'{payload["trainval_emb"].shape[0]} trainval / {payload["test_emb"].shape[0]} test) -> {base_fp}', flush=True)
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--encoder', required=True, choices=['roberta', 'bert'])
    ap.add_argument('--task', required=True, help='amazon task: bin_15 | bin_24 | multi_5')
    ap.add_argument('--data_root', default='./data')
    ap.add_argument('--batch_size', type=int, default=256)
    a = ap.parse_args()
    run(a.dataset, a.encoder, a.data_root, a.batch_size, task=a.task)
