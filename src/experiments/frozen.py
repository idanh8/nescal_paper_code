import os, copy, json, argparse, pickle
import numpy as np

import experiments.nescal as run_nescal
import experiments.competitors as run_competitors
import experiments.posthoc as run_posthoc
import experiments.ablations as run_ablations
import experiments.reporting as run_all
from experiments.common import timestamp, run_dir, RunTracker, train_one, load_pkl, load_json
from data.frozen_datasets import (load_frozen, frozen_dim, load_frozen_corruption, load_frozen_ood)
import encoders.extract_embeddings as extract_embeddings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CIFAR_C_15 = extract_embeddings.CIFAR_C_15


CONFIG = __import__('reproduction.recipes', fromlist=['source_config']).source_config('frozen')


def _ds_tag(dataset, task):
    return f'{dataset}_{task}' if (dataset == 'amazon' and task) else dataset


def _make_cfg(dataset, encoder, hidden_dims, task=None, dropout=0.0):
    import torch

    ds_tag = _ds_tag(dataset, task)
    base_dir = CONFIG['embedding_dir']
    payload = torch.load(
        os.path.join(base_dir, 'base.pt'), map_location='cpu'
    )

    cfg = copy.deepcopy(CONFIG)
    variant = os.path.basename(base_dir) if CONFIG['finetune'] else 'pretrained'
    cfg['dataset'] = f'{ds_tag}_{encoder}_{variant}_split{CONFIG["split_seed"]}'
    cfg['arch_name'] = 'frozen_mlp'
    cfg['model_params'] = {
        'arch': 'frozen_enc_mlp',
        'num_classes': len(payload['class_names']),
        'in_dim': int(payload['enc_dim']),
        'hidden_dims': tuple(hidden_dims),
        'norm_type': 'layer',
        'dropout': dropout,
    }
    cfg['loader_fn'] = lambda s: load_frozen(
        ds_tag, encoder,
        val_split=cfg['val_split'],
        batch_size=cfg['batch_size'],
        seed=s,
        num_workers=cfg['num_workers'],
        base_dir=base_dir,
        split_seed=cfg['split_seed'],
    )
    cfg['_dataset_raw'], cfg['_encoder'], cfg['_task'] = dataset, encoder, task
    return cfg


def _eval_head_on_loader(model, loader):
    import torch
    dev = model.device; model.model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(dev), batch[1]
            logits = model.model(x)
            ps.append(torch.softmax(logits, 1).cpu().numpy()); ys.append(np.asarray(y))
    return np.concatenate(ps), np.concatenate(ys)


def frozen_extras(run_root, dataset, encoder, cfg, task=None):
    from handlers.trainer import Trainer
    from metrics.calibration_metrics import ece_mce, brier_score
    import torch, torch.nn.functional as F
    from scipy.stats import rankdata
    ds_tag = _ds_tag(dataset, task)

    def _metrics(probs, labels):
        nll = float(F.nll_loss(torch.log(torch.tensor(probs).clamp_min(1e-12)), torch.tensor(labels)).item())
        return {'ece': float(ece_mce(probs, labels, num_bins=cfg['num_bins'])[0]), 'nll': nll,
                'accuracy': float((probs.argmax(1) == labels).mean()), 'brier_score': float(brier_score(probs, labels))}

    runs = run_all._reportable(run_root)
    for label, per_seed in runs.items():
        for seed, rp in per_seed.items():
            if not rp or not os.path.exists(os.path.join(rp, 'model_state.pth')):
                continue
            m = Trainer.load_model(rp)
            if cfg['do_robustness']:
                cres = {}
                for c in CIFAR_C_15:
                    cres[c] = {}
                    for sev in (1, 2, 3, 4, 5):
                        try:
                            ld = load_frozen_corruption(ds_tag, encoder, c, sev, batch_size=cfg['batch_size'], base_dir=cfg['embedding_dir'])
                        except FileNotFoundError:
                            continue
                        p, y = _eval_head_on_loader(m, ld); cres[c][sev] = _metrics(p, y)
                with open(os.path.join(rp, 'corruption_results.pkl'), 'wb') as f: pickle.dump(cres, f)
            if cfg['do_ood']:
                try:
                    id_probs = load_pkl(os.path.join(rp, 'test_data.pkl'))['test_conf_vectors']
                    ood_ld = load_frozen_ood(ds_tag, encoder, batch_size=cfg['batch_size'], base_dir=cfg['embedding_dir'])
                    ood_probs, _ = _eval_head_on_loader(m, ood_ld)
                    id_msp = np.asarray(id_probs).max(1); ood_msp = ood_probs.max(1)
                    scores = np.concatenate([id_msp, ood_msp]); is_id = np.concatenate([np.ones_like(id_msp), np.zeros_like(ood_msp)])
                    r = rankdata(scores); n_id = is_id.sum(); n_ood = (1 - is_id).sum()
                    auroc = float((r[is_id == 1].sum() - n_id * (n_id + 1) / 2) / (n_id * n_ood))
                    thr = np.quantile(id_msp, 0.05); fpr95 = float((ood_msp >= thr).mean())
                    with open(os.path.join(rp, 'ood_results.pkl'), 'wb') as f:
                        pickle.dump({'msp': {'auroc': auroc, 'fpr_at_95_tpr': fpr95}}, f)
                except Exception as e:
                    print(f'  [frozen_extras] OOD skipped for {label} seed={seed} ({type(e).__name__}: {e})', flush=True)
    print('[frozen_extras] robustness/OOD written', flush=True)


def run_main(dataset, encoder, ts, task=None, resume_cfg=None):
    cfg = (resume_cfg if resume_cfg is not None else _make_cfg(dataset, encoder, CONFIG['head_hidden_dims'], task=task))
    run_root = os.path.join(cfg['base_out'], cfg['dataset'], cfg['arch_name'], ts)
    if resume_cfg is None:
        run_nescal.run(cfg, ts)
        run_competitors.run_root_ref[0] = run_root
        run_competitors.run(ts, cfg=cfg, reuse_baseline_ts=ts)
        run_ablations.run(run_root, cfg)
    run_posthoc.POSTHOC = ('ts', 'local', 'dc', 'dac', 'lore')
    run_posthoc.run(run_root, cfg['dataset'], cfg['arch_name'], data_root=cfg['data_root'], num_bins=cfg['num_bins'], batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], loader_fn=cfg['loader_fn'],)
    if (cfg['do_robustness'] or cfg['do_ood']) and dataset in ('cifar10', 'cifar100') and encoder in ('vit_b16', 'dinov2'):
        frozen_extras(run_root, dataset, encoder, cfg, task=task)
    run_dropout_ablation(dataset, encoder, ts, task=task)
    run_all.assemble_report(run_root)
    return run_root


def run_headline_only(dataset, encoder, ts, task=None, resume_cfg=None):
    cfg = (resume_cfg if resume_cfg is not None else _make_cfg(dataset, encoder, CONFIG['head_hidden_dims'], task=task))
    run_root = os.path.join(cfg['base_out'], cfg['dataset'], cfg['arch_name'], ts)
    if resume_cfg is None:
        run_nescal.run(cfg, ts)
    run_posthoc.POSTHOC = ('local',)
    run_posthoc.run(run_root, cfg['dataset'], cfg['arch_name'], data_root=cfg['data_root'], num_bins=cfg['num_bins'], batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], loader_fn=cfg['loader_fn'],)
    run_all.assemble_report(run_root)
    return run_root


def run_dropout_ablation(dataset, encoder, ts, task=None):
    rows = []
    for do in (0.0, 0.1, 0.3):
        cfg = _make_cfg(dataset, encoder, CONFIG['head_hidden_dims'], task=task, dropout=do)
        cfg['confirmatory_seeds'] = []
        sub_ts = f'{ts}_dropout{do}'
        run_root = os.path.join(cfg['base_out'], cfg['dataset'], cfg['arch_name'], sub_ts)
        run_nescal.run(cfg, sub_ts)
        def _v(mk):
            per = run_all._reportable(run_root).get('ncl_FINAL_cosine', {}); vals = []
            for seed, rp in per.items():
                p = os.path.join(rp, 'test_results.pkl')
                if os.path.exists(p): vals.append(load_pkl(p).get(mk))
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else float('nan')
        rows.append({'dropout': do, 'nll': _v('nll'), 'ece': _v('ece'), 'accuracy': _v('accuracy')})
        print(f'  dropout={do}: {rows[-1]}', flush=True)
    out = os.path.join(CONFIG['base_out'], cfg['dataset'], 'frozen_mlp', f'{ts}_dropout_ablation.json')
    with open(out, 'w') as f: json.dump(rows, f, indent=2, default=str)
    print('\n### Dropout ablation (NeSCal, test)\n\n| dropout | nll | ece | accuracy |\n|---|---|---|---|')
    for r in rows: print(f'| {r["dropout"]} | {r["nll"]:.4f} | {r["ece"]:.4f} | {r["accuracy"]:.4f} |')
