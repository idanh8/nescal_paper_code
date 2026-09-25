import os, argparse, pickle
from experiments.common import load_json
from handlers.trainer import Trainer, evaluate_corruptions, evaluate_ood_separation
from data.image_datasets import CIFAR_C_CORRUPTIONS_15
CORRUPTIONS_15 = CIFAR_C_CORRUPTIONS_15

def _reportable_runs(run_root):
    out = {}
    for mf in ('run_manifest.json', 'competitors_manifest.json'):
        p = os.path.join(run_root, mf)
        if os.path.exists(p):
            for label, per_seed in load_json(p).get('reportable', {}).items():
                out.setdefault(label, {}).update({int(k): v for k, v in per_seed.items()})
    return out

def run(run_root, dataset, do_corruption=True, do_ood=True, data_root='./data', num_bins=15, batch_size=256,
        only_labels=None, skip_existing=False):
    if do_corruption:
        folder = {'cifar10': 'CIFAR-10-C', 'cifar100': 'CIFAR-100-C',
                  'tinyimagenet': 'Tiny-ImageNet-C'}.get(dataset)
        probe = os.path.join(data_root, folder or '', 'gaussian_noise.npy' if dataset != 'tinyimagenet'
                             else os.path.join('gaussian_noise', '1'))
        if not folder or not os.path.exists(probe):
            print(f'[extras] corruption data not found ({probe}) — SKIPPING corruption. '
                  f'Download {folder} into {data_root}/ and re-run extras to add it.', flush=True)
            do_corruption = False
    runs = _reportable_runs(run_root)
    if only_labels is not None: runs = {k: v for k, v in runs.items() if k in only_labels}
    total = sum(len(v) for v in runs.values()); done = 0
    print(f'=== Extras over {total} runs (corruption={do_corruption}, ood={do_ood}) ===', flush=True)
    ood_size = 64 if dataset == 'tinyimagenet' else None
    for label, per_seed in runs.items():
        for seed, rp in per_seed.items():
            done += 1
            if not os.path.exists(os.path.join(rp, 'model_state.pth')):
                print(f'[{done}/{total}] SKIP {label} seed={seed}', flush=True); continue
            _dc = do_corruption and not (skip_existing and os.path.exists(os.path.join(rp, 'corruption_results.pkl')))
            _do = do_ood and not (skip_existing and os.path.exists(os.path.join(rp, 'ood_results.pkl')))
            if not (_dc or _do):
                print(f'[{done}/{total}] {label} seed={seed}: extras already done', flush=True); continue
            m = Trainer.load_model(rp)
            if _dc and dataset in ('cifar10', 'cifar100', 'tinyimagenet'):
                try:
                    corr = evaluate_corruptions(m, dataset, CORRUPTIONS_15, (1, 2, 3, 4, 5),
                                                num_bins, data_root=data_root, batch_size=batch_size)
                    with open(os.path.join(rp, 'corruption_results.pkl'), 'wb') as f: pickle.dump(corr, f)
                except Exception as e:
                    print(f'  !! corruption {label} seed={seed} failed ({type(e).__name__}: {e}) — skipping.', flush=True)
            if _do and dataset in ('cifar10', 'cifar100', 'tinyimagenet'):
                try:
                    ood = evaluate_ood_separation(m, dataset, data_root=data_root, batch_size=batch_size, size=ood_size)
                    with open(os.path.join(rp, 'ood_results.pkl'), 'wb') as f: pickle.dump(ood, f)
                except Exception as e:
                    print(f'  !! ood {label} seed={seed} failed ({type(e).__name__}: {e}) — skipping.', flush=True)
            print(f'[{done}/{total}] {label} seed={seed} done', flush=True)
    print('=== Extras complete ===', flush=True)
