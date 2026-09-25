import os, argparse
from experiments.common import (run_dir, RunTracker, train_one, write_manifest, load_json)
from data.image_datasets import load_dataset
import experiments.nescal as run_nescal

def _ckpt(run_root, seed, epoch):
    return os.path.join(run_root, 'XE_baseline', f'seed_{seed}', 'checkpoints', f'epoch_{epoch}.pt')

def run(run_root, cfg=run_nescal.CONFIG, manifest_name='ablations_manifest.json'):
    ds, arch, base = cfg['dataset'], cfg['arch_name'], cfg['base_out']
    mf = load_json(os.path.join(run_root, 'run_manifest.json'))
    E, lam, N = mf['best_onset'], mf['lambda_cosine'], cfg['num_epochs']
    seeds = cfg['ablation_seeds']; ts = run_root.rstrip('/').split(os.sep)[-1]


    arms = []
    for d in cfg['ablations']['divergence']: arms.append((f'abl_div_{d}', {'divergence_type': d}, True))
    for m in cfg['ablations']['mask']:       arms.append((f'abl_mask_{m}', {'mask_mode': m}, False))
    for k in cfg['ablations']['k']:          arms.append((f'abl_k{k}',      {'k': k}, False))
    for g in cfg['ablations'].get('graph', []): arms.append((f'abl_graph_{g}', {'graph_reducer': g}, False))
    penalty_target = cfg['ablations'].get('penalty_target', 3.0)

    tr = RunTracker(len(arms) * len(seeds) * (N - E), label=f'{ds}/{arch}:ablations')
    print(f'=== Ablations {ds}/{arch} | E*={E} λ*={lam} | {len(arms)} arms × {len(seeds)} seeds ===', flush=True)
    manifest = {'run_root': run_root, 'best_onset': E, 'lambda': lam, 'reportable': {}, 'diverged': []}

    manifest['errored'] = []
    for label, override, scale_match in arms:
        for seed in seeds:
            try:
                loaders = run_nescal._loaders(cfg, seed)
                lp = run_nescal._ncl_params(cfg, lam)
                lp['secondary_params'].update(override)
                root = run_dir(base, ds, arch, ts, label, seed)
                extra = dict(loss_params=lp, resume_from=_ckpt(run_root, seed, E),
                             activation_params={'strategy': 'fixed', 'fixed_epoch': E},
                             lambda_kind='cosine', lambda_decay='cosine')
                if scale_match:
                    extra['lambda_from_penalty_target'] = penalty_target
                val, test, div, rp = train_one(tr, label=f'{label} seed={seed}', epochs_to_train=N - E,
                                               run_kwargs=run_nescal._base_kwargs(cfg, loaders, seed, root, **extra))
                manifest['reportable'].setdefault(label, {})[seed] = rp
                if div: manifest['diverged'].append({'arm': label, 'seed': seed})
            except Exception as e:
                print(f'  !! {label} seed={seed} ERRORED ({type(e).__name__}: {e}) — skipping, continuing.', flush=True)
                manifest['errored'].append({'arm': label, 'seed': seed, 'error': f'{type(e).__name__}: {e}'})
            write_manifest(os.path.join(run_root, manifest_name), manifest)

    print('=== Ablations complete ===', flush=True)
