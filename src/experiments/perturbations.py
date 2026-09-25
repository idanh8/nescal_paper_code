import os, argparse
from experiments.common import run_dir, RunTracker, train_one, write_manifest, load_json
from robustness.perturbations import build_perturbed_cifar_loaders
import experiments.nescal as run_nescal

WAVE5 = {
    'longtail_factors': [10, 50, 100],
    'longtail_arms': {'js': {'divergence_type': 'js'}, 'fisher': {'divergence_type': 'fisher'},
                      'uniform': {'divergence_type': 'l2'}},
    'noise_rates': [0.2, 0.4], 'noise_kind': 'symmetric',
    'noise_arms': {'js': {'divergence_type': 'js'}},
    'seeds': [18, 42],
    'penalty_target': 3.0,
}

def run(run_root, cfg=run_nescal.CONFIG, w5=WAVE5):
    ds, arch, base = cfg['dataset'], cfg['arch_name'], cfg['base_out']
    assert ds in ('cifar10', 'cifar100'), 'Wave 5 is CIFAR-only.'
    mf = load_json(os.path.join(run_root, 'run_manifest.json'))
    E, lam, N = mf['best_onset'], mf['lambda_cosine'], cfg['num_epochs']
    ts = run_root.rstrip('/').split(os.sep)[-1]; seeds = w5['seeds']

    planned = (len(w5['longtail_factors']) * len(seeds) * (N + len(w5['longtail_arms']) * (N - E))
               + len(w5['noise_rates']) * len(seeds) * (N + len(w5['noise_arms']) * (N - E)))
    tr = RunTracker(planned, label=f'{ds}/{arch}:wave5')
    manifest = {'run_root': run_root, 'best_onset': E, 'lambda': lam, 'penalty_target': w5['penalty_target'],
                'reportable': {}, 'diverged': [], 'errored': []}

    def _block(tag, pert, arms, scale_match=None):
        for seed in seeds:
            loaders = build_perturbed_cifar_loaders(ds, pert, seed, batch_size=cfg['batch_size'],
                                                    data_root=cfg['data_root'], num_workers=cfg['num_workers'])
            broot = run_dir(base, ds, arch, ts, f'w5_{tag}_XE', seed)
            try:
                _, _, _, brp = train_one(tr, label=f'w5 {tag} XE seed={seed}', epochs_to_train=N,
                    run_kwargs=run_nescal._base_kwargs(cfg, loaders, seed, broot,
                        loss_params={'primary_loss': 'XE', 'secondary_loss': None},
                        activation_params=None, log_graph_maturity=False, checkpoint_epochs=[E]))
                manifest['reportable'].setdefault(f'w5_{tag}_XE', {})[seed] = brp
            except Exception as e:
                print(f'  !! {tag} XE seed={seed} ERRORED ({type(e).__name__}: {e}) — no ckpt, skipping its arms.',
                      flush=True)
                manifest['errored'].append({'run': f'w5_{tag}_XE', 'seed': seed, 'error': f'{type(e).__name__}: {e}'})
                write_manifest(os.path.join(run_root, 'wave5_manifest.json'), manifest)
                continue
            write_manifest(os.path.join(run_root, 'wave5_manifest.json'), manifest)

            ckpt = os.path.join(broot, 'checkpoints', f'epoch_{E}.pt')
            for arm, override in arms.items():
                try:
                    lp = run_nescal._ncl_params(cfg, 1.0 if scale_match is not None else lam)
                    lp['secondary_params'].update(override)
                    sm = {} if scale_match is None else {'lambda_from_js_target': scale_match}
                    aroot = run_dir(base, ds, arch, ts, f'w5_{tag}_ncl_{arm}', seed)
                    _, _, div, arp = train_one(tr,
                        label=f'w5 {tag} ncl-{arm} seed={seed}' + ('' if scale_match is None else f' target={scale_match}'),
                        epochs_to_train=N - E,
                        run_kwargs=run_nescal._base_kwargs(cfg, loaders, seed, aroot, loss_params=lp,
                            resume_from=ckpt, activation_params={'strategy': 'fixed', 'fixed_epoch': E},
                            lambda_kind='cosine', lambda_decay='cosine', **sm))
                    manifest['reportable'].setdefault(f'w5_{tag}_ncl_{arm}', {})[seed] = arp
                    if div: manifest['diverged'].append({'run': f'w5_{tag}_ncl_{arm}', 'seed': seed})
                except Exception as e:
                    print(f'  !! {tag} ncl-{arm} seed={seed} ERRORED ({type(e).__name__}: {e}) — skipping, continuing.',
                          flush=True)
                    manifest['errored'].append({'run': f'w5_{tag}_ncl_{arm}', 'seed': seed,
                                                'error': f'{type(e).__name__}: {e}'})
                write_manifest(os.path.join(run_root, 'wave5_manifest.json'), manifest)

    for rho in w5['longtail_factors']:
        _block(f'lt{rho}', {'type': 'longtail', 'imbalance_factor': rho}, w5['longtail_arms'],
               scale_match=w5['penalty_target'])
    for r in w5['noise_rates']:
        _block(f'noise{int(r * 100)}', {'type': 'noise', 'rate': r, 'kind': w5['noise_kind']}, w5['noise_arms'])

