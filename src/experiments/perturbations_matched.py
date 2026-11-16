import os
import numpy as np
from experiments.common import run_dir, RunTracker, train_one, write_manifest, load_json, load_pkl
from experiments.reporting import _fmt, _table, _write_csv
import experiments.nescal as run_nescal
from robustness.perturbations import build_perturbed_cifar_loaders

DATASET = None
ARCH = None
RUN_ROOT = None
LT_LAMBDA = 1.0
LT_ARMS = {'js': 'js', 'fisher': 'fisher', 'uniform': 'l2'}
LT_FACTORS = [10, 50, 100]
NOISE_ARMS = {'js': 'js'}
NOISE_RATES = [0.2, 0.4]
SEEDS = [18, 42]
EPS_FISHER = 1e-2
SKIP_DONE = True

MPATH_NAME = 'wave5_fisherfix_manifest.json'


def run():
    cfg = run_nescal.CONFIG
    assert cfg['dataset'] == DATASET and cfg['arch_name'] == ARCH, \
        f"CONFIG is {cfg['dataset']}/{cfg['arch_name']} but script set to {DATASET}/{ARCH} -- align them."
    assert DATASET in ('cifar10', 'cifar100'), 'Wave 5 is CIFAR-only.'
    base = cfg['base_out']
    mf = load_json(os.path.join(RUN_ROOT, 'run_manifest.json'))
    E, lam, N = mf['best_onset'], mf['lambda_cosine'], cfg['num_epochs']
    ts = RUN_ROOT.rstrip('/').split(os.sep)[-1]
    mpath = os.path.join(RUN_ROOT, MPATH_NAME)

    manifest = load_json(mpath) if (SKIP_DONE and os.path.exists(mpath)) else {}
    manifest.setdefault('run_root', RUN_ROOT)
    manifest.setdefault('best_onset', E)
    manifest.setdefault('longtail_lambda', LT_LAMBDA)
    for key in ('reportable', 'diverged', 'errored'):
        manifest.setdefault(key, {} if key == 'reportable' else [])

    def _done(name, seed):
        return SKIP_DONE and seed in manifest['reportable'].get(name, {})

    planned = ((len(LT_FACTORS) * (N + len(LT_ARMS) * (N - E))
                + len(NOISE_RATES) * (N + len(NOISE_ARMS) * (N - E))) * len(SEEDS))
    tr = RunTracker(planned, label=f'{DATASET}/{ARCH}:wave5-full')

    def _block(tag, pert, arms, arm_lam):
        for seed in SEEDS:
            loaders = build_perturbed_cifar_loaders(DATASET, pert, seed, batch_size=cfg['batch_size'],
                                                    data_root=cfg['data_root'], num_workers=cfg['num_workers'])
            broot = run_dir(base, DATASET, ARCH, ts, f'w5_{tag}_XE', seed)
            ckpt = os.path.join(broot, 'checkpoints', f'epoch_{E}.pt')
            if not os.path.exists(ckpt):
                print(f'[w5] {tag} seed={seed}: no XE ckpt -- training perturbed XE base.', flush=True)
                try:
                    train_one(tr, label=f'w5 {tag} XE seed={seed}', epochs_to_train=N,
                        run_kwargs=run_nescal._base_kwargs(cfg, loaders, seed, broot,
                            loss_params={'primary_loss': 'XE', 'secondary_loss': None},
                            activation_params=None, log_graph_maturity=False, checkpoint_epochs=[E]))
                except Exception as e:
                    print(f'  !! {tag} XE seed={seed} ERRORED ({type(e).__name__}: {e}) -- skipping arms.',
                          flush=True)
                    manifest['errored'].append({'run': f'w5_{tag}_XE', 'seed': seed,
                                                'error': f'{type(e).__name__}: {e}'})
                    write_manifest(mpath, manifest)
                    continue
            else:
                print(f'[w5] {tag} seed={seed}: resuming from existing XE ckpt.', flush=True)
            manifest['reportable'].setdefault(f'w5_{tag}_XE', {})[seed] = broot
            write_manifest(mpath, manifest)

            for arm, div in arms.items():
                name = f'w5_{tag}_ncl_{arm}'
                if _done(name, seed):
                    print(f'[w5] {name} seed={seed}: already done -- skip.', flush=True)
                    continue
                try:
                    lp = run_nescal._ncl_params(cfg, arm_lam)
                    lp['secondary_params'].update({'divergence_type': div, 'eps_fisher': EPS_FISHER})
                    aroot = run_dir(base, DATASET, ARCH, ts, name, seed)
                    _, _, dvg, arp = train_one(tr,
                        label=f'w5 {tag} ncl-{arm} seed={seed} λ={arm_lam}',
                        epochs_to_train=N - E,
                        run_kwargs=run_nescal._base_kwargs(cfg, loaders, seed, aroot, loss_params=lp,
                            resume_from=ckpt, activation_params={'strategy': 'fixed', 'fixed_epoch': E},
                            lambda_kind='cosine', lambda_decay='cosine'))
                    manifest['reportable'].setdefault(name, {})[seed] = arp
                    if dvg:
                        manifest['diverged'].append({'run': name, 'seed': seed})
                        print(f'  !! {name} seed={seed} DIVERGED', flush=True)
                    else:
                        print(f'[w5] {name} seed={seed} done.', flush=True)
                except Exception as e:
                    print(f'  !! {name} seed={seed} ERRORED ({type(e).__name__}: {e}).', flush=True)
                    manifest['errored'].append({'run': name, 'seed': seed, 'error': f'{type(e).__name__}: {e}'})
                write_manifest(mpath, manifest)

    for rho in LT_FACTORS:
        _block(f'lt{rho}', {'type': 'longtail', 'imbalance_factor': rho}, LT_ARMS, LT_LAMBDA)
    for r in NOISE_RATES:
        _block(f'noise{int(r * 100)}', {'type': 'noise', 'rate': r, 'kind': 'symmetric'},
               NOISE_ARMS, lam)


def build_report():
    cfg = run_nescal.CONFIG
    mpath = os.path.join(RUN_ROOT, MPATH_NAME)
    if not os.path.exists(mpath):
        print(f'[report] no manifest at {mpath}', flush=True)
        return
    rep = load_json(mpath).get('reportable', {})
    if not rep:
        print('[report] manifest has no reportable arms yet', flush=True)
        return
    csv_dir = os.path.join(RUN_ROOT, 'report_csvs')
    os.makedirs(csv_dir, exist_ok=True)
    print(f'\n{"="*70}\nWAVE 5 REPORT (fisher-fix): {"/".join(RUN_ROOT.rstrip("/").split(os.sep)[-3:])}'
          f'\n{"="*70}', flush=True)

    def w5c(label, mk):
        return [load_pkl(os.path.join(rp, 'test_results.pkl')).get(mk)
                for rp in rep[label].values() if os.path.exists(os.path.join(rp, 'test_results.pkl'))]
    rows = [[l] + [_fmt([v for v in w5c(l, mk) if v is not None])
                   for mk in ('nll', 'ece', 'avg_cw_ece', 'accuracy')] for l in sorted(rep)]
    _table('Perturbation robustness (test)',
           rows, ['config', 'nll', 'ece', 'avg_cw_ece', 'accuracy'])
    _write_csv(os.path.join(csv_dir, 'wave5_fisherfix_robustness.csv'),
               ['config', 'nll', 'ece', 'avg_cw_ece', 'accuracy'], rows)

    C = cfg['model_params']['num_classes']

    def ht(o, key):
        p = np.asarray(o['test_conf_vectors']); y = np.asarray(o['test_labels']); ts_ = int(C * 2 / 3)
        head, tail = y < ts_, y >= ts_
        def _acc(m): return float((p[m].argmax(1) == y[m]).mean()) if m.any() else float('nan')
        def _ece(m):
            if not m.any(): return float('nan')
            conf = p[m].max(1); corr = (p[m].argmax(1) == y[m]).astype(float)
            idx = np.clip(np.digitize(conf, np.linspace(0, 1, 16)) - 1, 0, 14); e = 0.0
            for b in range(15):
                bm = idx == b
                if bm.sum(): e += bm.sum() / len(conf) * abs(conf[bm].mean() - corr[bm].mean())
            return float(e)
        return {'head_acc': _acc(head), 'tail_acc': _acc(tail),
                'head_ece': _ece(head), 'tail_ece': _ece(tail)}[key]

    rows = []
    for l in sorted(x for x in rep if '_lt' in x):
        vals = {k: [] for k in ('head_acc', 'tail_acc', 'head_ece', 'tail_ece')}
        for rp in rep[l].values():
            dp = os.path.join(rp, 'test_data.pkl')
            if os.path.exists(dp):
                o = load_pkl(dp)
                for k in vals:
                    vals[k].append(ht(o, k))
        rows.append([l] + [_fmt([v for v in vals[k] if v == v])
                           for k in ('head_acc', 'tail_acc', 'head_ece', 'tail_ece')])
    if rows:
        _table('Fisher test: head vs tail (tail = least-frequent third; compare ncl_fisher vs ncl_uniform on tail_ece)',
               rows, ['config', 'head_acc', 'tail_acc', 'head_ece', 'tail_ece'])
        _write_csv(os.path.join(csv_dir, 'wave5_fisherfix_head_tail.csv'),
                   ['config', 'head_acc', 'tail_acc', 'head_ece', 'tail_ece'], rows)
