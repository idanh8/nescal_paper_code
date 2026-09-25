import os, sys, copy, argparse, contextlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sklearn.metrics
import numpy as np
import torch
import losses.nescal as nescal_mod
import experiments.nescal as rn
from experiments.common import run_dir, RunTracker, train_one, write_manifest, load_json, load_pkl

LABEL = 'abl_randomgraph_matched'
MANIFEST_NAME = 'random_graph_ablation_manifest_matched.json'


class RandomNeighborLoss(nescal_mod.PreNeighborhoodConsistencyLoss):
    random_seed = 0

    def precompute_matrices(self, features, chunk_size=512):
        N = features.shape[0]
        rng = np.random.default_rng(self.random_seed)
        idx = rng.integers(0, N - 1, size=(N, self.k))
        rows = np.arange(N)[:, None]
        idx = np.where(idx >= rows, idx + 1, idx)
        self.neighbors = torch.from_numpy(idx.astype(np.int64))
        self.neighbor_weights = torch.ones(N, self.k, dtype=torch.float32)


@contextlib.contextmanager
def _random_graph_patched(seed):
    RandomNeighborLoss.random_seed = seed
    orig = nescal_mod.PreNeighborhoodConsistencyLoss
    nescal_mod.PreNeighborhoodConsistencyLoss = RandomNeighborLoss
    try:
        yield
    finally:
        nescal_mod.PreNeighborhoodConsistencyLoss = orig


def _paradigm(cfg):
    if 'blob' in cfg: return 'synthetic'
    if cfg.get('arch_name') == 'frozen_mlp': return 'frozen'
    if cfg.get('arch_name') == 'gpcnn': return 'gpcnn'
    if cfg.get('arch_name') == 'tabular_mlp': return 'tabular'
    return 'image'


def _loader_fn(cfg, paradigm, root):
    if paradigm == 'tabular':
        from data.tabular_datasets import build_tabular
        return build_tabular(cfg['dataset'], batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])[0]
    if paradigm == 'gpcnn':
        from data.gpcnn_datasets import build_20ng_gpcnn
        return build_20ng_gpcnn(cache_dir='./data/glove', max_len=1000, batch_size=cfg['batch_size'],
                                num_workers=cfg['num_workers'])[0]
    if paradigm == 'frozen':
        from data.frozen_datasets import load_frozen
        raw, task, enc = cfg['_dataset_raw'], cfg.get('_task'), cfg['_encoder']
        tag = f'{raw}_{task}' if (raw == 'amazon' and task) else raw
        return lambda s: load_frozen(tag, enc, val_split=cfg['val_split'], batch_size=cfg['batch_size'], seed=s,
                                     num_workers=cfg['num_workers'], base_dir=cfg['embedding_dir'],
                                     split_seed=cfg['split_seed'])
    if paradigm == 'synthetic':
        from data.synthetic_datasets import load_synthetic
        sidecar = os.path.join(root, 'random_graph_sidecar')
        os.makedirs(sidecar, exist_ok=True)
        return lambda s: load_synthetic(val_split=0.2, batch_size=cfg['batch_size'], seed=s,
                                        data_root=cfg['data_root'], num_workers=cfg['num_workers'],
                                        sidecar_dir=sidecar, plot_path=None, **dict(cfg['blob']))
    return None


def _ckpt(root, seed, epoch):
    return os.path.join(root, 'XE_baseline', f'seed_{seed}', 'checkpoints', f'epoch_{epoch}.pt')

def _headline_settings(mf, par):
    if par == 'synthetic':
        return mf['lambda_final'], 'constant', 'ncl_FINAL'
    return mf['lambda_cosine'], 'cosine', 'ncl_FINAL_cosine'

def _preflight(root):
    mf = load_json(os.path.join(root, 'run_manifest.json'))
    cfg = copy.deepcopy(mf['config'])
    par = _paradigm(cfg)

    E = mf.get('best_onset')
    seeds = cfg.get('ablation_seeds', [18, 42])
    problems = []


    root_dataset = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.abspath(root)))
    )
    cfg_dataset = str(cfg.get('dataset', '')).lower()

    if root_dataset.lower() in {'cifar10', 'cifar100'}:
        if cfg_dataset != root_dataset.lower():
            problems.append(
                f'{root}: root implies dataset={root_dataset!r}, '
                f'but run_manifest.json config has '
                f'dataset={cfg.get("dataset")!r}. '
                'Check the source manifest and checkpoints before running.'
            )

    valid_onset = (
        isinstance(E, int)
        and not isinstance(E, bool)
        and 0 <= E < cfg['num_epochs']
    )
    if not valid_onset:
        problems.append(
            f'{root}: invalid best_onset={E!r}; '
            f'expected an integer in [0, {cfg["num_epochs"]}).'
        )

    lambda_key = 'lambda_final' if par == 'synthetic' else 'lambda_cosine'
    if mf.get(lambda_key) is None:
        problems.append(f'{root}: missing {lambda_key} in run_manifest.json.')


    if par != 'synthetic' and valid_onset:
        problems.extend(
            f'{root}: missing {_ckpt(root, seed, E)}'
            for seed in seeds
            if not os.path.exists(_ckpt(root, seed, E))
        )

    return mf, cfg, par, seeds, E, problems

def run_setting(root, mf, cfg, par, seeds, E, matched=False):

    mname = MANIFEST_NAME.replace('.json', '_matched.json') if matched else MANIFEST_NAME
    label = LABEL + ('_matched' if matched else '')
    cfg = copy.deepcopy(cfg)
    lam, lambda_kind, ref_label = _headline_settings(mf, par)

    lf = _loader_fn(cfg, par, root)
    if lf is not None:
        cfg['loader_fn'] = lf

    from_scratch = par == 'synthetic'
    epochs_to_train = (
        cfg['num_epochs']
        if from_scratch
        else cfg['num_epochs'] - E
    )

    manifest = {
        'run_root': root,
        'best_onset': E,
        'lambda': lam,
        'lambda_kind': lambda_kind,
        'reference_label': ref_label,
        'initialization': 'scratch' if from_scratch else 'XE_checkpoint',
        'reportable': {},
        'diverged': [],
    }

    tr = RunTracker(
        len(seeds) * epochs_to_train,
        label=f'{cfg["dataset"]}:random_graph',
    )

    print(
        f'\n########## {cfg["dataset"]}/{cfg["arch_name"]} '
        f'| onset={E} lambda={lam} lambda_kind={lambda_kind} '
        f'seeds={seeds} '
        f'initialization={manifest["initialization"]} ##########',
        flush=True,
    )

    for seed in seeds:
        loaders = rn._loaders(cfg, seed)
        lp = rn._ncl_params(cfg, lam)


        root_out = os.path.join(root, label, f'seed_{seed}')
        os.makedirs(root_out, exist_ok=True)

        extra = dict(
            loss_params=lp,
            activation_params={
                'strategy': 'fixed',
                'fixed_epoch': E,
            },
            lambda_kind=lambda_kind,
            lambda_decay=lambda_kind,
        )

        if matched:
            head = _resolve(root, mf['reportable']['ncl_FINAL_cosine'][str(seed)])
            ejs = load_pkl(os.path.join(head, 'graph_data.pkl'))['mean_edge_js_at_activation']
            extra['lambda_from_penalty_target'] = mf['lambda_cosine'] * ejs

        if not from_scratch:
            extra['resume_from'] = _ckpt(root, seed, E)

        run_kwargs = rn._base_kwargs(
            cfg, loaders, seed, root_out, **extra
        )


        if from_scratch:
            run_kwargs.pop('resume_from', None)

        with _random_graph_patched(seed):
            _, _, div, rp = train_one(
                tr,
                label=f'{label} seed={seed}',
                epochs_to_train=epochs_to_train,
                run_kwargs=run_kwargs,
            )

        manifest['reportable'].setdefault(label, {})[seed] = rp

        if div:
            manifest['diverged'].append({
                'arm': label,
                'seed': seed,
            })

        write_manifest(
            os.path.join(root, mname),
            manifest,
        )


def _ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return f'{np.mean(v):.4f} ± {(np.std(v, ddof=1) if len(v) > 1 else 0):.4f}' if v else '--'


def _summarize(root, mf, cfg):
    if _paradigm(cfg) == 'synthetic':
        ref_label = 'ncl_FINAL'
    else:
        ref_label = (
            'ncl_FINAL_cosine'
            if 'ncl_FINAL_cosine' in mf.get('reportable', {})
            else 'ncl_FINAL_constant'
        )
    ref_dirs = [p for p in mf['reportable'][ref_label].values() if p]
    ab = load_json(os.path.join(root, MANIFEST_NAME)) if os.path.exists(os.path.join(root, MANIFEST_NAME)) else {}
    rand_dirs = list(ab.get('reportable', {}).get(LABEL, {}).values())
    print(f'\n### {cfg["dataset"]}/{cfg["arch_name"]}: kNN graph vs. random graph\n')
    print('| arm | NLL | Brier | ECE | Acc |\n|---|---|---|---|---|')
    for label, dirs in [('kNN graph (headline)', ref_dirs), ('random graph (control)', rand_dirs)]:
        rows = [load_pkl(os.path.join(d, 'test_results.pkl')) for d in dirs
                if d and os.path.exists(os.path.join(d, 'test_results.pkl'))]
        print(f'| {label} | {_ms([r.get("nll") for r in rows])} | {_ms([r.get("brier_score") for r in rows])} | '
              f'{_ms([r.get("ece") for r in rows])} | {_ms([r.get("accuracy") for r in rows])} | (n={len(rows)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry_run', action='store_true')
    mode.add_argument('--summary_only', action='store_true')
    ap.add_argument('--scale_matched', action='store_true')
    a = ap.parse_args()

    plans = []

    for supplied_root in a.root:
        root = os.path.abspath(os.path.expanduser(supplied_root))

        if a.summary_only:
            mf = load_json(os.path.join(root, 'run_manifest.json'))
            _summarize(root, mf, mf['config'])
            continue

        mf, cfg, par, seeds, E, problems = _preflight(root)

        print(
            f'[{par}] {root}: '
            f'config dataset={cfg.get("dataset")!r} '
            f'arch={cfg.get("arch_name")!r} '
            f'onset={E} '
            f'lambda={mf.get("lambda_final" if par == "synthetic" else "lambda_cosine")} '
            f'seeds={seeds}',
            flush=True,
        )

        if problems:
            print(
                '  PROBLEMS (skipping this root):\n   '
                + '\n   '.join(problems),
                flush=True,
            )
            continue

        plans.append((root, mf, cfg, par, seeds, E))

    if a.dry_run or a.summary_only:
        return 0

    for root, mf, cfg, par, seeds, E in plans:
        run_setting(root, mf, cfg, par, seeds, E, matched=a.scale_matched)
        _summarize(root, mf, cfg)

    return 0


def _resolve(root, p):
    if p and os.path.isdir(p):
        return p
    p = (p or '').rstrip('/')
    alt = os.path.join(root, os.path.basename(os.path.dirname(p)), os.path.basename(p))
    return alt if os.path.isdir(alt) else p
