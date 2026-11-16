import os
from experiments.common import (timestamp, run_dir, RunTracker, train_one, select_by_val_nll, write_manifest)
from data.image_datasets import load_dataset
pass


CONFIG = __import__('reproduction.recipes', fromlist=['source_config']).source_config('image')


def _loaders(cfg, seed):
    loader_fn = cfg.get('loader_fn')
    if loader_fn is not None:
        return loader_fn(seed)
    return load_dataset(cfg['dataset'], val_split=0.1, batch_size=cfg['batch_size'],
                        seed=seed, data_root=cfg['data_root'], num_workers=cfg['num_workers'])

def _xe_params(): return {'primary_loss': 'XE', 'secondary_loss': None}

def _ncl_params(cfg, lam):
    n = cfg['nescal']
    return {'primary_loss': 'XE', 'secondary_loss': 'ncl', 'lambda_reg': lam,
            'secondary_params': {'k': n['k'], 'mask_mode': n['mask_mode'], 'divergence_type': n['divergence_type'],
                                 'weight_type': n['weight_type'], 'tau': n['tau']}}

def _base_kwargs(cfg, loaders, seed, root, **extra):
    tr, va, te, emb, names = loaders
    kw = dict(train_loader=tr, val_loader=va, test_loader=te, embedding_loader=emb, label_list=names,
              model_params=cfg['model_params'], seed=seed, num_epochs=cfg['num_epochs'], num_bins=cfg['num_bins'],
              lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'],
              diag_every_n=cfg['diag_every_n'], root_path=root, verbose=False, activation_params=None,
              optimizer=cfg.get('optimizer', 'sgd'),
              lr_schedule=cfg.get('lr_schedule', 'multistep'),
              candidate_layers=cfg.get('candidate_layers', ('final', 'stage_minus_1', 'stage_minus_2')),
              embedding_layer=cfg.get('embedding_layer', 'final'), lr_milestones_frac=cfg.get('lr_milestones_frac', (0.4, 0.6)),
              lr_gamma=cfg.get('lr_gamma', 0.1),)
    if 'early_stopping' in cfg:
        kw['early_stopping'] = cfg['early_stopping']

    kw.update(extra)
    return kw

def _ckpt(base_out, cfg, ts, seed, epoch):
    return os.path.join(run_dir(base_out, cfg['dataset'], cfg['arch_name'], ts, 'XE_baseline', seed),
                        'checkpoints', f'epoch_{epoch}.pt')

def _rerun_final_xe(cfg, ts):
    if not cfg.get('early_stopping', False):
        return

    import shutil
    import tempfile

    seeds = list(dict.fromkeys(cfg['stage_b_seeds'] + cfg['confirmatory_seeds']))
    tracker = RunTracker(len(seeds) * cfg['num_epochs'], label='XE early stopping')

    for seed in seeds:
        root = run_dir(cfg['base_out'], cfg['dataset'], cfg['arch_name'], ts, 'XE_baseline', seed)
        loaders = _loaders(cfg, seed)

        with tempfile.TemporaryDirectory(prefix='.xe_rerun_', dir=os.path.dirname(root)) as temp:
            previous = os.path.join(temp, 'previous')
            os.replace(root, previous)

            try:
                os.makedirs(root)
                _, _, diverged, _ = train_one(
                    tracker,
                    label=f'XE_baseline early stopping seed={seed}',
                    epochs_to_train=cfg['num_epochs'],
                    run_kwargs=_base_kwargs(cfg, loaders, seed, root, loss_params=_xe_params(), early_stopping=True, resume_from=None, checkpoint_epochs=[], log_graph_maturity=False))
                if diverged:
                    raise RuntimeError(f'Final XE rerun diverged for seed={seed}')


                old_cp = os.path.join(previous, 'checkpoints')
                new_cp = os.path.join(root, 'checkpoints')
                if os.path.isdir(old_cp):
                    if os.path.isdir(new_cp):
                        shutil.rmtree(new_cp)
                    os.replace(old_cp, new_cp)

            except BaseException:

                if os.path.isdir(root):
                    shutil.rmtree(root)
                os.replace(previous, root)
                raise

def run(cfg=CONFIG, ts=None, schedules=None):
    ts = ts or timestamp()
    base, ds, arch = cfg['base_out'], cfg['dataset'], cfg['arch_name']
    onsets, sb_seeds, cf_seeds = cfg['candidate_onsets'], cfg['stage_b_seeds'], cfg['confirmatory_seeds']
    all_seeds = list(dict.fromkeys(sb_seeds + cf_seeds))
    N = cfg['num_epochs']
    baseline_kwargs = ( {'early_stopping': False} if cfg.get('early_stopping', False) else {})


    med = sorted(onsets)[len(onsets) // 2]
    planned = (len(all_seeds) * N
               + len(sb_seeds) * sum(N - e for e in onsets)
               + len(sb_seeds) * (len(cfg['lambda_grid_cosine']) + len(cfg['lambda_grid_constant'])) * (N - med)
               + len(cf_seeds) * 2 * (N - med))
    tr = RunTracker(planned, label=f'{ds}/{arch}')

    cache = {}
    manifest = {'timestamp': ts, 'config': cfg, 'runs': {}, 'diverged': [], 'reportable': {}}

    def do_nescal(seed, loaders, onset, lam, schedule, method):
        key = (seed, onset, lam, schedule)
        if key in cache:
            val_nll, rp = cache[key]
        else:
            root = run_dir(base, ds, arch, ts, method, seed)
            extra = dict(loss_params=_ncl_params(cfg, lam), resume_from=_ckpt(base, cfg, ts, seed, onset),
                         activation_params={'strategy': 'fixed', 'fixed_epoch': onset},
                         lambda_kind=schedule, lambda_decay='cosine')
            val, test, div, rp = train_one(tr, label=f'{method} seed={seed} onset={onset} λ={lam} {schedule}',
                                           epochs_to_train=N - onset,
                                           run_kwargs=_base_kwargs(cfg, loaders, seed, root, **extra))
            val_nll = None if div else val['nll']
            cache[key] = (val_nll, rp)
            if div: manifest['diverged'].append({'method': method, 'seed': seed, 'onset': onset, 'lam': lam})
        manifest['runs'].setdefault(method, {})[seed] = rp
        return val_nll


    loaders_by_seed = {}
    for seed in sb_seeds:
        loaders_by_seed[seed] = _loaders(cfg, seed)
        root = run_dir(base, ds, arch, ts, 'XE_baseline', seed)
        dense = sorted(set(onsets) | set(range(0, N, cfg['baseline_checkpoint_every'])))
        val, test, div, rp = train_one(tr, label=f'XE_baseline seed={seed}', epochs_to_train=N,
            run_kwargs=_base_kwargs(cfg, loaders_by_seed[seed], seed, root, loss_params=_xe_params(),
                                    log_graph_maturity=True, checkpoint_epochs=dense, **baseline_kwargs))
        manifest['runs'].setdefault('XE_baseline', {})[seed] = rp
        manifest['reportable'].setdefault('XE_baseline', {})[seed] = rp


    if schedules is not None:
        best_onset = onsets[0]
        lam_cos = cfg['lambda_grid_cosine'][0] if 'cosine' in schedules else None
        lam_const = cfg['lambda_grid_constant'][0] if 'constant' in schedules else None
        cos_table, const_table = {}, {}
        onset_table = {'mode': 'explicit'}
        for schedule, lam in (('cosine', lam_cos), ('constant', lam_const)):
            if schedule not in schedules:
                continue
            scores = {}
            method = (f'ncl_J_cos_onset{best_onset}_lam{lam}' if schedule == 'cosine'
                      else f'ncl_B_const_lam{lam}')
            for seed in sb_seeds:
                scores.setdefault((best_onset, lam), {})[seed] = do_nescal(
                    seed, loaders_by_seed[seed], best_onset, lam, schedule, method=method)
            _, table = select_by_val_nll(scores)
            if schedule == 'cosine':
                cos_table = table
            else:
                const_table = table
    elif cfg.get('selection_mode', 'sequential') == 'joint':
        cos_scores = {}
        for seed in sb_seeds:
            for e in onsets:
                for lam in cfg['lambda_grid_cosine']:
                    cos_scores.setdefault((e, lam), {})[seed] = do_nescal(
                        seed, loaders_by_seed[seed], e, lam, 'cosine', method=f'ncl_J_cos_onset{e}_lam{lam}')
        best_cos, cos_table = select_by_val_nll(cos_scores)
        best_onset, lam_cos = best_cos
        print(f'  -> joint winner: onset E*={best_onset}, λ_cosine={lam_cos}', flush=True)
        if best_onset == max(onsets) or lam_cos == max(cfg['lambda_grid_cosine']):
            print(f'  !! WARNING: joint winner is on a GRID EDGE (onset max={max(onsets)}, '
                  f'λ max={max(cfg["lambda_grid_cosine"])}) - widen the grid before trusting this.', flush=True)
        const_scores = {}
        for seed in sb_seeds:
            for lam in cfg['lambda_grid_constant']:
                const_scores.setdefault(('constant', lam), {})[seed] = do_nescal(
                    seed, loaders_by_seed[seed], best_onset, lam, 'constant', method=f'ncl_B_const_lam{lam}')
        best_const, const_table = select_by_val_nll(const_scores)
        lam_const = best_const[1]
        onset_table = {'mode': 'joint'}
    else:
        onset_scores = {e: {} for e in onsets}
        for seed in sb_seeds:
            for e in onsets:
                onset_scores[e][seed] = do_nescal(seed, loaders_by_seed[seed], e, cfg['lambda_ref'], 'cosine',
                                                  method=f'ncl_onset{e}_ref')
        best_onset, onset_table = select_by_val_nll(onset_scores)
        print(f'  -> selected onset E* = {best_onset}  (mean val NLL: '
              f'{ {e: round(v, 4) for e, v in onset_table.items()} })', flush=True)
        if best_onset == max(onsets):
            print(f'  !! WARNING: selected onset is the GRID MAX ({max(onsets)}) - widen candidate_onsets.', flush=True)
        cos_scores, const_scores = {}, {}
        for seed in sb_seeds:
            for lam in cfg['lambda_grid_cosine']:
                cos_scores.setdefault(('cosine', lam), {})[seed] = do_nescal(
                    seed, loaders_by_seed[seed], best_onset, lam, 'cosine', method=f'ncl_B_cos_lam{lam}')
            for lam in cfg['lambda_grid_constant']:
                const_scores.setdefault(('constant', lam), {})[seed] = do_nescal(
                    seed, loaders_by_seed[seed], best_onset, lam, 'constant', method=f'ncl_B_const_lam{lam}')
        best_cos, cos_table = select_by_val_nll(cos_scores)
        best_const, const_table = select_by_val_nll(const_scores)
        lam_cos, lam_const = best_cos[1], best_const[1]

    print(f'  -> FINAL: E*={best_onset}, λ_cosine={lam_cos}, λ_constant={lam_const}', flush=True)
    manifest.update({'best_onset': best_onset, 'lambda_cosine': lam_cos, 'lambda_constant': lam_const,
                     'selection_mode': cfg.get('selection_mode', 'sequential'),
                     'onset_table': onset_table,
                     'cosine_table': {str(k): v for k, v in cos_table.items()},
                     'constant_table': {str(k): v for k, v in const_table.items()}})

    for seed in cf_seeds:
        loaders_by_seed[seed] = _loaders(cfg, seed)
        root = run_dir(base, ds, arch, ts, 'XE_baseline', seed)
        checkpoint_epochs=sorted(set([best_onset]) | set(range(0, N, cfg['baseline_checkpoint_every'])))
        _, _, _, rp = train_one(tr, label=f'XE_baseline seed={seed} (confirmatory)', epochs_to_train=N,
            run_kwargs=_base_kwargs(cfg, loaders_by_seed[seed], seed, root, loss_params=_xe_params(),
                                    log_graph_maturity=True, checkpoint_epochs=checkpoint_epochs, **baseline_kwargs))
        manifest['runs'].setdefault('XE_baseline', {})[seed] = rp
        manifest['reportable'].setdefault('XE_baseline', {})[seed] = rp

    for seed in all_seeds:
        if schedules is None or 'cosine' in schedules:
            do_nescal(seed, loaders_by_seed[seed], best_onset, lam_cos, 'cosine', method='ncl_FINAL_cosine')
            manifest['reportable'].setdefault('ncl_FINAL_cosine', {})[seed] = manifest['runs']['ncl_FINAL_cosine'][seed]
        if schedules is None or 'constant' in schedules:
            do_nescal(seed, loaders_by_seed[seed], best_onset, lam_const, 'constant', method='ncl_FINAL_constant')
            manifest['reportable'].setdefault('ncl_FINAL_constant', {})[seed] = manifest['runs']['ncl_FINAL_constant'][seed]

    write_manifest(os.path.join(base, ds, arch, ts, 'run_manifest.json'), manifest)
    print(f'\n=== NeSCal suite {ds}/{arch} complete. E*={best_onset}, λ_cos={lam_cos}, λ_const={lam_const}. '
          f'{len(manifest["diverged"])} diverged configs. ===', flush=True)
    pass
    _rerun_final_xe(cfg, ts)
