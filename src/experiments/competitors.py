import os, argparse, itertools
from experiments.common import run_dir, RunTracker, train_one, write_manifest, select_by_val_nll
from data.image_datasets import load_dataset
from configs.method_specs import method_specs, fl_primary_spec
pass
import experiments.nescal as run_nescal

def _loaders(cfg, seed):


    loader_fn = cfg.get('loader_fn')
    if loader_fn is not None:
        return loader_fn(seed)
    return load_dataset(cfg['dataset'], val_split=0.1, batch_size=cfg['batch_size'], seed=seed,
                        data_root=cfg['data_root'], num_workers=cfg['num_workers'])

def _loss_params(primary, secondary, secondary_keys, cand):
    lp = {'primary_loss': primary, 'secondary_loss': secondary}; sp = {}
    for k, v in cand.items():
        (sp if k in secondary_keys else lp)[k] = v
    if secondary is not None: lp['secondary_params'] = sp
    return lp

def _tag(name, cand):
    if not cand: return name
    body = '_'.join(f'{k}{str(cand[k]).replace(".", "p")}' for k in sorted(cand))
    return f'{name}_{body}'

def _candidate_grid(search_space):
    keys = list(search_space); vals = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]

def run(ts, cfg=None, reuse_baseline_ts=None, specs=None, fl_spec=None):
    cfg = cfg or run_nescal.CONFIG
    specs = specs if specs is not None else method_specs
    fl_spec = fl_spec if fl_spec is not None else fl_primary_spec
    ds, arch, base = cfg['dataset'], cfg['arch_name'], cfg['base_out']
    sb, cf = cfg['stage_b_seeds'], cfg['confirmatory_seeds']
    all_seeds = list(dict.fromkeys(sb + cf)); N = cfg['num_epochs']

    tr = RunTracker(10 ** 9, label=f'{ds}/{arch}:competitors')
    manifest = {'timestamp': ts, 'reportable': {}, 'winners': {}, 'diverged': [], 'errored': []}
    loaders_by_seed = {s: _loaders(cfg, s) for s in all_seeds}
    print(f'=== Competitors {ds}/{arch} @ {ts} | {list(specs) + ["FL"]} | seeds={all_seeds} ===', flush=True)

    cache = {}
    def _run(name, primary, secondary, secondary_keys, cand, seed):
        key = (name, tuple(sorted(cand.items())), seed)
        if key in cache: return cache[key]
        root = run_dir(base, ds, arch, ts, _tag(name, cand), seed)
        lp = _loss_params(primary, secondary, secondary_keys, cand)
        try:
            val, _, div, rp = train_one(tr, label=f'{name} {cand} seed={seed}', epochs_to_train=N,
                run_kwargs=run_nescal._base_kwargs(cfg, loaders_by_seed[seed], seed, root,
                                                   loss_params=lp, activation_params=None, log_graph_maturity=False))
            v = None if div else val['nll']
        except Exception as e:
            print(f'  !! {name} {cand} seed={seed} ERRORED ({type(e).__name__}: {e}) — skipping.', flush=True)
            manifest['errored'].append({'method': name, 'cand': cand, 'seed': seed, 'error': f'{type(e).__name__}: {e}'})
            v, rp = None, root
        cache[key] = (v, rp)
        write_manifest(os.path.join(run_root_ref[0], 'competitors_manifest.json'), manifest)
        return cache[key]

    def _select(name, primary, secondary, secondary_keys, candidates):
        scores = {i: {s: _run(name, primary, secondary, secondary_keys, c, s)[0] for s in sb}
                  for i, c in enumerate(candidates)}
        best_i, table = select_by_val_nll(scores)
        return candidates[best_i], {str(candidates[i]): table[i] for i in table}

    def _select_sequential(name, spec, primary, secondary_keys):
        order, space = spec['search_order'], spec['search_space']
        best = {k: space[k][0] for k in order}; trace = {}
        for param in order:
            per_val = {}
            for v in space[param]:
                cand = dict(best); cand[param] = v
                seed_vals = [_run(name, primary, spec['secondary_loss'], secondary_keys, cand, s)[0] for s in sb]
                finite = [x for x in seed_vals if x is not None]
                per_val[v] = sum(finite) / len(finite) if len(finite) == len(seed_vals) and finite else None
            ok = {v: s for v, s in per_val.items() if s is not None}
            best[param] = min(ok, key=ok.get) if ok else space[param][0]
            trace[param] = per_val
        return best, trace


    if reuse_baseline_ts:
        print(f'  reusing XE baseline from timestamp {reuse_baseline_ts}', flush=True)
    else:
        print('\n### XE baseline (own) ###', flush=True)
        for s in all_seeds:
            root = run_dir(base, ds, arch, ts, 'XE_baseline', s)
            try:
                _, _, _, rp = train_one(tr, label=f'XE_baseline seed={s}', epochs_to_train=N,
                    run_kwargs=run_nescal._base_kwargs(cfg, loaders_by_seed[s], s, root,
                        loss_params={'primary_loss': 'XE', 'secondary_loss': None}, activation_params=None,
                        log_graph_maturity=False))
                manifest['reportable'].setdefault('XE_baseline', {})[s] = rp
            except Exception as e:
                manifest['errored'].append({'method': 'XE_baseline', 'seed': s, 'error': f'{type(e).__name__}: {e}'})
            write_manifest(os.path.join(run_root_ref[0], 'competitors_manifest.json'), manifest)


    for name, spec in specs.items():
        primary = spec['primary_losses'][0]
        secondary = spec['secondary_loss']; sk = spec.get('secondary_param_keys', [])
        st = spec['search_type']
        print(f'\n### {name}: {st} ###', flush=True)
        if st == 'candidate_set':
            winner, table = _select(name, primary, secondary, sk, _candidate_grid(spec['search_space']))
        elif st == 'sequential':
            winner, table = _select_sequential(name, spec, primary, sk)
        elif st == 'fixed_per_dataset':
            key = f'{ds}/{arch}'
            if key in spec['fixed_params_by_dataset']:
                winner, table = spec['fixed_params_by_dataset'][key], f'matched {key} (no search)'
                print(f'  {name}: using published params for {key} -> {winner}', flush=True)
            else:
                winner, table = _select(name, primary, secondary, sk, _candidate_grid(spec['fallback_search_space']))
        else:
            raise ValueError(f'{name}: unknown search_type {st!r}')
        manifest['winners'][name] = {'candidate': winner, 'selection': table}
        for s in all_seeds:
            _, rp = _run(name, primary, secondary, sk, winner, s)
            manifest['reportable'].setdefault(f'{name}_FINAL', {})[s] = rp
        print(f'  -> {name} winner: {winner}', flush=True)


    fl_cands = ([{'gamma': g, 'fl_variant': 'fixed'} for g in fl_spec['fixed_gamma_search']]
                + [{'gamma': g, 'fl_variant': 'adaptive'} for g in fl_spec['adaptive_gamma_search']])
    if fl_cands:
        print('\n### FL: fixed + adaptive gamma ###', flush=True)
        fl_winner, fl_table = _select('FL', 'FL', None, [], fl_cands)
        manifest['winners']['FL'] = {'candidate': fl_winner, 'selection': fl_table}
        for s in all_seeds:
            _, rp = _run('FL', 'FL', None, [], fl_winner, s)
            manifest['reportable'].setdefault('FL_FINAL', {})[s] = rp
        print(f'  -> FL winner: {fl_winner}', flush=True)

    write_manifest(os.path.join(run_root_ref[0], 'competitors_manifest.json'), manifest)
    print(f'\n=== Competitors complete. Winners: '
          f'{ {k: v["candidate"] for k, v in manifest["winners"].items()} } ===', flush=True)
    pass
    return ts

run_root_ref = ['']
