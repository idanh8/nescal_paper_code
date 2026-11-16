from copy import deepcopy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def source_config(name):
    return json.loads((REPO / 'configs' / 'source' / (name + '.json')).read_text())


def data_kind(cfg):
    if cfg.get('dataset') == 'synthetic':
        return 'synthetic'
    arch = cfg.get('model_params', {}).get('arch', cfg.get('arch_name'))
    return {'gpcnn': 'gpcnn', 'tabular_mlp': 'tabular',
            'frozen_enc_mlp': 'frozen', 'frozen_mlp': 'frozen'}.get(arch, 'image')


def read_recipe(path):
    raw = json.loads(Path(path).read_text())
    if any(key in raw for key in ('reportable', 'runs', 'winners', 'best_onset', 'timestamp')):
        raise ValueError('Supply an experiment configuration, not a run manifest.')
    if 'config' not in raw:
        raw = {'config': raw}
    recipe = deepcopy(raw)
    cfg = recipe['config']
    if cfg.get('embedding_dir') or cfg.get('loader_fn'):
        raise ValueError('Supply data/encoder settings; loaders and features are built by this run.')
    recipe.setdefault('data', {'kind': data_kind(cfg)})
    recipe.setdefault('workflow', 'synthetic' if recipe['data']['kind'] == 'synthetic' else 'standard')
    if recipe.get('selected'):
        raise ValueError('Pass chosen hyperparameters with --onset, --lambda-* and --method-params.')
    recipe['selected'] = {}
    return recipe


def override(recipe, expression):
    key, sep, value = expression.partition('=')
    if not sep:
        raise ValueError('--set expects dotted.path=JSON_VALUE')
    parts = key.split('.')
    obj = recipe
    for part in parts[:-1]:
        if part not in obj or not isinstance(obj[part], dict):
            raise ValueError(f'Unknown object in --set: {key}')
        obj = obj[part]
    obj[parts[-1]] = json.loads(value)


def selected_config(recipe, mode):
    cfg = deepcopy(recipe['config'])
    if mode == 'search':
        return cfg
    selected = recipe['selected']
    if selected.get('best_onset') is None:
        raise ValueError('Explicit NeSCal requires --onset.')
    onset = selected['best_onset']
    if not 0 <= onset < cfg['num_epochs']:
        raise ValueError('--onset must be at least 0 and smaller than num_epochs.')
    if recipe['data']['kind'] == 'synthetic':
        lam = selected.get('lambda_constant')
        if lam is None:
            raise ValueError('Synthetic NeSCal requires --lambda-constant.')
        if selected.get('lambda_cosine') is not None:
            raise ValueError('Synthetic NeSCal uses constant lambda only.')
        cfg['onset_sweep'] = [onset]
        cfg['lambda_grid_constant'] = [lam]
    else:
        schedules = [name for name in ('cosine', 'constant') if selected.get('lambda_' + name) is not None]
        if not schedules:
            raise ValueError('Explicit NeSCal requires --lambda-cosine and/or --lambda-constant.')
        cfg['candidate_onsets'] = [onset]
        for schedule in ('cosine', 'constant'):
            cfg['lambda_grid_' + schedule] = ([selected['lambda_' + schedule]] if schedule in schedules else [])
        cfg['selection_mode'] = 'joint'
        cfg['_explicit_schedules'] = schedules
    return cfg


METHOD_ALIASES = {'xe': 'XE', 'fl': 'FL', 'ls': 'ls_alone', 'ls_alone': 'ls_alone',
                  'mmce': 'mmce', 'mdca': 'mdca', 's-avuc': 's_avuc',
                  's_avuc': 's_avuc', 'mbls': 'mbls'}


def method_name(name):
    try:
        return METHOD_ALIASES[name.lower()]
    except KeyError:
        raise ValueError('Unknown method: ' + name) from None


def competitor_policy(recipe, mode):
    from configs.method_specs import method_specs, fl_primary_spec
    policy = recipe.get('competitors', {})
    specs = deepcopy(policy.get('specs', method_specs))
    fl = deepcopy(policy.get('fl_spec', fl_primary_spec))
    methods = recipe.get('methods', list(specs) + ['FL'])
    unknown = set(methods) - (set(specs) | {'FL', 'XE'})
    if unknown:
        raise ValueError('Unknown methods: ' + ', '.join(sorted(unknown)))
    specs = {name: spec for name, spec in specs.items() if name in methods}
    if 'FL' not in methods:
        fl = {'fixed_gamma_search': [], 'adaptive_gamma_search': []}
    if mode == 'search':
        return specs, fl
    chosen = recipe['selected'].get('competitors', {})
    expected = set(methods) - {'XE'}
    missing = expected - set(chosen)
    if missing:
        raise ValueError('Supply --method-params METHOD=JSON for: ' + ', '.join(sorted(missing)))
    extra = set(chosen) - expected
    if extra:
        raise ValueError('Parameters given for unrequested methods: ' + ', '.join(sorted(extra)))
    for name, spec in specs.items():
        candidate = chosen[name]
        keys = set(spec.get('search_space', spec.get('fallback_search_space', {})))
        if set(candidate) != keys:
            raise ValueError(f'{name} requires exactly these parameters: {sorted(keys)}')
        spec['search_type'] = 'candidate_set'
        spec['search_space'] = {key: [value] for key, value in candidate.items()}
    if 'FL' in methods:
        candidate = chosen['FL']
        if set(candidate) != {'fl_variant', 'gamma'} or candidate['fl_variant'] not in ('fixed', 'adaptive'):
            raise ValueError('FL needs gamma and fl_variant (fixed or adaptive).')
        fl = {'fixed_gamma_search': [], 'adaptive_gamma_search': []}
        fl[candidate['fl_variant'] + '_gamma_search'] = [candidate['gamma']]
    return specs, fl



def bind_data(recipe, cfg, run_root=None, extract=False):
    kind = recipe['data']['kind']
    data = recipe['data']
    cfg = deepcopy(cfg)
    common = {'batch_size': cfg['batch_size'], 'num_workers': cfg['num_workers']}
    if kind == 'image':
        return cfg
    if kind == 'tabular':
        from data.tabular_datasets import build_tabular
        loader, classes, width = build_tabular(cfg['dataset'], **common, **data.get('kwargs', {}))
        cfg['loader_fn'] = loader
        cfg['model_params'].update(num_classes=classes, in_dim=width)
    elif kind == 'gpcnn':
        from data.gpcnn_datasets import build_20ng_gpcnn
        kwargs = {'cache_dir': str(Path(cfg['data_root']) / 'glove'), 'max_len': 1000}
        kwargs.update(data.get('kwargs', {}))
        loader, classes, vocab, matrix = build_20ng_gpcnn(**common, **kwargs)
        cfg['loader_fn'] = loader
        cfg['model_params'].update(num_classes=classes, vocab_size=vocab, embedding_matrix_path=matrix)
    elif kind == 'synthetic':
        from experiments.synthetic import _load
        if run_root is None:
            raise ValueError('Synthetic loaders need a run root for eta sidecars.')
        sidecars = str(Path(run_root) / 'eta_sidecars')
        cfg['loader_fn'] = lambda seed: _load(cfg, seed, cfg['blob']['heterogeneity_strength'], sidecars)
    elif kind == 'frozen':
        from experiments import frozen
        from data.frozen_datasets import load_frozen
        raw = data.get('dataset', cfg.get('_dataset_raw'))
        encoder = data.get('encoder', cfg.get('_encoder'))
        task = data.get('task', cfg.get('_task'))
        if not raw or not encoder:
            raise ValueError('Frozen recipes need data.dataset and data.encoder.')
        if run_root is None:
            raise ValueError('Frozen experiments need an output run directory.')
        from encoders.extract_embeddings import run
        cfg['embedding_dir'] = run(raw, encoder, data_root=cfg['data_root'],
            robustness=bool(cfg.get('do_robustness') or cfg.get('do_ood')) and raw in ('cifar10', 'cifar100'),
            task=task, finetune=cfg['finetune'], split_seed=cfg['split_seed'],
            val_split=cfg['val_split'], ft_cfg=cfg.get('finetune_cfg', {}),
            output_root=str(Path(run_root) / 'features'), num_workers=cfg['num_workers'])
        frozen.CONFIG.clear()
        frozen.CONFIG.update(cfg)
        if not cfg.get('model_params'):
            dataset_name = cfg['dataset']
            cfg = frozen._make_cfg(raw, encoder, cfg['head_hidden_dims'], task=task)
            cfg['dataset'] = dataset_name
        else:
            tag = frozen._ds_tag(raw, task)
            cfg['loader_fn'] = lambda seed: load_frozen(tag, encoder, val_split=cfg['val_split'],
                batch_size=cfg['batch_size'], seed=seed, num_workers=cfg['num_workers'],
                base_dir=cfg['embedding_dir'], split_seed=cfg['split_seed'])
            cfg.update(_dataset_raw=raw, _encoder=encoder, _task=task)
    else:
        raise ValueError(f'Unknown data.kind: {kind}')
    return cfg


def serializable_config(cfg):
    return {key: value for key, value in cfg.items() if key != 'loader_fn'}
