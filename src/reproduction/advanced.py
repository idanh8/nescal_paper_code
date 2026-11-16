import argparse
import importlib
import json
from pathlib import Path
import sys
import types
from reproduction.recipes import read_recipe, bind_data


def theory_main(argv=None):
    p=argparse.ArgumentParser(description='Theory diagnostics using the original real/synthetic analysis cores.')
    p.add_argument('--run-root',required=True)
    p.add_argument('--kind',choices=('real','synthetic'),default='real')
    p.add_argument('--config',required=True,help='Experiment settings for rebuilding loaders')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('options',nargs=argparse.REMAINDER,help='Original analysis options after --, e.g. --skip_pass')
    a=p.parse_args(argv)
    root=str(Path(a.run_root).resolve())
    forwarded=[x for x in a.options if x!='--']
    if a.dry_run:
        print(json.dumps({'kind':a.kind,'run_root':root,'options':forwarded},indent=2));return 0
    name='experiments.theory_'+('real' if a.kind=='real' else 'synthetic')
    module=importlib.import_module(name)
    options=['--run_root',root]+forwarded
    if a.kind=='real' and '--skip_pass' not in forwarded:
        path=a.config
        recipe=read_recipe(path)
        if recipe['data']['kind']!='image':
            bound=types.ModuleType('reproduction.bound_data')
            bound.CONFIG=bind_data(recipe,recipe['config'],root)
            sys.modules[bound.__name__]=bound
            options.extend(['--loader_module',bound.__name__])
    old=sys.argv
    try:
        sys.argv=[name]+options
        return module.main()
    finally:
        sys.argv=old


JOBS = {('cifar100', 'resnet50'): 'c100_r50', ('cifar10', 'resnet50'): 'c10_r50',
        ('cifar100', 'wrn-28-10'): 'c100_wrn', ('cifar10', 'wrn-28-10'): 'c10_wrn',
        ('tinyimagenet', 'resnet50'): 'tin'}


def _job_roots(paths):
    out = {}
    for path in paths:
        root = str(Path(path).expanduser()).rstrip('/')
        parts = Path(root).parts
        if len(parts) < 3:
            raise ValueError(f'{path}: expected a run root ending in dataset/architecture/tag')
        name = JOBS.get((parts[-3], parts[-2]), parts[-3])
        if name in out:
            raise ValueError(f'two run roots for {name}: {out[name]} and {root}')
        out[name] = root
    return out


def ncr_main(argv=None):
    p = argparse.ArgumentParser(description='NCR under the NeSCal protocol on existing CIFAR and Tiny-ImageNet run roots. '
                                'Other options are passed on: --stage {select,finish,all,report} --grid {coarse,full} '
                                '--shard i/n --no_extras --preflight_only --smoke.')
    p.add_argument('--run-root', nargs='+', required=True)
    a, rest = p.parse_known_args(argv)
    from experiments import ncr as core
    core.ROOTS = _job_roots(a.run_root)
    if not any(x == '--grid' or x.startswith('--grid=') for x in rest):
        rest += ['--grid', 'coarse']
    return core.main(['--jobs', *core.ROOTS, *rest])


def additional_main(argv=None):
    p = argparse.ArgumentParser(description='Additional experiments on existing run roots. Other options are passed on: '
                                '--stages {analysis,factorial,divergences,scale_arms,scale_bench,mixup_base,mixup,rankmixup} '
                                '--preflight_only --smoke. Roots other than CIFAR and Tiny-ImageNet are used by the analysis stage only.')
    p.add_argument('--run-root', nargs='+', required=True)
    a, rest = p.parse_known_args(argv)
    from experiments import ncr as rnc, additional as core
    roots = _job_roots(a.run_root)
    rnc.ROOTS = {n: r for n, r in roots.items() if n in JOBS.values()}
    core.ANALYSIS_EXTRA_ROOTS = {n: r for n, r in roots.items() if n not in rnc.ROOTS or n == 'tin'}
    core.TIN_ROOT = roots.get('tin', '')
    if not core.TIN_ROOT:
        core.TIN_SCALE_ARMS = []
    return core.main((['--jobs', *rnc.ROOTS] if rnc.ROOTS else []) + rest)


def backbones_main(argv=None):
    p = argparse.ArgumentParser(description='Frozen pre-trained backbones, ViT-B/16 fine-tuning and the cache benchmark. '
                                'Other options are passed on: --stages {head,vit_ft,cache_bench,head_baselines,head_mem} '
                                '--jobs {imagenet_dinov2,c100_vit_b16} --preflight_only --smoke.')
    p.add_argument('--output', default='./runs')
    p.add_argument('--data-root', default='./data')
    a, rest = p.parse_known_args(argv)
    from experiments import backbones as core
    core.BASE_OUT = a.output
    core.FT['base_out'], core.FT['data_root'] = a.output, a.data_root
    return core.main(rest)


def additional_tables_main(argv=None):
    p = argparse.ArgumentParser(description='Read-only tables over the additional experiments. Other options are passed on: '
                                '--heads BASE_OUT (default ./runs).')
    p.add_argument('--run-root', nargs='+', required=True)
    a, rest = p.parse_known_args(argv)
    from experiments import additional_tables as core
    return core.main(['--roots', *[f'{n}={r}' for n, r in _job_roots(a.run_root).items()], *rest])
