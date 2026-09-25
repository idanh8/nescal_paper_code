import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types
from reproduction.recipes import read_recipe, bind_data


def composition_main(argv=None):
    p = argparse.ArgumentParser(description='Dual epoch selection and composition at existing paper settings.')
    p.add_argument('--run-root', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--stages', nargs='+', choices=('dual', 'composition'), default=['composition'])
    p.add_argument('--tag', default='composition')
    p.add_argument('--settings', help='JSON overrides: selection_seeds, dual_seeds, composition_onset, lambda_multipliers, composition_seeds')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    root = str(Path(a.run_root).resolve())
    cfg = read_recipe(a.config)['config']
    settings = json.loads(Path(a.settings).read_text()) if a.settings else {}
    names = {'selection_seeds':'SEL_SEEDS',
             'dual_seeds':'DUAL_SEEDS','composition_onset':'COMP_ONSET',
             'lambda_multipliers':'COMP_LAM_MULT','composition_seeds':'COMP_SEEDS'}
    if set(settings)-set(names):
        p.error('Unknown setting: '+', '.join(sorted(set(settings)-set(names))))
    ds = cfg['dataset']
    if ds not in ('cifar10','cifar100'):
        p.error('This protocol supports CIFAR-10 and CIFAR-100.')
    if a.dry_run:
        print(json.dumps({'stages':a.stages,'run_root':root,'tag':a.tag,'settings':settings},indent=2))
        return 0
    from experiments import composition as core
    core.ROOTS={ds:root};core.REV_TS=a.tag
    for key,value in settings.items():
        setattr(core,names[key],value)
    contexts,problems=core._preflight(a.stages,[ds])
    if problems:
        p.error('\n'.join(problems))
    core.TRACKER=core.RunTracker(10**9,label=a.tag)
    fns={'dual':core.stage_dual,'composition':core.stage_composition}
    for stage in ('dual','composition'):
        if stage in a.stages:
            fns[stage](contexts[ds]);core._free()
    return 0


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


def analysis_main(argv=None):
    from experiments.selection_analysis import main
    old=sys.argv
    try:
        if argv is not None:sys.argv=['selection-analysis']+list(argv)
        return main()
    finally:
        sys.argv=old
