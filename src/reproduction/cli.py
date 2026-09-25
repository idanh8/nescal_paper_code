"""Simple launchers around the experiment cores used in the research repository."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from reproduction.recipes import (read_recipe, selected_config, competitor_policy,
                                  bind_data, serializable_config, override, method_name)

STAGES = ('nescal', 'competitors', 'posthoc', 'corruptions', 'ood', 'report',
          'ablations', 'random-graph', 'heuristics', 'perturbations',
          'perturbations-original', 'perturbations-longtail', 'author-recipes',
          'synthetic', 'synthetic-figures', 'extract', 'frozen-depth', 'frozen-dropout',
          'headline-selection', 'onset-control')


def parser(default_stage=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Experiment settings JSON (configs/examples/*.json)')
    p.add_argument('--run-root', help='Existing dataset/architecture/tag directory for downstream stages')
    p.add_argument('--stages', nargs='+', choices=STAGES,
                   default=[default_stage] if default_stage else None)
    p.add_argument('--mode', choices=('search', 'explicit'), default='search')
    p.add_argument('--methods', nargs='+', help='Competitors to run: XE FL LS MMCE MDCA s-AvUC MbLS')
    p.add_argument('--method-params', action='append', default=[], metavar='METHOD=JSON')
    p.add_argument('--seeds', nargs='+', type=int, help='Use these seeds without a separate confirmatory phase')
    p.add_argument('--scale-matched', action='store_true', help='Match random-graph initial penalty to the trained kNN reference')
    p.add_argument('--tag', help='Output run name; defaults to a timestamp')
    p.add_argument('--output', help='Base output directory; defaults to the recipe')
    p.add_argument('--data-root')
    p.add_argument('--search-seeds', nargs='+', type=int)
    p.add_argument('--confirmatory-seeds', nargs='*', type=int)
    p.add_argument('--onset', type=int)
    p.add_argument('--lambda-cosine', type=float)
    p.add_argument('--lambda-constant', type=float)
    p.add_argument('--set', action='append', default=[], metavar='PATH=JSON')
    p.add_argument('--calibrators', nargs='+', choices=('ts', 'local', 'dc', 'dac', 'lore'))
    p.add_argument('--labels', nargs='+', help='Existing reportable labels to evaluate')
    p.add_argument('--skip-existing', action='store_true', help='Skip completed posthoc/robustness outputs')
    p.add_argument('--dry-run', action='store_true', help='Print resolved settings without imports, downloads or training')
    return p


def prepare(args):
    recipe = read_recipe(args.config)
    for expr in args.set:
        override(recipe, expr)
    if args.methods:
        recipe['methods'] = list(dict.fromkeys(method_name(m) for m in args.methods))
    for expr in args.method_params:
        name, sep, value = expr.partition('=')
        if not sep:
            raise ValueError('--method-params expects METHOD=JSON_OBJECT')
        name = method_name(name)
        params = json.loads(value)
        if not isinstance(params, dict):
            raise ValueError('--method-params expects a JSON object')
        recipe['selected'].setdefault('competitors', {})[name] = params
    if args.mode == 'search' and (args.onset is not None or args.lambda_cosine is not None or
                                  args.lambda_constant is not None or args.method_params):
        raise ValueError('Use --mode explicit with chosen hyperparameters; use --set for sweep grids.')
    cfg = recipe['config']
    if recipe['data']['kind'] == 'frozen':
        d = recipe['data']
        variant = 'finetuned' if cfg['finetune'] else 'pretrained'
        raw = d['dataset'] + ('_' + d['task'] if d.get('task') and d['dataset'] == 'amazon' else '')
        cfg.setdefault('dataset', f'{raw}_{d["encoder"]}_{variant}_split{cfg["split_seed"]}')
        cfg.setdefault('arch_name', 'frozen_mlp')
    if args.seeds is not None:
        if args.search_seeds is not None or args.confirmatory_seeds is not None:
            raise ValueError('Use --seeds or separate --search-seeds/--confirmatory-seeds.')
        cfg['stage_b_seeds'] = args.seeds
        cfg['confirmatory_seeds'] = []
    if args.output is not None:
        cfg['base_out'] = args.output
    if args.data_root is not None:
        cfg['data_root'] = args.data_root
    if args.search_seeds is not None:
        cfg['stage_b_seeds'] = args.search_seeds
    if args.confirmatory_seeds is not None:
        cfg['confirmatory_seeds'] = args.confirmatory_seeds
    for key, value in [('best_onset', args.onset), ('lambda_cosine', args.lambda_cosine),
                       ('lambda_constant', args.lambda_constant)]:
        if value is not None:
            recipe['selected'][key] = value
    if args.lambda_constant is not None and recipe['data']['kind'] == 'synthetic':
        recipe['selected']['lambda_final'] = args.lambda_constant
    stages = args.stages or recipe.get('stages')
    if stages is None:
        stages = (['synthetic', 'synthetic-figures', 'report'] if recipe['workflow'] == 'synthetic'
                  else ['nescal', 'competitors', 'posthoc', 'report'])
    cfg = selected_config(recipe, args.mode) if any(s in stages for s in ('nescal', 'synthetic')) else deepcopy(cfg)
    if any(s in stages for s in ('competitors', 'synthetic')):
        competitor_policy(recipe, args.mode)
    if not cfg.get('stage_b_seeds') and any(s in stages for s in ('nescal', 'competitors', 'synthetic')):
        raise ValueError('The original selection routines require at least one search seed.')
    if args.run_root:
        root = Path(args.run_root).expanduser().resolve()
        if cfg.get('dataset') and cfg.get('arch_name'):
            if root.parent.name != cfg['arch_name'] or root.parent.parent.name != cfg['dataset']:
                raise ValueError('Run root must end in dataset/architecture/tag matching --config.')
            cfg['base_out'] = str(root.parents[2])
        tag = root.name
    else:
        tag = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
        if tag in ('.', '..') or Path(tag).name != tag or '/' in tag or '\\' in tag:
            raise ValueError('--tag must be one directory name.')
        root = Path(cfg['base_out']) / cfg['dataset'] / cfg.get('arch_name', 'frozen_mlp') / tag
    return recipe, cfg, stages, str(root), tag


def _set_core_config(cfg):
    from experiments import nescal
    nescal.CONFIG.clear()
    nescal.CONFIG.update(cfg)


def _perturbation_options(module, recipe, cfg, root):
    module.DATASET, module.ARCH, module.RUN_ROOT = cfg['dataset'], cfg['arch_name'], root
    options = recipe.get('perturbations', {})
    names = {'seeds': 'SEEDS', 'penalty_target': 'PENALTY_TARGET', 'eps_fisher': 'EPS_FISHER',
             'longtail_factors': 'LT_FACTORS', 'noise_rates': 'NOISE_RATES',
             'longtail_arms': 'LT_ARMS', 'noise_arms': 'NOISE_ARMS', 'skip_done': 'SKIP_DONE'}
    if hasattr(module, 'FACTORS'):
        names['longtail_factors'] = 'FACTORS'
        names['longtail_arms'] = 'ARMS'
    for key, name in names.items():
        if key in options:
            setattr(module, name, options[key])


def execute_stage(stage, recipe, cfg, root, tag, args):
    if stage == 'nescal':
        if recipe['data']['kind'] == 'synthetic':
            from experiments import synthetic
            rawcfg = {k: v for k, v in cfg.items() if k != 'loader_fn'}
            run_root, sidecar, _, _ = synthetic.run_headline(rawcfg, tag, include_followups=False)
            synthetic.compute_eta_report(rawcfg, run_root, sidecar)
            return
        from experiments.nescal import run
        if args.mode == 'explicit':
            run(cfg, tag, schedules=cfg['_explicit_schedules'])
        else:
            run(cfg, tag)
    elif stage == 'competitors':
        from experiments import competitors
        specs, fl = competitor_policy(recipe, args.mode)
        competitors.run_root_ref[0] = root
        reuse = tag if Path(root, 'run_manifest.json').exists() else None
        competitors.run(tag, cfg=cfg, reuse_baseline_ts=reuse, specs=specs, fl_spec=fl)
    elif stage == 'posthoc':
        from experiments import posthoc
        if args.calibrators:
            posthoc.POSTHOC = tuple(args.calibrators)
        posthoc.run(root, cfg['dataset'], cfg['arch_name'],
                    batch_size=cfg['batch_size'] if recipe['data']['kind'] == 'frozen' else 256,
                    data_root=cfg['data_root'],
                    num_workers=cfg['num_workers'] if recipe['data']['kind'] == 'frozen' else 4,
                    num_bins=cfg['num_bins'] if recipe['data']['kind'] in ('frozen', 'synthetic') else 15,
                    loader_fn=cfg.get('loader_fn'), only_labels=args.labels, skip_existing=args.skip_existing)
    elif stage in ('corruptions', 'ood'):
        if recipe['data']['kind'] == 'frozen':
            from experiments import frozen
            c = deepcopy(cfg)
            c['do_robustness'], c['do_ood'] = stage == 'corruptions', stage == 'ood'
            frozen.frozen_extras(root, cfg['_dataset_raw'], cfg['_encoder'], c, task=cfg.get('_task'))
        else:
            if cfg['dataset'] not in ('cifar10', 'cifar100', 'tinyimagenet'):
                raise ValueError('Corruption/OOD stages require an image setting.')
            from experiments.robustness import run
            run(root, cfg['dataset'], do_corruption=stage == 'corruptions', do_ood=stage == 'ood',
                data_root=cfg['data_root'], num_bins=cfg['num_bins'],
                only_labels=args.labels, skip_existing=args.skip_existing)
    elif stage == 'report':
        from experiments.reporting import assemble_report
        assemble_report(root)
        if Path(root, 'eta_metrics.json').is_file():
            print(json.dumps(json.loads(Path(root, 'eta_metrics.json').read_text()).get('aggregate', {}), indent=2))
    elif stage == 'ablations':
        from experiments.ablations import run
        run(root, cfg)
    elif stage == 'heuristics':
        from experiments.heuristics import run
        run(root, cfg)
    elif stage == 'random-graph':
        from experiments.random_graph import _preflight, run_setting, _summarize
        mf, oldcfg, par, seeds, onset, problems = _preflight(root)
        if problems:
            raise ValueError('\n'.join(problems))
        run_setting(root, mf, oldcfg, par, seeds, onset, matched=args.scale_matched)
        _summarize(root, mf, oldcfg)
    elif stage.startswith('perturbations'):
        if cfg['dataset'] not in ('cifar10', 'cifar100'):
            raise ValueError('Perturbation stages require CIFAR.')
        if stage == 'perturbations-original':
            from experiments import perturbations
            options = deepcopy(perturbations.WAVE5)
            options.update(recipe.get('perturbations', {}))
            perturbations.run(root, cfg, options)
        else:
            from experiments import perturbations_matched, perturbations_longtail
            module = perturbations_matched if stage == 'perturbations' else perturbations_longtail
            _perturbation_options(module, recipe, cfg, root)
            module.run()
            if stage == 'perturbations':
                module.build_report()
    elif stage == 'author-recipes':
        from experiments import author_recipes
        if 'author_seeds' in recipe:
            author_recipes.SEEDS = recipe['author_seeds']
        author_recipes.run(root, cfg)
        author_recipes.report(root)
    elif stage == 'synthetic':
        from experiments import synthetic, competitors
        competitors.method_specs, competitors.fl_primary_spec = competitor_policy(recipe, args.mode)
        rawcfg = {k: v for k, v in cfg.items() if k != 'loader_fn'}
        run_root, sidecar, _, _ = synthetic.run_headline(rawcfg, tag)
        synthetic.compute_eta_report(rawcfg, run_root, sidecar)
    elif stage == 'synthetic-figures':
        from experiments import synthetic
        mf = json.loads(Path(root, 'run_manifest.json').read_text())
        onset = mf['best_onset']
        lam = mf.get('lambda_final', mf.get('best', {}).get('lambda'))
        rawcfg = {k: v for k, v in cfg.items() if k != 'loader_fn'}
        synthetic.figure_bound_predictiveness(rawcfg, tag, onset)
        synthetic.figure_activation_criterion(rawcfg, tag, onset, lam)
    elif stage in ('frozen-depth', 'frozen-dropout'):
        from experiments import frozen
        frozen.CONFIG.clear()
        frozen.CONFIG.update(cfg)
        fn = frozen.run_depth_sweep if stage == 'frozen-depth' else frozen.run_dropout_ablation
        fn(cfg['_dataset_raw'], cfg['_encoder'], tag, task=cfg.get('_task'))
    elif stage == 'headline-selection':
        from experiments.headline_selection import select_headline
        onset, lam, schedule, decision = select_headline(cfg, tag)
        Path(root, 'relock_decision.json').write_text(json.dumps({
            'best_onset': onset, 'lambda_cosine': lam, 'lr_schedule': schedule, **decision}, indent=2))
        print('Selection complete. The decision is in relock_decision.json; constant lambda is still a separate selection.')
    elif stage == 'onset-control':
        from experiments.onset_control import main
        if not recipe.get('onset_jobs'):
            raise ValueError('onset-control requires the original job policy in recipe.onset_jobs.')
        main(recipe['onset_jobs'])
    elif stage == 'extract':
        print('Embedding cache:', cfg['embedding_dir'])
    else:
        raise ValueError(stage)


def main(argv=None, default_stage=None):
    p = parser(default_stage)
    a = p.parse_args(argv)
    try:
        recipe, cfg, stages, root, tag = prepare(a)
        if a.dry_run:
            print(json.dumps({'stages': stages, 'mode': a.mode, 'run_root': root,
                              'config': cfg, 'data': recipe['data'], 'selected': recipe.get('selected'),
                              'competitors': competitor_policy(recipe, a.mode) if any(s in stages for s in ('competitors', 'synthetic')) else None}, indent=2))
            return 0
        training = any(s in stages for s in ('nescal', 'competitors', 'synthetic', 'headline-selection'))
        if not a.run_root and training and Path(root).exists():
            raise ValueError(f'Output already exists: {root}. Use a new --tag or an explicit --run-root.')
        needs_data = any(s not in ('report', 'corruptions', 'ood', 'random-graph') for s in stages)
        if recipe['data']['kind'] == 'frozen' and any(s in stages for s in ('corruptions', 'ood')):
            cfg['do_robustness'] = 'corruptions' in stages
            cfg['do_ood'] = 'ood' in stages
            needs_data = True
        if needs_data:
            cfg = bind_data(recipe, cfg, root, extract='extract' in stages)
        if not a.run_root:
            root = str(Path(cfg['base_out']) / cfg['dataset'] / cfg['arch_name'] / tag)
        Path(root).mkdir(parents=True, exist_ok=True)
        _set_core_config(cfg)
        saved = deepcopy(recipe)
        saved['config'] = serializable_config(cfg)
        saved['mode'] = a.mode
        if training or not Path(root, 'settings.json').exists():
            Path(root, 'settings.json').write_text(json.dumps(saved, indent=2) + '\n')
        for stage in stages:
            print(f'\n[{stage}] {root}', flush=True)
            execute_stage(stage, recipe, cfg, root, tag, a)
        return 0
    except (ValueError, KeyError, FileNotFoundError) as e:
        p.error(str(e))
