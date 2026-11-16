import os
import gc
import re
import copy
import glob
import math
import time
import shutil
import pickle
import argparse
import datetime
import statistics
import traceback

import numpy as np
import torch
from scipy import stats as sps
import experiments.nescal as rn
import experiments.reporting as ra
import experiments.posthoc as rpo
import experiments.robustness as rx
from experiments.common import run_dir, RunTracker, train_one, select_by_val_nll, write_manifest, load_json, load_pkl
from data.image_datasets import load_dataset
from handlers.trainer import Trainer

ROOTS = {}
OUT_TAG = 'additional'
SEL_SEEDS, CONF_SEEDS = [18, 42], [613, 2026]
ALPHAS = [0.1, 0.5, 0.9]
EXTRA_ONSETS = [40, 160]
COARSE_ALPHAS = [0.1, 0.5, 0.9]
COARSE_ONSETS_BY_JOB = {'tin': [20, 80]}
COARSE_ONSETS = [40, 80, 125, 160]
NCR_K, NCR_T = 10, 2.0
NCR_FORM = 'code'
MIN_FREE_GB = 30
SMOKE_EPOCHS = 2

ERRORS, TRACKER, SMOKE = [], None, False
_LC = {}
TS_KEYS = ('nll', 'ece', 'brier_score', 'accuracy')


def _now():
    return datetime.datetime.now().strftime('%a %H:%M:%S')


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tag(x):
    return f'{x:g}'.replace('.', 'p')


def _out_tag():
    return OUT_TAG + ('_smoke' if SMOKE else '')


def _resolve(root, p):
    if p and os.path.isdir(p):
        return p
    p = (p or '').rstrip('/')
    alt = os.path.join(root, os.path.basename(os.path.dirname(p)), os.path.basename(p))
    return alt if os.path.isdir(alt) else p


def _loaders(cfg, seed):
    key = (cfg['dataset'], cfg['arch_name'], cfg['batch_size'], seed)
    if key not in _LC:
        if any(k[:3] != key[:3] for k in _LC) or len(_LC) >= 2:
            _LC.clear()
            _free()
        fn = cfg.get('loader_fn')
        _LC[key] = fn(seed) if fn else load_dataset(cfg['dataset'], val_split=0.1, batch_size=cfg['batch_size'], seed=seed,
                                                    data_root=cfg['data_root'], num_workers=cfg['num_workers'])
    return _LC[key]


def context(name, root=None):
    root = (root or ROOTS[name]).rstrip('/')
    rm, cm = os.path.join(root, 'run_manifest.json'), os.path.join(root, 'competitors_manifest.json')
    for p in (rm, cm):
        if not os.path.exists(p):
            raise FileNotFoundError(f'{name}: {p} missing -- fix ROOTS')
    mf, comp = load_json(rm), load_json(cm)
    cfg = copy.deepcopy(mf['config'])
    for k in ('dataset', 'arch_name', 'model_params', 'num_epochs', 'batch_size', 'num_bins', 'lr', 'momentum',
              'weight_decay', 'diag_every_n', 'data_root', 'base_out', 'num_workers', 'nescal'):
        if k not in cfg:
            raise KeyError(f'{name}: run_manifest config lacks {k!r}')
    xe = _resolve(root, mf.get('runs', {}).get('XE_baseline', {}).get(str(SEL_SEEDS[0]), ''))
    mp = os.path.join(xe, 'model_params.pkl')
    if not os.path.exists(mp):
        raise FileNotFoundError(f'{name}: cannot verify the executed recipe, {mp} missing')
    ck = load_pkl(mp)
    for k in ('num_epochs', 'lr', 'momentum', 'weight_decay', 'lr_milestones_frac', 'lr_gamma', 'lr_schedule', 'optimizer'):
        if k in ck:
            cfg[k] = list(ck[k]) if isinstance(ck[k], tuple) else ck[k]
    cfg['early_stopping'] = bool(cfg.get('early_stopping', False))
    return {'name': name, 'root': root, 'mf': mf, 'comp': comp, 'cfg': cfg, 'ds': cfg['dataset'],
            'onset': int(mf['best_onset']), 'lam': float(mf['lambda_cosine'])}


def ckpt_for(ctx, seed, onset):
    xe = _resolve(ctx['root'], ctx['mf'].get('runs', {}).get('XE_baseline', {}).get(str(seed), ''))
    best = (None, 0)
    for p in glob.glob(os.path.join(xe, 'checkpoints', 'epoch_*.pt')):
        m = re.fullmatch(r'epoch_(\d+)\.pt', os.path.basename(p))
        if m and best[1] <= int(m.group(1)) <= onset:
            best = (p, int(m.group(1)))
    return best


def cells_for(ctx, grid='full'):
    onsets = sorted({int(e) for e in ctx['cfg'].get('candidate_onsets', [])} | set(EXTRA_ONSETS))
    onsets = [e for e in onsets if 0 < e < ctx['cfg']['num_epochs']]
    coarse = [(e, a) for e in COARSE_ONSETS_BY_JOB.get(ctx['name'], COARSE_ONSETS) for a in COARSE_ALPHAS]
    rest = [] if grid == 'coarse' else [(e, a) for e in onsets for a in ALPHAS if (e, a) not in coarse]
    return coarse + rest


def preflight(names):
    problems, ctxs = [], {}
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'[preflight] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} device={dev} smoke={SMOKE}', flush=True)
    for n in names:
        try:
            ctx = context(n)
        except Exception as e:
            problems.append(f'{n}: {type(e).__name__}: {e}')
            continue
        ctxs[n] = ctx
        cfg, cells = ctx['cfg'], cells_for(ctx)
        print(f'\n[preflight] {n} root={ctx["root"]}\n  {ctx["ds"]}/{cfg["arch_name"]} epochs={cfg["num_epochs"]} bs={cfg["batch_size"]} '
              f'lr={cfg["lr"]} milestones={cfg.get("lr_milestones_frac")} wd={cfg["weight_decay"]} '
              f'early_stopping={cfg["early_stopping"]}\n  NeSCal headline: onset={ctx["onset"]} lambda={ctx["lam"]} '
              f'| reportable={sorted(ctx["mf"].get("reportable", {}))}', flush=True)
        print(f'  NCR grid: {len(cells)} cells, onsets={sorted({c[0] for c in cells})} alphas={sorted({c[1] for c in cells})}', flush=True)
        for s in SEL_SEEDS + CONF_SEEDS:
            p, e = ckpt_for(ctx, s, max(c[0] for c in cells))
            if p is None:
                problems.append(f'{n}: seed {s} has no XE_baseline checkpoint at all (every branch would start from scratch)')
        p, e = ckpt_for(ctx, SEL_SEEDS[0], ctx['onset'])
        if p:
            sch = torch.load(p, map_location='cpu', weights_only=False)['scheduler_state']
            print(f'  checkpoint {os.path.basename(p)}: milestones={dict(sch.get("milestones", {}))} last_epoch={sch.get("last_epoch")}', flush=True)
        gaps = {c[0]: ckpt_for(ctx, SEL_SEEDS[0], c[0])[1] for c in cells}
        print(f'  onset -> resume epoch (seed {SEL_SEEDS[0]}): {gaps}', flush=True)
        free = shutil.disk_usage(cfg['base_out']).free / 1e9
        print(f'  free disk under {cfg["base_out"]}: {free:.0f} GB', flush=True)
        if free < MIN_FREE_GB:
            problems.append(f'{n}: only {free:.0f} GB free')
    print('\n[preflight] ' + ('PROBLEMS:\n  ' + '\n  '.join(problems) if problems else 'all checks passed'), flush=True)
    return ctxs, problems


def run_path(ctx, method, seed, cfg=None):
    c = cfg or ctx['cfg']
    return os.path.join(c['base_out'], ctx['ds'], c['arch_name'], _out_tag(), method, f'seed_{seed}')


def finished(root):
    return os.path.exists(os.path.join(root, 'val_results.pkl')) and os.path.exists(os.path.join(root, 'test_results.pkl'))


def train(ctx, seed, method, loss_params, onset, label, cfg=None, fresh=False, **extra):
    c = copy.deepcopy(cfg or ctx['cfg'])
    root = run_path(ctx, method, seed, c)
    div = os.path.join(root, 'DIVERGED')
    if os.path.exists(div):
        print(f'[skip] {label}: previously diverged', flush=True)
        return None
    if finished(root):
        print(f'[reuse] {label}', flush=True)
        return root
    free = shutil.disk_usage(c['base_out']).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(f'only {free:.0f} GB free under {c["base_out"]}; stopping cleanly (relaunch resumes)')
    try:
        ck, start = (None, 0) if fresh else ckpt_for(ctx, seed, onset)
        if SMOKE:
            c['num_epochs'] = onset + SMOKE_EPOCHS
        root = run_dir(c['base_out'], ctx['ds'], c['arch_name'], _out_tag(), method, seed)
        kw = rn._base_kwargs(c, _loaders(c, seed), seed, root, loss_params=loss_params, resume_from=ck,
                             activation_params={'strategy': 'fixed', 'fixed_epoch': onset}, log_graph_maturity=False,
                             checkpoint_epochs=[], early_stopping=c['early_stopping'], **extra)
        print(f'[{_now()}] start {label} (resume epoch {start})', flush=True)
        _, _, diverged, _ = train_one(TRACKER, label=label, epochs_to_train=c['num_epochs'] - start, run_kwargs=kw)
        _free()
        for p in glob.glob(os.path.join(root, 'checkpoints', '*')):
            os.remove(p)
        if diverged:
            with open(div, 'w') as f:
                f.write(label)
            return None
        return root
    except SystemExit:
        raise
    except Exception as e:
        ERRORS.append({'label': label, 'error': f'{type(e).__name__}: {e}'})
        print(f'!! {label} FAILED: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        _free()
        return None


def ncr_params(alpha):
    return {'primary_loss': 'XE', 'secondary_loss': 'ncr',
            'secondary_params': {'alpha': alpha, 'k': NCR_K, 'temperature': NCR_T, 'form': NCR_FORM}}


def cell_method(e, a, prefix='ncr'):
    return f'{prefix}_e{e}_a{_tag(a)}'


def val_nll(root):
    try:
        return float(load_pkl(os.path.join(root, 'val_results.pkl'))['nll'])
    except Exception:
        return None


def select_finished(ctx, keys, method_fn, seeds, cfg=None):
    scores = {}
    for key in keys:
        per = {}
        for s in seeds:
            root = run_path(ctx, method_fn(key), s, cfg)
            if os.path.exists(os.path.join(root, 'DIVERGED')):
                per[s] = None
            elif finished(root):
                per[s] = val_nll(root)
            else:
                per = None
                break
        if per is not None:
            scores[key] = per
    if not scores:
        raise RuntimeError(f'{ctx["name"]}: no config finished on seeds {seeds}')
    best, table = select_by_val_nll(scores)
    if not math.isfinite(table[best]):
        raise RuntimeError(f'{ctx["name"]}: every finished config diverged: {scores}')
    return best, table


def method_dirs(ctx, label):
    out = {}
    for fn in ('run_manifest.json', 'competitors_manifest.json', 'mixup_manifest.json', 'ncr_manifest.json'):
        p = os.path.join(ctx['root'], fn)
        if os.path.exists(p):
            for s, d in load_json(p).get('reportable', {}).get(label, {}).items():
                if d:
                    out[int(s)] = _resolve(ctx['root'], d)
    if label not in ('FL_FINAL', 'ls_alone_FINAL') or not out:
        return out
    final = {}
    for s, d in out.items():
        sub = os.path.join(d, 'nllsel')
        if os.path.exists(os.path.join(sub, 'test_data.pkl')):
            final[s] = sub
    for p in sorted(glob.glob(os.path.join(ctx['root'], 'nllsel_summary_*.json'))):
        if p.endswith('_smoke.json'):
            continue
        summ = load_json(p)
        if (summ.get('label') or {'FL': 'FL_FINAL', 'LS': 'ls_alone_FINAL'}.get(summ.get('method'))) == label:
            for s, d in (summ.get('dirs_new') or {}).items():
                if os.path.exists(os.path.join(d, 'test_data.pkl')):
                    final[int(s)] = d
    missing = sorted(set(out) - set(final))
    if missing:
        print(f'  !! {ctx["name"]} {label}: no validation-NLL-epoch dir for seeds {missing}; using the own-loss-epoch run there', flush=True)
    return {**out, **final}


def seed_metrics(rp):
    out = {}
    pp, tp = os.path.join(rp, 'posthoc_results.pkl'), os.path.join(rp, 'test_results.pkl')
    if os.path.exists(pp):
        ph = load_pkl(pp)
        out.update({k: ph.get('pre', {}).get(k) for k in TS_KEYS})
        ts = ph.get('post', {}).get('ts', {})
        out.update({f'{k}_ts': ts.get(k) for k in ('nll', 'ece')})
    elif os.path.exists(tp):
        t = load_pkl(tp)
        out.update({k: t.get(k) for k in TS_KEYS})
    return {k: float(v) for k, v in out.items() if v is not None}


def paired_ci(a, b):
    shared = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in shared], dtype=float)
    if len(d) < 2:
        return {'n': len(d), 'mean': float(d.mean()) if len(d) else float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    h = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return {'n': len(d), 'mean': float(d.mean()), 'lo': float(d.mean() - h), 'hi': float(d.mean() + h)}


def md_table(rows, columns, maximize=('accuracy',), nd=4):
    print('| method | ' + ' | '.join(columns) + ' |\n|---|' + '---|' * len(columns), flush=True)
    rank = {}
    for c in columns:
        vals = sorted({round(statistics.mean(r[c]), nd) for r in rows.values() if r.get(c)}, reverse=c in maximize)
        rank[c] = vals[:2]
    for lab, r in rows.items():
        cells = []
        for c in columns:
            if not r.get(c):
                cells.append('--')
                continue
            m, sd = statistics.mean(r[c]), (statistics.stdev(r[c]) if len(r[c]) > 1 else 0.0)
            txt = f'{m:.{nd}f} ± {sd:.{nd}f}'
            pos = rank[c].index(round(m, nd)) if round(m, nd) in rank[c] else None
            cells.append(f'**{txt}**' if pos == 0 else f'*{txt}*' if pos == 1 else txt)
        print(f'| {lab} (n={max((len(v) for v in r.values()), default=0)}) | ' + ' | '.join(cells) + ' |', flush=True)


def edge_js(rp):
    p = os.path.join(rp, 'train_time_diagnostics.pkl')
    if not os.path.exists(p):
        return None, None
    th = load_pkl(p).get('theory', {}) or {}
    bd = th.get('bound_diagnostics') or {}
    final = next((v for k, v in bd.items() if 'js' in k.lower() and 'mean' in k.lower() and isinstance(v, (int, float))), None)
    return th.get('mean_edge_js_at_activation'), final


def report(ctx, extra_labels=('ncr_FINAL',)):
    labels = ['XE_baseline', 'FL_FINAL', 'ls_alone_FINAL', 'mbls_FINAL', *extra_labels, 'ncl_FINAL_cosine']
    cols = ['nll', 'nll_ts', 'ece', 'ece_ts', 'brier_score', 'accuracy']
    per = {lab: {s: seed_metrics(d) for s, d in method_dirs(ctx, lab).items()} for lab in labels}
    rows = {lab: {c: [m[c] for m in v.values() if c in m] for c in cols} for lab, v in per.items() if v}
    print(f'\n### {ctx["ds"]} / {ctx["cfg"]["arch_name"]}: NCR vs reference methods (test; *_ts = validation-fitted TS)\n', flush=True)
    md_table(rows, cols)
    for lab in extra_labels:
        if not per.get(lab) or not per.get('ncl_FINAL_cosine'):
            continue
        print(f'\n{lab} minus NeSCal, paired over seeds (95% t interval; negative favours {lab} on losses):', flush=True)
        for c in cols:
            a = {s: m[c] for s, m in per[lab].items() if c in m}
            b = {s: m[c] for s, m in per['ncl_FINAL_cosine'].items() if c in m}
            ci = paired_ci(a, b)
            print(f'  {c:12s} {ci["mean"]:+.4f}  [{ci["lo"]:+.4f}, {ci["hi"]:+.4f}]  n={ci["n"]}', flush=True)
    print('\nEdge JS on the frozen kNN graph (at activation -> at selected weights):', flush=True)
    for lab in ('XE_baseline', *extra_labels, 'ncl_FINAL_cosine'):
        vals = [edge_js(d) for d in method_dirs(ctx, lab).values()]
        act = [v[0] for v in vals if v[0] is not None]
        fin = [v[1] for v in vals if v[1] is not None]
        print(f'  {lab:20s} {np.mean(act) if act else float("nan"):.4f} -> {np.mean(fin) if fin else float("nan"):.4f}', flush=True)


def rebuild_report(ctx):
    old = os.path.join(ctx['root'], 'report_csvs')
    try:
        if os.path.isdir(old):
            shutil.copytree(old, old + '_backup_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        prev, rn.CONFIG = rn.CONFIG, ctx['cfg']
        try:
            ra.assemble_report(ctx['root'])
        finally:
            rn.CONFIG = prev
    except Exception as e:
        print(f'  !! report rebuild failed ({type(e).__name__}: {e}); ncr_manifest.json is intact', flush=True)


def smoke_posthoc(ctx, rp, seed):
    from posthoc.calibrators import apply_posthoc_suite
    m = Trainer.load_model(rp)
    res = apply_posthoc_suite(m.all_val_logits, m.all_val_labels, m.all_val_features, m.all_test_logits, m.all_test_labels,
                              m.all_test_features, calibrators=('ts',), seed=seed, num_bins=ctx['cfg']['num_bins'],
                              lce_projection=m.lce_projection)
    with open(os.path.join(rp, 'posthoc_results.pkl'), 'wb') as f:
        pickle.dump(res, f)
    print(f'  smoke posthoc OK: pre NLL {res["pre"]["nll"]:.4f} -> TS NLL {res["post"]["ts"]["nll"]:.4f}', flush=True)


def stage_select(ctx, grid, shard):
    cells = cells_for(ctx, grid)
    i, n = shard
    mine = [c for j, c in enumerate(cells) if j % n == i]
    print(f'\n########## [{_now()}] NCR SELECT {ctx["name"]} | {len(mine)}/{len(cells)} cells (shard {i}/{n}) | '
          f'seeds={SEL_SEEDS} ##########', flush=True)
    for e, a in mine:
        for s in SEL_SEEDS:
            train(ctx, s, cell_method(e, a), ncr_params(a), e, f'[{ctx["name"]}] NCR e={e} alpha={a} seed={s}')


def stage_finish(ctx, grid, do_extras=True):
    cells = cells_for(ctx, grid)
    (e, a), table = select_finished(ctx, cells, lambda c: cell_method(*c), SEL_SEEDS)
    done = [c for c in cells if c in table]
    print(f'\n########## [{_now()}] NCR FINISH {ctx["name"]} | selected over {len(done)}/{len(cells)} cells ##########', flush=True)
    for c, v in sorted(table.items(), key=lambda kv: kv[1])[:8]:
        print(f'  e={c[0]:<4d} alpha={c[1]:<4} val NLL {v:.4f}', flush=True)
    print(f'  -> winner e={e} alpha={a}', flush=True)
    onsets, alphas = sorted({c[0] for c in done}), sorted({c[1] for c in done})
    if e in (onsets[0], onsets[-1]) or a in (alphas[0], alphas[-1]):
        print(f'  NOTE: winner is on a grid edge (onsets {onsets[0]}..{onsets[-1]}, alphas {alphas[0]}..{alphas[-1]}).', flush=True)
    method = cell_method(e, a)
    rep = {str(s): train(ctx, s, method, ncr_params(a), e, f'[{ctx["name"]}] NCR winner seed={s}') for s in SEL_SEEDS + CONF_SEEDS}
    rep = {s: r for s, r in rep.items() if r}
    for c in done:
        if c != (e, a):
            for s in SEL_SEEDS:
                d = run_path(ctx, cell_method(*c), s)
                for fn in ('model_state.pth', 'test_data.pkl', 'val_data.pkl', 'graph_data.pkl'):
                    if os.path.exists(os.path.join(d, fn)):
                        os.remove(os.path.join(d, fn))
                shutil.rmtree(os.path.join(d, 'plots'), ignore_errors=True)
    path = os.path.join(ctx['cfg']['base_out'], ctx['ds'], ctx['cfg']['arch_name'], _out_tag(), 'ncr_manifest.json')
    root_path = path if SMOKE else os.path.join(ctx['root'], 'ncr_manifest.json')
    old = load_json(root_path) if os.path.exists(root_path) else {}
    manifest = {**old, 'timestamp': _out_tag(), 'reportable': {**old.get('reportable', {}), 'ncr_FINAL': rep},
                'winners': {**old.get('winners', {}), 'ncr': {'candidate': {'e': e, 'alpha': a, 'k': NCR_K, 'T': NCR_T},
                                                              'selection': {str(k): v for k, v in table.items()},
                                                              'cells_finished': len(done), 'cells_planned': len(cells)}},
                'errors': ERRORS}
    write_manifest(root_path, manifest)
    if not SMOKE:
        write_manifest(path, manifest)
    if SMOKE:
        for s, r in rep.items():
            smoke_posthoc(ctx, r, int(s))
    else:
        _free()
        rpo.run(ctx['root'], ctx['ds'], ctx['cfg']['arch_name'], data_root=ctx['cfg']['data_root'], only_labels={'ncr_FINAL'}, skip_existing=True)
        if do_extras:
            _free()
            rx.run(ctx['root'], ctx['ds'], do_corruption=True, do_ood=True, data_root=ctx['cfg']['data_root'], only_labels={'ncr_FINAL'}, skip_existing=True)
        _free()
        rebuild_report(ctx)
    report(ctx)


def main(argv=None):
    global TRACKER, SMOKE, COARSE_ONSETS, COARSE_ALPHAS, SEL_SEEDS, CONF_SEEDS, COARSE_ONSETS_BY_JOB
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', nargs='+', default=list(ROOTS), choices=list(ROOTS))
    ap.add_argument('--stage', default='all', choices=['select', 'finish', 'all', 'report'])
    ap.add_argument('--grid', default='full', choices=['coarse', 'full'])
    ap.add_argument('--shard', default='0/1', help='i/n: run every n-th grid cell, for several processes on one GPU')
    ap.add_argument('--no_extras', action='store_true', help='skip CIFAR-C and SVHN for ncr_FINAL')
    ap.add_argument('--preflight_only', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='1 cell, 1+1 seeds, 2 epochs after onset, under <OUT_TAG>_smoke')
    a = ap.parse_args(argv)
    SMOKE = a.smoke
    if SMOKE:
        COARSE_ONSETS, COARSE_ALPHAS, SEL_SEEDS, CONF_SEEDS = COARSE_ONSETS[1:2], COARSE_ALPHAS[1:2], SEL_SEEDS[:1], CONF_SEEDS[:1]; COARSE_ONSETS_BY_JOB = {k: v[1:2] for k, v in COARSE_ONSETS_BY_JOB.items()}
    ctxs, problems = preflight(a.jobs)
    if problems:
        return 1
    if a.preflight_only:
        return 0
    TRACKER = RunTracker(10 ** 7, label='ncr')
    shard = tuple(int(x) for x in a.shard.split('/'))
    t0 = time.time()
    for n in a.jobs:
        ctx = ctxs[n]
        try:
            if a.stage in ('select', 'all'):
                stage_select(ctx, 'coarse' if SMOKE else a.grid, shard)
            if a.stage in ('finish', 'all'):
                stage_finish(ctx, 'coarse' if SMOKE else a.grid, do_extras=not a.no_extras)
            if a.stage == 'report':
                report(ctx)
        except SystemExit as e:
            print(f'!! {e}', flush=True)
            return 2
        except Exception as e:
            ERRORS.append({'label': f'{n}:{a.stage}', 'error': f'{type(e).__name__}: {e}'})
            print(f'!! {n} {a.stage} FAILED: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
    print(f'\n=== NCR {a.stage} done in {(time.time() - t0) / 3600:.2f} h | errors: {ERRORS or "none"} ===', flush=True)
    return 0
