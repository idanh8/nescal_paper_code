import os
import re
import glob
import json
import math
import time
import argparse
import importlib.util
import traceback
import statistics

import numpy as np
import torch
from scipy.optimize import minimize_scalar
import experiments.ncr as rnc
import experiments.nescal as rn
from experiments.common import RunTracker, write_manifest, load_json, load_pkl
from handlers.trainer import Trainer
from metrics.graph_metrics import _embed_probs_labels
from losses.nescal import PreNeighborhoodConsistencyLoss, js_divergence_torch
from losses.ncr import ann_knn, recall_at_k

TIN_ROOT = ''
ANALYSIS_EXTRA_ROOTS = {}
METHODS = ['XE_baseline', 'FL_FINAL', 'ls_alone_FINAL', 'mmce_FINAL', 'mdca_FINAL', 's_avuc_FINAL', 'mbls_FINAL',
           'mixup_FINAL', 'ncr_FINAL', 'ncl_FINAL_cosine']
REF = 'ncl_FINAL_cosine'
LABEL_FIXED_AGAINST = ['XE_baseline', 'FL_FINAL', 'ncr_FINAL']
ACC_TOL = 0.0025
FACTORIAL_JOBS, DIVERGENCE_JOBS, MIXUP_JOBS = ['c100_r50', 'c10_r50'], ['c100_r50', 'c10_r50'], ['c100_r50', 'c10_r50']
FACTORIAL_CELLS = [('edge', 'minibatch'), ('mixture', 'minibatch'), ('mixture', 'frozen')]
FACTORIAL_LAMS = {'minibatch': [2.0, 5.0, 10.0, 20.0], 'frozen': [10.0, 20.0, 30.0]}
DIVERGENCES = ['kl', 'skl', 'logit_l2']
PENALTY_TARGETS = [0.1, 0.3, 1.0, 3.0, 10.0]
SCALE_JOBS = ['c100_r50', 'c10_r50']
SCALE_ARMS = {'ann_ivf': {'graph_backend': 'faiss_ivf'}, 'ann_hnsw': {'graph_backend': 'faiss_hnsw'},
              'rebuild10': {'graph_recompute_every': 10}, 'rebuild25': {'graph_recompute_every': 25},
              'cache_drift': {}, 'cache_sync1': {'cache_sync_epochs': 1}, 'cache_sync5': {'cache_sync_epochs': 5}}
TIN_SCALE_ARMS, TIN_SEEDS = ['ann_ivf', 'rebuild25'], [18, 42, 613, 2026]
MIXUP_LAM_MULT = [0.5, 1.0, 2.0]
MIXUP_BASE_ALPHAS = [0.2, 0.4]
RANKMIXUP_JOBS = ['c100_r50', 'c10_r50']
RANKMIXUP_ALPHAS, RANKMIXUP_M, RANKMIXUP_W = [2.0, 1.0], 2.0, 0.1
IMAGENET_N, IMAGENET_K, IMAGENET_D = 1_281_167, 1000, 2048
QUERY_SAMPLE = 20_000

ERRORS = []


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels):
    y = np.asarray(val_labels).astype(int)
    idx = np.arange(len(y))

    def nll(log_t):
        return -np.log(np.clip(_softmax(np.asarray(val_logits) / np.exp(log_t))[idx, y], 1e-300, 1.0)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3.0, 3.0), method='bounded', options={'xatol': 1e-4}).x))


def ece_score(conf, correct, num_bins=15):
    idx = np.clip(np.digitize(conf, np.linspace(0.0, 1.0, num_bins + 1)) - 1, 0, num_bins - 1)
    out = 0.0
    for b in np.unique(idx):
        m = idx == b
        out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def score(p, y, num_bins=15):
    y = np.asarray(y).astype(int)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = (pred == y).astype(float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return {'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
            'ece': ece_score(conf, correct, num_bins), 'brier': float(((p - onehot) ** 2).sum(axis=1).mean()),
            'acc': float(correct.mean()), 'conf': float(conf.mean())}


def load_run(rp):
    tp, vp = os.path.join(rp, 'test_data.pkl'), os.path.join(rp, 'val_data.pkl')
    if not (os.path.exists(tp) and os.path.exists(vp)):
        return None

    def logits(d, split):
        z = d.get(f'{split}_logits')
        if z is not None and np.asarray(z).ndim == 2:
            return np.asarray(z, dtype=np.float64)
        return np.log(np.clip(np.asarray(d[f'{split}_conf_vectors'], dtype=np.float64), 1e-12, 1.0))
    t, v = load_pkl(tp), load_pkl(vp)
    r = {'tl': logits(t, 'test'), 'ty': np.asarray(t['test_labels']).astype(int),
         'vl': logits(v, 'val'), 'vy': np.asarray(v['val_labels']).astype(int)}
    r['T'] = fit_temperature(r['vl'], r['vy'])
    r['p'], r['p_ts'] = _softmax(r['tl']), _softmax(r['tl'] / r['T'])
    return r


def run_metrics(rp, stale_check=True):
    r = load_run(rp)
    if r is None:
        return None
    m = score(r['p'], r['ty'])
    ts = score(r['p_ts'], r['ty'])
    m.update({'nll_ts': ts['nll'], 'ece_ts': ts['ece'], 'brier_ts': ts['brier'], 'T': r['T'], 'gap': m['nll'] - ts['nll']})
    sp = os.path.join(rp, 'test_results.pkl')
    if stale_check and os.path.exists(sp):
        stored = load_pkl(sp).get('ece')
        if stored is not None and abs(float(stored) - m['ece']) > 2e-4:
            print(f'  [provenance] {rp}: stored ECE {float(stored):.5f} != recomputed {m["ece"]:.5f}', flush=True)
    return m


def _pm(vals, nd=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '--'
    return f'{statistics.mean(vals):.{nd}f} ± {(statistics.stdev(vals) if len(vals) > 1 else 0.0):.{nd}f}'


def _ci(a, b):
    return rnc.paired_ci(a, b)


def _fmt_ci(ci, nd=4):
    return f'{ci["mean"]:+.{nd}f} [{ci["lo"]:+.{nd}f}, {ci["hi"]:+.{nd}f}]'


def _verdict(ci, lower_is_better=True):
    if not (math.isfinite(ci['lo']) and math.isfinite(ci['hi'])):
        return 'n/a'
    if ci['lo'] <= 0.0 <= ci['hi']:
        return 'tie'
    return 'win' if (ci['hi'] < 0.0) == lower_is_better else 'loss'


def label_fixed(a, b, ts=False):
    if not np.array_equal(a['ty'], b['ty']):
        raise ValueError('test labels differ between runs; the test order is not shared')
    pa, pb = (a['p_ts'], b['p_ts']) if ts else (a['p'], b['p'])
    same = pa.argmax(axis=1) == pb.argmax(axis=1)
    return score(pa[same], a['ty'][same]), score(pb[same], b['ty'][same]), float(same.mean())


def grid_cells(root):
    out = {}
    for d in glob.glob(os.path.join(root, '*onset*_lam*', 'seed_*')):
        m = re.search(r'onset(\d+)_lam([\d.]+)', d)
        if m and os.path.exists(os.path.join(d, 'val_results.pkl')) and 'const' not in d and 'cosLR' not in d:
            out.setdefault((int(m.group(1)), float(m.group(2))), {})[int(d.rsplit('_', 1)[-1])] = d
    return out


def accuracy_matched(ctx):
    xe = rnc.method_dirs(ctx, 'XE_baseline')
    cells = grid_cells(ctx['root'])
    seeds = sorted({s for per in cells.values() for s in per} & set(xe))
    if not cells or not seeds:
        return None
    ce_val = statistics.mean(float(load_pkl(os.path.join(xe[s], 'val_results.pkl'))['accuracy']) for s in seeds)
    candidates = []
    for key, per in cells.items():
        if not all(s in per for s in seeds):
            continue
        val = [load_pkl(os.path.join(per[s], 'val_results.pkl')) for s in seeds]
        acc, nll = statistics.mean(float(v['accuracy']) for v in val), statistics.mean(float(v['nll']) for v in val)
        if abs(acc - ce_val) <= ACC_TOL:
            candidates.append((nll, key, acc))
    if not candidates:
        return {'n_within_tol': 0, 'n_cells': len(cells), 'ce_val_acc': ce_val}
    _, key, acc = min(candidates)
    m_ce = {s: run_metrics(xe[s], stale_check=False) for s in seeds}
    m_ne = {s: run_metrics(cells[key][s], stale_check=False) for s in seeds}
    if any(v is None for v in list(m_ce.values()) + list(m_ne.values())):
        return {'n_within_tol': len(candidates), 'n_cells': len(cells), 'ce_val_acc': ce_val, 'config': key, 'error': 'logits missing'}
    delta = {k: _ci({s: m_ne[s][k] for s in seeds}, {s: m_ce[s][k] for s in seeds}) for k in ('nll', 'nll_ts', 'ece', 'ece_ts', 'acc', 'conf')}
    return {'n_within_tol': len(candidates), 'n_cells': len(cells), 'ce_val_acc': ce_val, 'matched_val_acc': acc,
            'config': key, 'seeds': seeds, 'delta': delta}


def analyse_setting(name, root):
    ctx = {'name': name, 'root': root.rstrip('/')}
    print(f'\n{"=" * 100}\n## {name}: {root}\n{"=" * 100}', flush=True)
    per = {}
    for lab in METHODS:
        dirs = rnc.method_dirs(ctx, lab)
        vals = {s: run_metrics(d) for s, d in dirs.items()}
        vals = {s: v for s, v in vals.items() if v is not None}
        if vals:
            per[lab] = vals
    if REF not in per:
        print(f'  !! no logits for {REF}; skipping', flush=True)
        return None
    keys = ['nll', 'nll_ts', 'gap', 'ece', 'ece_ts', 'brier', 'acc', 'conf']
    print('\nRecomputed from logits (gap = NLL removable by a validation-fitted temperature; conf = mean top-label confidence)\n', flush=True)
    print('| method | n | ' + ' | '.join(keys) + ' | T |\n|---|---|' + '---|' * (len(keys) + 1), flush=True)
    for lab, v in per.items():
        print(f'| {lab} | {len(v)} | ' + ' | '.join(_pm([m[k] for m in v.values()]) for k in keys) + f' | {_pm([m["T"] for m in v.values()], 3)} |', flush=True)

    out = {'setting': name, 'root': root, 'methods': {lab: {str(s): m for s, m in v.items()} for lab, v in per.items()}}
    if 'XE_baseline' in per:
        xe, ne = per['XE_baseline'], per[REF]
        print('\nDoes accuracy explain the ECE change? NeSCal minus CE, paired over seeds:', flush=True)
        for k in ('acc', 'ece', 'conf', 'gap'):
            print(f'  Δ{k:5s} {_fmt_ci(_ci({s: m[k] for s, m in ne.items()}, {s: m[k] for s, m in xe.items()}))}', flush=True)

    print(f'\nLabel-fixed: {REF} vs each method on the test samples both label identically (NeSCal minus other, paired over seeds):\n', flush=True)
    print('| against | agree | ΔECE | ΔNLL | ΔBrier | Δconf | ΔECE after TS | ΔNLL after TS |\n|---|---|---|---|---|---|---|---|', flush=True)
    out['label_fixed'] = {}
    for lab in LABEL_FIXED_AGAINST:
        if lab not in per:
            continue
        ne_dirs, other_dirs = rnc.method_dirs(ctx, REF), rnc.method_dirs(ctx, lab)
        base, ts, cover = {}, {}, {}
        for s in sorted(set(ne_dirs) & set(other_dirs)):
            a, b = load_run(ne_dirs[s]), load_run(other_dirs[s])
            if a is None or b is None:
                continue
            base[s], ts[s] = label_fixed(a, b), label_fixed(a, b, ts=True)
            cover[s] = base[s][2]
        if not base:
            continue
        cell = {k: _ci({s: v[0][k] for s, v in base.items()}, {s: v[1][k] for s, v in base.items()}) for k in ('ece', 'nll', 'brier', 'conf')}
        cell_ts = {k: _ci({s: v[0][k] for s, v in ts.items()}, {s: v[1][k] for s, v in ts.items()}) for k in ('ece', 'nll')}
        print(f'| {lab} | {statistics.mean(cover.values()):.3f} | {_fmt_ci(cell["ece"])} | {_fmt_ci(cell["nll"])} | {_fmt_ci(cell["brier"])} | '
              f'{_fmt_ci(cell["conf"])} | {_fmt_ci(cell_ts["ece"])} | {_fmt_ci(cell_ts["nll"])} |', flush=True)
        out['label_fixed'][lab] = {'agree': statistics.mean(cover.values()), 'base': cell, 'ts': cell_ts}

    am = accuracy_matched(ctx)
    out['accuracy_matched'] = am
    print(f'\nAccuracy-matched from the existing grid (tolerance {100 * ACC_TOL:.2f} points on val accuracy):', flush=True)
    if am is None:
        print('  no grid cells on disk', flush=True)
    elif 'delta' not in am:
        print(f'  {am}', flush=True)
    else:
        print(f'  config (onset, λ)={am["config"]}, {am["n_within_tol"]}/{am["n_cells"]} cells within tolerance, val acc {am["matched_val_acc"]:.4f} vs CE {am["ce_val_acc"]:.4f}, seeds {am["seeds"]}', flush=True)
        for k, ci in am['delta'].items():
            print(f'  Δ{k:7s} {_fmt_ci(ci)}', flush=True)

    print(f'\n{REF} minus each method, paired over seeds; verdict from the 95% t interval:\n', flush=True)
    print('| against | n | ΔNLL | verdict | ΔNLL after TS | verdict | ΔECE after TS | verdict |\n|---|---|---|---|---|---|---|---|', flush=True)
    out['paired'] = {}
    for lab, v in per.items():
        if lab == REF:
            continue
        row = {k: _ci({s: m[k] for s, m in per[REF].items()}, {s: m[k] for s, m in v.items()}) for k in ('nll', 'nll_ts', 'ece_ts')}
        print(f'| {lab} | {row["nll"]["n"]} | {_fmt_ci(row["nll"])} | {_verdict(row["nll"])} | {_fmt_ci(row["nll_ts"])} | {_verdict(row["nll_ts"])} | '
              f'{_fmt_ci(row["ece_ts"])} | {_verdict(row["ece_ts"])} |', flush=True)
        out['paired'][lab] = {k: {**ci, 'verdict': _verdict(ci)} for k, ci in row.items()}
    best = min((lab for lab in per if lab != REF), key=lambda lab: statistics.mean(m['nll_ts'] for m in per[lab].values()))
    out['runner_up_nll_ts'] = {'label': best, **out['paired'][best]['nll_ts']}
    print(f'\nAfter TS, runner-up is {best}: NeSCal minus it {_fmt_ci(out["paired"][best]["nll_ts"])} -> {out["paired"][best]["nll_ts"]["verdict"]}', flush=True)
    try:
        with open(os.path.join(root, 'additional_analysis.json'), 'w') as f:
            json.dump(out, f, indent=2, default=str)
    except OSError as e:
        print(f'  could not write additional_analysis.json: {e}', flush=True)
    return out


def stage_analysis(names):
    roots = {**{n: rnc.ROOTS[n] for n in names}, **ANALYSIS_EXTRA_ROOTS}
    results = {}
    for n, r in roots.items():
        try:
            if os.path.isdir(r):
                results[n] = analyse_setting(n, r)
            else:
                print(f'[analysis] {n}: {r} not found, skipped', flush=True)
        except Exception as e:
            ERRORS.append({'label': f'analysis:{n}', 'error': f'{type(e).__name__}: {e}'})
            traceback.print_exc()
    print(f'\n{"=" * 100}\n## Across settings: {REF} against the runner-up after TS\n', flush=True)
    print('| setting | runner-up | ΔNLL after TS | verdict |\n|---|---|---|---|', flush=True)
    for n, res in results.items():
        if res:
            u = res['runner_up_nll_ts']
            print(f'| {n} | {u["label"]} | {_fmt_ci(u)} | {u["verdict"]} |', flush=True)
    print('\nWith n=4 the t interval is wide (t=3.18), so "tie" means not separated at these seeds, not equal.', flush=True)


def tag_dir(ctx):
    return os.path.join(ctx['cfg']['base_out'], ctx['ds'], ctx['cfg']['arch_name'], rnc._out_tag())


def save_arm(ctx, label, entry):
    p = os.path.join(tag_dir(ctx), 'additional_manifest.json')
    os.makedirs(tag_dir(ctx), exist_ok=True)
    mf = load_json(p) if os.path.exists(p) else {'arms': {}}
    mf['arms'][label] = entry
    write_manifest(p, mf)


def search_arm(ctx, arm, grid, make_lp, onset, extra):
    method = lambda v: f'{arm}_v{rnc._tag(v)}'
    for v in grid:
        for s in rnc.SEL_SEEDS:
            rnc.train(ctx, s, method(v), make_lp(v), onset, f'[{ctx["name"]}] {arm} v={v} seed={s}', **extra(v))
    best, table = rnc.select_finished(ctx, grid, method, rnc.SEL_SEEDS)
    if len(grid) > 1 and best in (grid[0], grid[-1]):
        print(f'  NOTE: {arm} winner v={best} is on the grid edge {grid[0]}..{grid[-1]}', flush=True)
    rep = {str(s): rnc.train(ctx, s, method(best), make_lp(best), onset, f'[{ctx["name"]}] {arm} v={best} seed={s}', **extra(best))
           for s in rnc.SEL_SEEDS + rnc.CONF_SEEDS}
    entry = {'winner': best, 'selection': {str(k): v for k, v in table.items()}, 'reportable': {s: r for s, r in rep.items() if r}}
    save_arm(ctx, f'{arm}_FINAL', entry)
    print(f'  -> {arm}: winner v={best} ({ {k: round(v, 4) for k, v in table.items()} })', flush=True)
    return entry


def arm_table(ctx, title, arms):
    cols = ['nll', 'nll_ts', 'ece', 'ece_ts', 'brier', 'acc']
    entries = {'XE_baseline': rnc.method_dirs(ctx, 'XE_baseline'), **arms, REF: rnc.method_dirs(ctx, REF)}
    rows = {}
    for lab, dirs in entries.items():
        vals = [m for m in (run_metrics(d, stale_check=False) for d in dirs.values()) if m]
        if vals:
            rows[lab] = {c: [m[c] for m in vals] for c in cols}
    print(f'\n### {ctx["ds"]} / {ctx["cfg"]["arch_name"]}: {title}\n', flush=True)
    rnc.md_table(rows, cols, maximize=('acc',))
    ref = {s: run_metrics(d, stale_check=False) for s, d in entries[REF].items()}
    print(f'\nArm minus {REF}, paired over seeds (95% t interval):', flush=True)
    for lab, dirs in arms.items():
        m = {int(s): run_metrics(d, stale_check=False) for s, d in dirs.items()}
        shown = []
        for c in ('nll', 'nll_ts', 'ece_ts'):
            ci = _ci({s: v[c] for s, v in m.items() if v}, {s: v[c] for s, v in ref.items() if v})
            shown.append(f'{c} {_fmt_ci(ci)}')
        print(f'  {lab:32s} ' + ' | '.join(shown), flush=True)


def variant_params(lam, **secondary):
    return {'primary_loss': 'XE', 'secondary_loss': 'nbr_variant', 'lambda_reg': lam,
            'secondary_params': {'k': 10, 'divergence_type': 'js', **secondary}}


COSINE = {'lambda_kind': 'cosine', 'lambda_decay': 'cosine'}


def stage_factorial(ctx):
    print(f'\n########## [{rnc._now()}] FACTORIAL {ctx["name"]} | onset={ctx["onset"]} λ grids={FACTORIAL_LAMS} ##########', flush=True)
    arms = {}
    for target, graph in FACTORIAL_CELLS:
        arm = f'fact_{target}_{graph}'
        grid = FACTORIAL_LAMS[graph][:1] if rnc.SMOKE else FACTORIAL_LAMS[graph]
        res = search_arm(ctx, arm, grid, lambda lam, t=target, g=graph: variant_params(lam, target=t, graph=g), ctx['onset'], lambda v: dict(COSINE))
        arms[f'{arm}_FINAL'] = res['reportable']
    arm_table(ctx, f'target x graph factorial; {REF} is the (edge, frozen) cell', arms)


def stage_divergences(ctx):
    grid = PENALTY_TARGETS[2:3] if rnc.SMOKE else PENALTY_TARGETS
    print(f'\n########## [{rnc._now()}] DIVERGENCES {ctx["name"]} | onset={ctx["onset"]} penalty targets={grid} ##########', flush=True)
    arms = {}
    for d in DIVERGENCES:
        res = search_arm(ctx, f'div_{d}', grid, lambda t, dd=d: variant_params(1.0, divergence_type=dd), ctx['onset'],
                         lambda t: {**COSINE, 'lambda_from_penalty_target': t})
        arms[f'div_{d}_FINAL'] = res['reportable']
    arm_table(ctx, 'pairwise KL, symmetric KL, centred logit pairing (each tuned over the penalty-target grid)', arms)


def stage_scale_arms(ctx, arm_names, seeds):
    print(f'\n########## [{rnc._now()}] SCALE ARMS {ctx["name"]} | {arm_names} | headline onset={ctx["onset"]} λ={ctx["lam"]} seeds={seeds} ##########', flush=True)
    arms = {}
    for arm in arm_names:
        spec, rep = SCALE_ARMS[arm], {}
        for s in seeds:
            root = rnc.run_path(ctx, arm, s)
            os.makedirs(root, exist_ok=True)
            if 'graph_recompute_every' in spec:
                lp, extra = rn._ncl_params(ctx['cfg'], ctx['lam']), {**COSINE, 'graph_recompute_every': spec['graph_recompute_every']}
            else:
                lp = variant_params(ctx['lam'], stats_path=os.path.join(root, 'ann_stats.json'), drift_path=os.path.join(root, 'drift.json'), **spec)
                extra = dict(COSINE)
            r = rnc.train(ctx, s, arm, lp, ctx['onset'], f'[{ctx["name"]}] {arm} seed={s}', **extra)
            if r:
                rep[str(s)] = r
        save_arm(ctx, f'{arm}_FINAL', {'winner': None, 'reportable': rep})
        arms[f'{arm}_FINAL'] = rep
        recalls = [load_json(os.path.join(r, 'ann_stats.json')) for r in rep.values() if os.path.exists(os.path.join(r, 'ann_stats.json'))]
        if recalls:
            print(f'  {arm}: recall@10 {_pm([x["recall_at_k"] for x in recalls], 3)}, build+search {_pm([x["build_s"] + x["search_s"] for x in recalls], 1)} s', flush=True)
        drifts = [load_json(os.path.join(r, 'drift.json')) for r in rep.values() if os.path.exists(os.path.join(r, 'drift.json'))]
        if drifts:
            mean_of = lambda key: _pm([statistics.mean(e[key] for e in h) for h in drifts if h], 4)
            print(f'  {arm}: mean over epochs, sqrt JS(p_i || cached p_j) {mean_of("sqrt_js_edge_to_cache")}, sqrt JS(cached p_j || current p_j) {mean_of("sqrt_js_cache_drift")}', flush=True)
    arm_table(ctx, 'ANN, graph-rebuild and target-cache arms at the headline configuration', arms)


def stage_mixup(ctx):
    p = os.path.join(ctx['root'], 'mixup_manifest.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} missing: run the September Mixup stage first, or set the alpha by hand here')
    mm = load_json(p)
    a = float(mm['winners']['mixup']['candidate']['alpha'])
    grid = [m * ctx['lam'] for m in (MIXUP_LAM_MULT[1:2] if rnc.SMOKE else MIXUP_LAM_MULT)]
    print(f'\n########## [{rnc._now()}] NeSCal+Mixup {ctx["name"]} | alpha={a} onset={ctx["onset"]} λ={grid} ##########', flush=True)
    res = search_arm(ctx, 'nescal_mixup', grid, lambda lam: rn._ncl_params(ctx['cfg'], lam), ctx['onset'],
                     lambda lam: {**COSINE, 'mixup_alpha': a, 'fresh': True})
    arm_table(ctx, f'Mixup alone and NeSCal+Mixup (alpha={a}), against CE and NeSCal',
              {'mixup_FINAL': rnc.method_dirs(ctx, 'mixup_FINAL'), 'nescal_mixup_FINAL': res['reportable']})


def stage_rankmixup(ctx):
    grid = RANKMIXUP_ALPHAS[:1] if rnc.SMOKE else RANKMIXUP_ALPHAS
    print(f'\n########## [{rnc._now()}] RankMixup {ctx["name"]} | alpha={grid} m={RANKMIXUP_M} w={RANKMIXUP_W} ##########', flush=True)
    make = lambda a: {'primary_loss': 'XE', 'secondary_loss': 'rank_mixup', 'lambda_reg': RANKMIXUP_W,
                      'secondary_params': {'alpha': a, 'margin': RANKMIXUP_M}}
    res = search_arm(ctx, 'rankmixup', grid, make, 0, lambda a: {'fresh': True})
    arm_table(ctx, 'Mixup and RankMixup, against CE and NeSCal',
              {'mixup_FINAL': rnc.method_dirs(ctx, 'mixup_FINAL'), 'rankmixup_FINAL': res['reportable']})


def stage_mixup_base(ctx):
    p = os.path.join(ctx['root'], 'mixup_manifest.json')
    if os.path.exists(p) and not rnc.SMOKE:
        print(f'[skip] mixup_base {ctx["name"]}: {p} already exists', flush=True)
        return
    grid = MIXUP_BASE_ALPHAS[:1] if rnc.SMOKE else MIXUP_BASE_ALPHAS
    print(f'\n########## [{rnc._now()}] MIXUP BASELINE {ctx["name"]} | alpha={grid} ##########', flush=True)
    res = search_arm(ctx, 'mixup_base', grid, lambda a: {'primary_loss': 'XE', 'secondary_loss': None}, 0,
                     lambda a: {'mixup_alpha': a, 'fresh': True})
    mf = {'timestamp': rnc._out_tag(), 'reportable': {'mixup_FINAL': res['reportable']},
          'winners': {'mixup': {'candidate': {'alpha': res['winner']}, 'selection': res['selection']}}}
    write_manifest(os.path.join(tag_dir(ctx), 'mixup_manifest.json') if rnc.SMOKE else p, mf)
    arm_table(ctx, f'Mixup (alpha={res["winner"]}) against CE and NeSCal', {'mixup_FINAL': res['reportable']})


def _gpu_peak(fn, label):
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t = time.time()
    try:
        fn()
    except torch.cuda.OutOfMemoryError as e:
        print(f'  {label}: GPU OUT OF MEMORY ({str(e)[:80]})', flush=True)
        return {'label': label, 'oom': True}
    out = {'label': label, 'seconds': time.time() - t}
    if on_gpu:
        torch.cuda.synchronize()
        out.update({'peak_above_base_gb': (torch.cuda.max_memory_allocated() - base) / 1e9, 'base_gb': base / 1e9})
    print(f'  {out}', flush=True)
    return out


def legacy_diag(gp, nb):
    with torch.no_grad():
        pj = gp[nb]
        pi = gp.unsqueeze(1).expand_as(pj)
        pi_flat, pj_flat = pi.reshape(-1, pi.shape[-1]), pj.reshape(-1, pj.shape[-1])
        return js_divergence_torch(pi_flat, pj_flat).mean().item()


def chunked_diag(gp, nb, chunk=4096):
    with torch.no_grad():
        total, n = 0.0, 0
        for s in range(0, nb.shape[0], chunk):
            nb_c = nb[s:s + chunk]
            pi_flat = gp[s:s + chunk].repeat_interleave(nb_c.shape[1], dim=0)
            total += float(js_divergence_torch(pi_flat, gp[nb_c].reshape(-1, gp.shape[1])).sum().item())
            n += pi_flat.shape[0]
        return total / n


def stage_scale_bench():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = rnc.context('tin', TIN_ROOT)
    xe = rnc.method_dirs(ctx, 'XE_baseline')[rnc.SEL_SEEDS[0]]
    print(f'\n########## [{rnc._now()}] SCALE BENCH | Tiny-ImageNet XE weights from {xe} | device={dev} ##########', flush=True)
    m = Trainer.load_model(xe)
    loaders = rnc._loaders(ctx['cfg'], rnc.SEL_SEEDS[0])
    t0 = time.time()
    Z, P, _ = _embed_probs_labels(m.model, loaders[3], m.device, 'final')
    extract_s = time.time() - t0
    N, D, C, k = Z.shape[0], Z.shape[1], P.shape[1], 10
    print(f'  N={N} D={D} C={C}: one embedding pass {extract_s:.1f} s', flush=True)
    del m
    rnc._free()
    res = {'tin': {'N': N, 'D': D, 'C': C, 'extract_s': extract_s}, 'imagenet': {}}
    loss = PreNeighborhoodConsistencyLoss(k=k)
    res['tin']['exact'] = _gpu_peak(lambda: loss.precompute_matrices(torch.tensor(Z, device=dev)), 'exact chunked kNN search')
    exact_I = loss.neighbors.numpy()
    rnc._free()
    diag = {'gp': torch.tensor(P, device=dev), 'nb': loss.neighbors.to(dev)}
    res['tin']['diag_legacy'] = _gpu_peak(lambda: diag.update(legacy=legacy_diag(diag['gp'], diag['nb'])), 'activation diagnostic, as submitted')
    res['tin']['diag_chunked'] = _gpu_peak(lambda: diag.update(chunked=chunked_diag(diag['gp'], diag['nb'])), 'activation diagnostic, chunked')
    res['tin']['diag_values'] = {'legacy': diag.get('legacy'), 'chunked': diag.get('chunked')}
    diag.clear()
    rnc._free()
    x = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)
    x = x.astype('float32')
    for be in ('faiss_ivf', 'faiss_hnsw'):
        I, st = ann_knn(x, k, be)
        st['recall_vs_chunked_exact'] = float(np.mean([len(set(I[i].tolist()) & set(exact_I[i].tolist())) / k
                                                       for i in np.random.RandomState(0).choice(N, min(5000, N), replace=False)]))
        st['recall_at_k'] = recall_at_k(x, I, k)
        res['tin'][be] = st
        print(f'  {st}', flush=True)

    legacy_gb = res['tin']['diag_legacy'].get('peak_above_base_gb')
    per_unit = legacy_gb / (N * k * C * 4 / 1e9) if legacy_gb else None
    res['imagenet'] = {'n': IMAGENET_N, 'classes': IMAGENET_K, 'prediction_cache_gb_float32': IMAGENET_N * IMAGENET_K * 4 / 1e9,
                       'edge_list_gb': IMAGENET_N * k * 8 / 1e9, 'legacy_diag_tensor_gb': IMAGENET_N * k * IMAGENET_K * 4 / 1e9,
                       'legacy_diag_peak_gb_extrapolated': (per_unit * IMAGENET_N * k * IMAGENET_K * 4 / 1e9) if per_unit else None,
                       'tin_legacy_peak_over_tensor': per_unit}
    frozen = [p for p in sorted(glob.glob('./data/frozen/imagenet_*/base.pt'))]
    if frozen:
        d = torch.load(frozen[0], map_location='cpu')
        emb = d['trainval_emb'].float().numpy()
        res['imagenet'].update({'features': frozen[0], 'n_actual': emb.shape[0], 'd': emb.shape[1]})
        print(f'  ImageNet features: REAL frozen embeddings {frozen[0]} {emb.shape}', flush=True)
    else:
        emb = np.random.default_rng(0).standard_normal((IMAGENET_N, IMAGENET_D), dtype=np.float32)
        res['imagenet'].update({'features': 'synthetic Gaussian (timing only; recall on these is meaningless)', 'd': IMAGENET_D})
        print(f'  ImageNet features: no frozen embeddings found; SYNTHETIC {emb.shape}, timing only', flush=True)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    n_full = emb.shape[0]
    try:
        zf = torch.tensor(emb, device=dev)
        chunk, n_chunks = 512, 20
        t = time.time()
        for i in range(n_chunks):
            torch.topk(torch.mm(zf[i * chunk:(i + 1) * chunk], zf.T), k + 1, dim=1)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        res['imagenet']['exact_search_h_extrapolated'] = (time.time() - t) / n_chunks * math.ceil(n_full / chunk) / 3600
        res['imagenet']['exact_chunk'] = chunk
        del zf
        rnc._free()
    except torch.cuda.OutOfMemoryError:
        res['imagenet']['exact_search_h_extrapolated'] = 'OOM holding the features on this GPU'
        rnc._free()
    _, st = ann_knn(emb, k, 'faiss_ivf', query_sample=QUERY_SAMPLE)
    res['imagenet']['faiss_ivf'] = st
    print(f'  {res["imagenet"]}', flush=True)
    os.makedirs(tag_dir(ctx), exist_ok=True)
    with open(os.path.join(tag_dir(ctx), 'scale_bench.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f'[scale] wrote {os.path.join(tag_dir(ctx), "scale_bench.json")}', flush=True)


GPU_STAGES = ('factorial', 'divergences', 'scale_arms', 'scale_bench', 'mixup_base', 'mixup', 'rankmixup')


def preflight(stages, jobs):
    problems = []
    gpu = [s for s in stages if s in GPU_STAGES]
    if gpu:
        if 'scale_arms' in stages or 'scale_bench' in stages:
            if importlib.util.find_spec('faiss') is None:
                problems.append('faiss not importable; pip install faiss-cpu')
    ctxs = {}
    needed = set()
    for st, js in (('factorial', FACTORIAL_JOBS), ('divergences', DIVERGENCE_JOBS), ('scale_arms', SCALE_JOBS), ('mixup_base', MIXUP_JOBS), ('mixup', MIXUP_JOBS), ('rankmixup', RANKMIXUP_JOBS)):
        if st in stages:
            needed |= set(js)
    for n in sorted(needed & set(jobs)):
        try:
            ctxs[n] = rnc.context(n)
            c = ctxs[n]
            print(f'[preflight] {n}: {c["root"]} onset={c["onset"]} λ={c["lam"]} epochs={c["cfg"]["num_epochs"]} bs={c["cfg"]["batch_size"]} '
                  f'milestones={c["cfg"].get("lr_milestones_frac")} λ-grid={c["cfg"].get("lambda_grid_cosine")}', flush=True)
            for s in rnc.SEL_SEEDS + rnc.CONF_SEEDS:
                p, e = rnc.ckpt_for(c, s, c['onset'])
                if p is None or e != c['onset']:
                    problems.append(f'{n}: seed {s} has no XE checkpoint exactly at onset {c["onset"]} (latest usable: {e})')
            if 'mixup' in stages and 'mixup_base' not in stages and n in MIXUP_JOBS and not os.path.exists(os.path.join(c['root'], 'mixup_manifest.json')):
                problems.append(f'{n}: mixup_manifest.json missing')
        except Exception as e:
            problems.append(f'{n}: {type(e).__name__}: {e}')
    if ('scale_bench' in stages or ('scale_arms' in stages and TIN_SCALE_ARMS)) and not os.path.isdir(TIN_ROOT):
        problems.append(f'TIN_ROOT {TIN_ROOT} not found')
    print('[preflight] ' + ('PROBLEMS:\n  ' + '\n  '.join(problems) if problems else 'all checks passed'), flush=True)
    return ctxs, problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', nargs='+', default=['analysis'], choices=['analysis', *GPU_STAGES])
    ap.add_argument('--jobs', nargs='+', default=list(rnc.ROOTS), choices=list(rnc.ROOTS))
    ap.add_argument('--preflight_only', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='1 grid value, 1+1 seeds, 2 epochs after onset, under <OUT_TAG>_smoke')
    a = ap.parse_args(argv)
    rnc.SMOKE = a.smoke
    if a.smoke:
        rnc.SEL_SEEDS, rnc.CONF_SEEDS = rnc.SEL_SEEDS[:1], rnc.CONF_SEEDS[:1]
    ctxs, problems = preflight(a.stages, a.jobs)
    if problems and not (a.stages == ['analysis']):
        return 1
    if a.preflight_only:
        return 0
    rnc.TRACKER = RunTracker(10 ** 7, label='additional')

    def guarded(label, fn, *args):
        try:
            fn(*args)
        except SystemExit:
            raise
        except Exception as e:
            ERRORS.append({'label': label, 'error': f'{type(e).__name__}: {e}'})
            print(f'!! {label} FAILED: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
    t0 = time.time()
    try:
        for st in a.stages:
            if st == 'analysis':
                guarded('analysis', stage_analysis, a.jobs)
            elif st == 'scale_bench':
                guarded('scale_bench', stage_scale_bench)
            elif st == 'factorial':
                for n in FACTORIAL_JOBS:
                    if n in ctxs:
                        guarded(f'factorial:{n}', stage_factorial, ctxs[n])
            elif st == 'divergences':
                for n in DIVERGENCE_JOBS:
                    if n in ctxs:
                        guarded(f'divergences:{n}', stage_divergences, ctxs[n])
            elif st == 'scale_arms':
                for n in SCALE_JOBS:
                    if n in ctxs:
                        guarded(f'scale_arms:{n}', stage_scale_arms, ctxs[n], list(SCALE_ARMS), rnc.SEL_SEEDS + rnc.CONF_SEEDS)
                if TIN_SCALE_ARMS:
                    guarded('scale_arms:tin', stage_scale_arms, rnc.context('tin', TIN_ROOT), TIN_SCALE_ARMS, TIN_SEEDS[:1] if a.smoke else TIN_SEEDS)
            elif st == 'mixup_base':
                for n in MIXUP_JOBS:
                    if n in ctxs:
                        guarded(f'mixup_base:{n}', stage_mixup_base, ctxs[n])
            elif st == 'mixup':
                for n in MIXUP_JOBS:
                    if n in ctxs:
                        guarded(f'mixup:{n}', stage_mixup, ctxs[n])
            elif st == 'rankmixup':
                for n in RANKMIXUP_JOBS:
                    if n in ctxs:
                        guarded(f'rankmixup:{n}', stage_rankmixup, ctxs[n])
    except SystemExit as e:
        print(f'!! {e}', flush=True)
        return 2
    print(f'\n=== additional {a.stages} done in {(time.time() - t0) / 3600:.2f} h | errors: {ERRORS or rnc.ERRORS or "none"} ===', flush=True)
    return 0
