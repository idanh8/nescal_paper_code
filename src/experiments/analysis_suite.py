import os
import re
import sys
import glob
import json
import pickle
import argparse
import warnings
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common import load_json, load_pkl

MANIFESTS = ('run_manifest.json', 'competitors_manifest.json')
NICE = {'XE_baseline': 'CE', 'FL_FINAL': 'FL', 'ls_alone_FINAL': 'LS', 'mmce_FINAL': 'MMCE',
        'mdca_FINAL': 'MDCA', 's_avuc_FINAL': 's-AvUC', 'mbls_FINAL': 'MbLS',
        'ncl_FINAL_cosine': 'NeSCal', 'ncl_FINAL_constant': 'NeSCal (const)'}
ORDER = list(NICE)
NLLSEL_LABEL = {'FL': 'FL_FINAL', 'LS': 'ls_alone_FINAL'}


def _resolve(root, p):
    if not isinstance(p, str) or not p:
        return None
    p = os.path.expanduser(p)
    parts = p.replace('\\', '/').strip('/').split('/')
    candidates = [p, os.path.join(root, p)]
    candidates += [os.path.join(root, *parts[i:]) for i in range(len(parts))]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.realpath(candidate)
    return None


def _manifests(root):
    for mf in MANIFESTS:
        fp = os.path.join(root, mf)
        if os.path.isfile(fp):
            yield load_json(fp)


def _put(per, seed, path, context):
    seed = int(seed)
    if seed in per and per[seed] != path:
        warnings.warn(f'{context}, seed {seed}: keeping {per[seed]}; ignoring {path}')
    else:
        per[seed] = path


def reportable(root, extra=()):
    out = defaultdict(dict)
    for base in (root, *extra):
        for manifest in _manifests(base):
            for lab, per in manifest.get('reportable', {}).items():

                for seed, path in per.items():
                    rp = _resolve(base, path)
                    if rp:
                        _put(out[lab], seed, rp, lab)
                    else:
                        warnings.warn(f'Unresolved reportable path in {base}: {path}')
    for base in (root, *extra):
        for lab, per in _nllsel_overrides(base).items():
            for seed, rp in per.items():
                if out[lab].get(seed) != rp:
                    print(f'[nllsel] {NICE.get(lab, lab)} seed {seed}: {rp}')
                out[lab][seed] = rp
    return dict(out)


def _setting(root):
    identities = set()
    for manifest in _manifests(root):
        cfg = manifest.get('config', {})
        if cfg.get('dataset') and cfg.get('arch_name'):
            identities.add((cfg['dataset'], cfg['arch_name']))
    if len(identities) > 1:
        raise ValueError(f'Conflicting dataset/architecture metadata in {root}')
    if identities:
        return identities.pop()

    parent = os.path.dirname(os.path.abspath(root))
    return os.path.basename(os.path.dirname(parent)), os.path.basename(parent)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _grid_dirs(root, cfg, extra=()):
    seen = set()
    for base in (root, *extra):
        bases = [base]
        manifests = list(_manifests(base))
        for manifest in manifests:
            local_cfg = manifest.get('config', {})
            if all(k in local_cfg for k in ('base_out', 'dataset', 'arch_name')):
                bases.append(os.path.join(local_cfg['base_out'], local_cfg['dataset'],
                                          local_cfg['arch_name'], os.path.basename(base)))
            for value in _strings(manifest):
                rp = _resolve(base, value)
                if rp:
                    if re.fullmatch(r'seed_\d+', os.path.basename(rp)):
                        bases.extend([os.path.dirname(rp), os.path.dirname(os.path.dirname(rp))])
                    else:
                        bases.append(rp)
        for directory in dict.fromkeys(bases):
            for pattern in ('seed_*', os.path.join('*', 'seed_*')):
                for d in sorted(glob.glob(os.path.join(directory, pattern))):
                    d = os.path.realpath(d)
                    if d not in seen and os.path.isdir(d) and re.fullmatch(r'seed_\d+', os.path.basename(d)):
                        seen.add(d)
                        yield d


def _ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)


def _auroc(conf, correct):
    order = np.argsort(conf)
    ranks = np.empty(len(conf), float)
    ranks[order] = np.arange(1, len(conf) + 1)
    c = conf[order]
    i = 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    pos, neg = correct.sum(), (~correct).sum()
    if pos == 0 or neg == 0:
        return None
    return float((ranks[correct].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _aurc(conf, correct):
    order = np.argsort(-conf)
    err = (~correct[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def discrimination(root, out_dir, extra=()):
    rows = {}
    for lab, per in reportable(root, extra).items():
        pre_auroc, pre_aurc, ts_auroc, ts_aurc = [], [], [], []
        for s, rp in sorted(per.items()):
            td = os.path.join(rp, 'test_data.pkl')
            if not os.path.exists(td):
                continue
            t = load_pkl(td)
            p, y = np.asarray(t['test_conf_vectors']), np.asarray(t['test_labels'])
            correct = p.argmax(1) == y
            pre_auroc.append(_auroc(p.max(1), correct))
            pre_aurc.append(_aurc(p.max(1), correct))
            ph = os.path.join(rp, 'posthoc_results.pkl')
            T = None
            if os.path.exists(ph):
                T = load_pkl(ph).get('params', {}).get('ts', {}).get('temperature')
            if T:
                z = np.asarray(t['test_logits'], dtype=np.float64) / float(T)
                z = z - z.max(1, keepdims=True)
                q = np.exp(z)
                q /= q.sum(1, keepdims=True)
                ts_auroc.append(_auroc(q.max(1), q.argmax(1) == y))
                ts_aurc.append(_aurc(q.max(1), q.argmax(1) == y))
        if pre_auroc:
            rows[lab] = {'auroc': _ms(pre_auroc), 'aurc': _ms(pre_aurc),
                         'auroc_ts': _ms(ts_auroc), 'aurc_ts': _ms(ts_aurc), 'n': len(pre_auroc)}
    print('\n| method | AUROC | AUROC+TS | AURC (x1000) | AURC+TS (x1000) | n |\n|---|---|---|---|---|---|')
    for lab in sorted(rows, key=lambda x: (ORDER.index(x) if x in ORDER else 99, x)):
        r = rows[lab]
        f = lambda t, k=1: '--' if t[0] is None else f'{k * t[0]:.4f} ± {k * t[1]:.4f}'
        print(f'| {NICE.get(lab, lab)} | {f(r["auroc"])} | {f(r["auroc_ts"])} | {f(r["aurc"], 1000)} | '
              f'{f(r["aurc_ts"], 1000)} | {r["n"]} |')
    json.dump(rows, open(os.path.join(out_dir, 'discrimination.json'), 'w'), indent=1)


GRID = re.compile(
    r'^ncl_(J?_?cos.*?)onset(\d+)_lam([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$'
)


def nescal_grid(root, cfg, extra=()):
    out = defaultdict(dict)
    for d in _grid_dirs(root, cfg, extra):
        match = GRID.fullmatch(os.path.basename(os.path.dirname(d)))
        if match:
            variant, onset, lam = match.groups()
            _put(out[(variant, int(onset), float(lam))], os.path.basename(d)[5:], d, 'NeSCal grid')
    return dict(out)


def competitor_grid(root, cfg, extra=()):
    out = defaultdict(lambda: defaultdict(dict))
    for d in _grid_dirs(root, cfg, extra):
        tag = os.path.basename(os.path.dirname(d))
        if 'FINAL' in tag:
            continue
        method = next((m for m in ('ls_alone', 's_avuc', 'mmce', 'mdca', 'mbls', 'FL')
                       if tag == m or tag.startswith(m + '_')), None)
        if method:
            _put(out[method][tag], os.path.basename(d)[5:], d, tag)
    return {k: dict(v) for k, v in out.items()}


def _val(rp, key='nll'):
    fp = os.path.join(rp, 'val_results.pkl')
    if not os.path.exists(fp):
        return None
    return load_pkl(fp).get(key)


def _test(rp, key):
    fp = os.path.join(rp, 'test_results.pkl')
    if not os.path.exists(fp):
        return None
    return load_pkl(fp).get(key)


def frontier(root, cfg, out_dir, name, extra=()):
    grid = nescal_grid(root, cfg, extra)
    manifest_paths = [os.path.join(r, 'run_manifest.json') for r in (root, *extra)]
    mf = next((load_json(fp) for fp in manifest_paths if os.path.isfile(fp)), {})
    onset, lam_sel = mf.get('best_onset'), mf.get('lambda_cosine')
    pts = []
    for (variant, e, lam), per in sorted(grid.items()):
        if 'cos' not in variant:
            continue
        if e != onset:
            continue
        acc = _ms([_test(rp, 'accuracy') for rp in per.values()])[0]
        nll = _ms([_test(rp, 'nll') for rp in per.values()])[0]
        ece = _ms([_test(rp, 'ece') for rp in per.values()])[0]
        if acc is not None:
            pts.append((lam, acc, nll, ece))
    if not pts:
        print(f'[frontier] {name}: no grid runs at onset {onset}')
        return
    pts.sort()
    others = {}
    for lab, per in reportable(root, extra).items():
        if lab.startswith('ncl_'):
            continue
        others[lab] = (_ms([_test(rp, 'accuracy') for rp in per.values()])[0],
                       _ms([_test(rp, 'nll') for rp in per.values()])[0],
                       _ms([_test(rp, 'ece') for rp in per.values()])[0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, idx, ylab in ((axes[0], 2, 'test NLL'), (axes[1], 3, 'test ECE')):
        ax.plot([p[1] for p in pts], [p[idx] for p in pts], '-o', ms=4, label=r'NeSCal $\lambda$-frontier')
        for lam, acc, nll, ece in pts:
            ax.annotate(f'{lam:g}', (acc, (nll, ece)[idx - 2]), fontsize=6,
                        textcoords='offset points', xytext=(3, 3))
            if lam == lam_sel:
                ax.plot([acc], [(nll, ece)[idx - 2]], marker='*', ms=14, mfc='none', mec='k')
        for lab, (acc, nll, ece) in others.items():
            if acc is None:
                continue
            ax.scatter([acc], [(nll, ece)[idx - 2]], marker='s', s=18)
            ax.annotate(NICE.get(lab, lab), (acc, (nll, ece)[idx - 2]), fontsize=6,
                        textcoords='offset points', xytext=(3, -6))
        ax.set_xlabel('test accuracy')
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{name}: NeSCal $\\lambda$-frontier at onset {onset} (star = selected)', fontsize=9)
    fig.tight_layout()
    fp = os.path.join(out_dir, f'frontier_{name.replace("/", "_")}.pdf')
    fig.savefig(fp)
    plt.close(fig)
    print(f'[frontier] {name}: {len(pts)} lambda points -> {fp}')
    json.dump({'onset': onset, 'selected_lambda': lam_sel,
               'points': [{'lambda': p[0], 'accuracy': p[1], 'nll': p[2], 'ece': p[3]} for p in pts],
               'methods': {NICE.get(k, k): v for k, v in others.items()}},
              open(os.path.join(out_dir, f'frontier_{name.replace("/", "_")}.json'), 'w'), indent=1)


def selection(root, cfg, out_dir, name, extra=()):
    cands = {'NeSCal': {f'{variant}onset{e}_lam{lam:g}': per for (variant, e, lam), per in nescal_grid(root, cfg, extra).items()}}
    for m, tags in competitor_grid(root, cfg, extra).items():
        cands[NICE.get(m + '_FINAL', m)] = tags
    rows = {}
    for method, tags in cands.items():
        scored = {}
        for tag, per in tags.items():
            nll = _ms([_val(rp, 'nll') for rp in per.values()])[0]
            ece = _ms([_val(rp, 'ece') for rp in per.values()])[0]
            if nll is not None and ece is not None:
                scored[tag] = (nll, ece, per)
        if len(scored) < 2:
            continue
        by_nll = min(scored, key=lambda t: scored[t][0])
        by_ece = min(scored, key=lambda t: scored[t][1])
        rows[method] = {'n_configs': len(scored), 'sel_nll': by_nll, 'sel_ece': by_ece,
                        'same': by_nll == by_ece}
        for which, tag in (('nll', by_nll), ('ece', by_ece)):
            per = scored[tag][2]
            for k in ('nll', 'ece', 'accuracy'):
                rows[method][f'{which}_test_{k}'] = _ms([_test(rp, k) for rp in per.values()])[0]
    print(f'\n### {name}: configuration selected by validation NLL vs validation ECE\n')
    print('| method | configs | selected by val NLL: test NLL / ECE | selected by val ECE: test NLL / ECE | same? |')
    print('|---|---|---|---|---|')
    for m, r in rows.items():
        f = lambda a, b: f'{a:.4f} / {100 * b:.2f}%' if a is not None and b is not None else '--'
        print(f'| {m} | {r["n_configs"]} | {f(r["nll_test_nll"], r["nll_test_ece"])} | '
              f'{f(r["ece_test_nll"], r["ece_test_ece"])} | {"yes" if r["same"] else "no"} |')
    json.dump(rows, open(os.path.join(out_dir, f'selection_{name.replace("/", "_")}.json'), 'w'), indent=1)


def evp(root, cfg, out_dir, name, max_m=None, extra=()):
    cands = {'NeSCal': {f'{variant}{e}_{lam:g}': per for (variant, e, lam), per in nescal_grid(root, cfg, extra).items()}}
    for m, tags in competitor_grid(root, cfg, extra).items():
        cands[NICE.get(m + '_FINAL', m)] = tags
    curves = {}
    for method, tags in cands.items():
        vals = [v for per in tags.values() if (v := _ms([_val(rp, 'nll') for rp in per.values()])[0]) is not None]
        if len(vals) < 2:
            continue
        v = np.sort(np.array(vals))
        n = len(v)

        exp_min = []
        for m_ in range(1, n + 1):
            num = np.array([np.prod([(n - i - 1 - k) / (n - k) for k in range(m_)]) if n - i - 1 >= m_ else 0.0
                            for i in range(n)])
            p_at_least = np.concatenate([[1.0], num[:-1]])
            probs = p_at_least - num
            exp_min.append(float((v * probs).sum() / probs.sum()))
        curves[method] = exp_min
    if not curves:
        print(f'[evp] {name}: no grids found')
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for method, c in curves.items():
        ax.plot(range(1, len(c) + 1), c, '-o', ms=3, label=f'{method} ({len(c)})')
    ax.set_xlabel('configurations sampled, $m$')
    ax.set_ylabel('expected best validation NLL')
    ax.set_title(f'{name}: expected validation performance', fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fp = os.path.join(out_dir, f'evp_{name.replace("/", "_")}.pdf')
    fig.savefig(fp)
    plt.close(fig)
    print(f'[evp] {name}: {", ".join(f"{k}={len(v)}" for k, v in curves.items())} -> {fp}')
    json.dump(curves, open(os.path.join(out_dir, f'evp_{name.replace("/", "_")}.json'), 'w'), indent=1)


def pareto(root, out_dir, name, extra=()):
    methods = {}
    for lab, per in reportable(root, extra).items():
        rows = []
        for seed, rp in sorted(per.items()):
            fp = os.path.join(rp, 'posthoc_results.pkl')
            post = load_pkl(fp).get('post', {}).get('ts') if os.path.isfile(fp) else None
            rows.append({'seed': seed, 'accuracy': _test(rp, 'accuracy'),
                         'base': _test(rp, 'nll'),
                         'ts': post.get('nll') if isinstance(post, dict) else None})
        methods[lab] = rows
    result = {}
    targets = [lab for lab in ('ncl_FINAL_cosine', 'ncl_FINAL_constant') if lab in methods]
    for target in targets:
        result[target] = {}
        for metric in ('base', 'ts'):
            pts = {}
            for lab, rows in methods.items():
                if rows and all(r[k] is not None and np.isfinite(r[k])
                                for r in rows for k in ('accuracy', metric)):
                    pts[lab] = (float(np.mean([r['accuracy'] for r in rows])),
                                float(np.mean([r[metric] for r in rows])))
            dom = []
            if target in pts:
                acc, nll = pts[target]
                dom = [lab for lab, (a, n) in pts.items() if lab != target
                       and a >= acc - 1e-12 and n <= nll + 1e-12
                       and (a > acc + 1e-12 or n < nll - 1e-12)]
            excluded = sorted(set(methods) - set(pts))
            status = ('unavailable' if target not in pts else 'dominated' if dom
                      else 'incomplete' if excluded else 'pareto_optimal')
            result[target][metric] = {'status': status, 'dominators': dom,
                'excluded': excluded, 'points': pts,
                'seeds': {lab: [r['seed'] for r in methods[lab]] for lab in pts}}
            print(f'[pareto] {name} {NICE[target]} ({metric}): {status}; '
                  f'dominators={dom}; eligible={len(pts)}/{len(methods)}; excluded={excluded}')
    with open(os.path.join(out_dir, 'pareto.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', action='append', required=True,
                    help='Repeat for all result roots; same dataset/architecture roots are merged.')
    ap.add_argument('--extra_root', '--extra-root', action='append', default=[],
                    help='Additional roots, matched by dataset/architecture; repeat as needed.')
    ap.add_argument('--only', nargs='+', default=['A', 'C', 'D', 'E'], choices=list('ABCDE'))
    ap.add_argument('--out', default='./analysis')
    a = ap.parse_args()
    groups = {}
    for root in dict.fromkeys(os.path.realpath(os.path.expanduser(r)) for r in a.root + a.extra_root):
        if not any(os.path.isfile(os.path.join(root, mf)) for mf in MANIFESTS):
            ap.error(f'No supported manifest found in {root}')
        groups.setdefault(_setting(root), []).append(root)
    summary = {}
    failed = False
    for (dataset, arch), roots in groups.items():
        root, *extra = roots
        cfg = {'dataset': dataset, 'arch_name': arch}
        name = f'{dataset}/{arch}'
        out_dir = os.path.join(a.out, re.sub(r'[^A-Za-z0-9_.-]', '_', dataset),
                               re.sub(r'[^A-Za-z0-9_.-]', '_', arch))
        os.makedirs(out_dir, exist_ok=True)
        print(f'\n== {name}: {roots}', flush=True)
        print('Duplicate method/config + seed: first supplied root wins, with warning.')
        for stage, fn in (('A', lambda: discrimination(root, out_dir, extra)),
                          ('B', lambda: frontier(root, cfg, out_dir, name, extra)),
                          ('C', lambda: selection(root, cfg, out_dir, name, extra)),
                          ('D', lambda: evp(root, cfg, out_dir, name, extra=extra)),
                          ('E', lambda: pareto(root, out_dir, name, extra))):
            if stage in a.only:
                try:
                    value = fn()
                    if stage == 'E':
                        summary[name] = value
                except Exception as e:
                    failed = True
                    print(f'!! stage {stage} failed on {name}: {type(e).__name__}: {e}', flush=True)
    if 'E' in a.only:
        with open(os.path.join(a.out, 'pareto_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        for target in ('ncl_FINAL_cosine', 'ncl_FINAL_constant'):
            for metric in ('base', 'ts'):
                rows = [v[target][metric] for v in summary.values() if target in v]
                if rows:
                    complete = [r for r in rows if not r['excluded'] and r['status'] != 'unavailable']
                    count = sum(r['status'] == 'pareto_optimal' for r in complete)
                    print(f'[summary] {NICE[target]} ({metric}): Pareto-optimal in {count}/{len(complete)} '
                          f'complete settings; {len(rows)-len(complete)} incomplete.')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
