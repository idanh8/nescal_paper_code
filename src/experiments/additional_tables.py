import os
import glob
import json
import argparse
import statistics
import traceback

import experiments.ncr as rnc
import experiments.additional as ad

COLS = [('nll', 'NLL'), ('nll_ts', 'NLL+TS'), ('ece', 'ECE %'), ('ece_ts', 'ECE+TS %'), ('acc', 'Acc %'), ('gap', 'NLL removed by TS')]
PCT = ('ece', 'ece_ts', 'acc')


def banner(title):
    print('\n' + '#' * 100 + f'\n# {title}\n' + '#' * 100, flush=True)


def cell(vals, c):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return '--'
    k, nd = (100.0, 2) if c in PCT else (1.0, 4)
    return f'{k * statistics.mean(vals):.{nd}f} ± {k * statistics.pstdev(vals):.{nd}f}'


def table(title, rows):
    print(f'\n### {title}\n', flush=True)
    print('| method | n | ' + ' | '.join(h for _, h in COLS) + ' |\n|---|---|' + '---|' * len(COLS), flush=True)
    for lab, ms in rows.items():
        print(f'| {lab} | {len(ms)} | ' + ' | '.join(cell([m.get(c) for m in ms], c) for c, _ in COLS) + ' |', flush=True)


def mean_or_none(vals):
    vals = [float(v) for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def ci_str(ci, nd=4, scale=1.0):
    return f'{scale * ci["mean"]:+.{nd}f} [{scale * ci["lo"]:+.{nd}f}, {scale * ci["hi"]:+.{nd}f}]'


def verdict(ci, lower_is_better=True):
    lo, hi = ci.get('lo'), ci.get('hi')
    if lo is None or hi is None or lo != lo or hi != hi:
        return 'n/a'
    if lo <= 0.0 <= hi:
        return 'tie'
    return 'win' if (hi < 0.0) == lower_is_better else 'loss'


def shown(ci, key):
    pct = key in PCT
    return f'{ci_str(ci, 2 if pct else 4, 100.0 if pct else 1.0)} {verdict(ci, lower_is_better=key != "acc")}'


def section(fn):
    def wrapped(*a, **k):
        try:
            fn(*a, **k)
        except Exception:
            print(f'\n!! section {fn.__name__} failed:', flush=True)
            traceback.print_exc()
    return wrapped


@section
def winners(roots, heads):
    banner('A. Winners')
    for n, r in roots.items():
        for fn, key in (('ncr_manifest.json', 'ncr'), ('mixup_manifest.json', 'mixup')):
            p = os.path.join(r, fn)
            if os.path.exists(p):
                w = ad.load_json(p).get('winners', {}).get(key, {})
                print(f'  {n:10s} {key:6s} {({k: w[k] for k in ("candidate", "cells_finished", "cells_planned") if k in w})}', flush=True)
    for n, r in roots.items():
        p = arm_manifest(r)
        if os.path.exists(p):
            for lab, e in ad.load_json(p).get('arms', {}).items():
                sel = {k: round(v, 4) for k, v in (e.get('selection') or {}).items()}
                print(f'  {n:10s} {lab:28s} winner {e.get("winner")}' + (f'  val NLL {sel}' if sel else ''), flush=True)
    for p, h in heads:
        print(f'  {p}: {h.get("winners")}', flush=True)


@section
def per_setting(analyses):
    banner('B. Every method per setting (population SDs)')
    for n, (r, d) in analyses.items():
        table(f'{n} ({r})', {lab: list(per.values()) for lab, per in d['methods'].items()})


@section
def versus_ncr(analyses, heads):
    banner(f'C. {ad.REF} minus NCR, paired over seeds (95% t interval); ECE and Acc in points')
    print('\n| setting | NLL | NLL+TS | ECE | ECE+TS | Acc |\n|---|---|---|---|---|---|', flush=True)
    for n, (r, d) in analyses.items():
        row = d.get('paired', {}).get('ncr_FINAL')
        if row:
            print(f'| {n} | ' + ' | '.join(shown(row[k], k) if k in row else '--' for k in ('nll', 'nll_ts', 'ece', 'ece_ts', 'acc')) + ' |', flush=True)
    for p, h in heads:
        row = h.get('paired', {}).get('ncr_FINAL')
        if row:
            print(f'| {os.path.basename(os.path.dirname(os.path.dirname(p)))} | '
                  + ' | '.join(shown(row[k], k) if k in row else '--' for k in ('nll', 'nll_ts', 'ece', 'ece_ts', 'acc')) + ' |', flush=True)
    print('\nCIFAR / TIN rows come from the analysis, which tests NLL, NLL+TS and ECE+TS only.', flush=True)


@section
def label_fixed(analyses):
    banner(f'D. Label-fixed: {ad.REF} minus each method on the test samples both label identically')
    print('\n| setting | against | agree | ΔNLL | ΔECE | ΔNLL after TS | ΔECE after TS |\n|---|---|---|---|---|---|---|', flush=True)
    keys = (('base', 'nll'), ('base', 'ece'), ('ts', 'nll'), ('ts', 'ece'))
    tally = {scope: {k: {'win': 0, 'tie': 0, 'loss': 0, 'n/a': 0} for k in keys} for scope in ('excl. CE', 'excl. CE and NCR')}
    for n, (r, d) in analyses.items():
        for lab, lf in d.get('label_fixed', {}).items():
            cis = [lf[part][k] for part, k in keys]
            print(f'| {n} | {lab} | {lf["agree"]:.3f} | ' + ' | '.join(shown(ci, k) for (_, k), ci in zip(keys, cis)) + ' |', flush=True)
            for scope, skip in (('excl. CE', ('XE_baseline',)), ('excl. CE and NCR', ('XE_baseline', 'ncr_FINAL'))):
                if lab not in skip:
                    for key, ci in zip(keys, cis):
                        tally[scope][key][verdict(ci)] += 1
    for scope, t in tally.items():
        print(f'\nTotals, {scope}:', flush=True)
        for (part, k), c in t.items():
            print(f'  Δ{k}{" after TS" if part == "ts" else ""}: {c["win"]} win / {c["tie"]} tie / {c["loss"]} loss'
                  + (f' / {c["n/a"]} n/a' if c['n/a'] else ''), flush=True)


@section
def arms(roots, analyses):
    banner('E. Additional arms per setting')
    for n, r in roots.items():
        p = arm_manifest(r)
        if not os.path.exists(p):
            continue
        ref = analyses.get(n, (None, {}))[1].get('methods', {}).get(ad.REF, {})
        rows, per_arm = {}, {}
        for lab, e in ad.load_json(p).get('arms', {}).items():
            dirs = {str(s): d for s, d in (e.get('reportable') or {}).items() if d}
            ms = {s: m for s, m in ((s, ad.run_metrics(d, stale_check=False)) for s, d in dirs.items()) if m}
            if ms:
                rows[lab], per_arm[lab] = list(ms.values()), (ms, dirs)
        if ref:
            rows[f'{ad.REF} (headline)'] = list(ref.values())
        table(f'{n}: additional arms ({p})', rows)

        if ref:
            print(f'\nArm minus {ad.REF}, paired over seeds; ECE in points:', flush=True)
            for lab, (ms, _) in per_arm.items():
                parts = [f'{k} {shown(rnc.paired_ci({s: m[k] for s, m in ms.items()}, {s: m[k] for s, m in ref.items()}), k)}'
                         for k in ('nll', 'nll_ts', 'ece', 'ece_ts')]
                print(f'  {lab:28s} ' + ' | '.join(parts), flush=True)
        for lab, (ms, dirs) in per_arm.items():
            ann = [ad.load_json(os.path.join(d, 'ann_stats.json')) for d in dirs.values() if os.path.exists(os.path.join(d, 'ann_stats.json'))]
            ann = [a for a in ann if a]
            if ann:
                rec = [a['recall_at_k'] for a in ann if 'recall_at_k' in a]
                t = [a.get('build_s', 0.0) + a.get('search_s', 0.0) for a in ann]
                print(f'  {lab}: recall@10 {statistics.mean(rec):.3f} ± {statistics.pstdev(rec):.3f}, build+search {statistics.mean(t):.1f} s (n={len(ann)})', flush=True)
            hist = [ad.load_json(os.path.join(d, 'drift.json')) for d in dirs.values() if os.path.exists(os.path.join(d, 'drift.json'))]
            hist = [h for h in hist if h]
            if hist:
                for key in ('sqrt_js_edge_to_cache', 'sqrt_js_cache_drift'):
                    phase = {'all': [statistics.mean(x[key] for x in h) for h in hist],
                             'first 10': [statistics.mean(x[key] for x in h[:10]) for h in hist],
                             'entries 50-59': [statistics.mean(x[key] for x in h[50:60]) for h in hist if len(h) >= 60],
                             'last 10': [statistics.mean(x[key] for x in h[-10:]) for h in hist]}
                    print(f'  {lab}: {key} ' + ', '.join(f'{ph} {statistics.mean(v):.4f} ± {statistics.pstdev(v):.4f}'
                                                         for ph, v in phase.items() if v) + f' (n={len(hist)})', flush=True)


@section
def head_tables(heads):
    banner('F. Pre-trained heads')
    for p, h in heads:
        table(f'{p}; winners {h.get("winners")}', {lab: list(per.values()) for lab, per in h['arms'].items()})
        c = list((h.get('costs') or {}).values())
        for kind, keys in (('exact', ('build_search_s', 'peak_gb')), ('ivf', ('build_search_s', 'recall_at_k'))):
            got = {k: mean_or_none([x[kind].get(k) for x in c if kind in x]) for k in keys}
            if any(v is not None for v in got.values()):
                print(f'  {kind} graph: ' + ', '.join(f'{k} {v:.3f}' for k, v in got.items() if v is not None), flush=True)
        for lab, per in h['arms'].items():
            mm = list(per.values())
            vals = {k: [m[k] for m in mm if m.get(k) is not None] for k in ('peak_gb', 'sec_per_epoch_ce', 'sec_per_epoch_reg')}
            print(f'  {lab:18s} ' + ' | '.join(f'{k} {statistics.mean(v):.2f}' for k, v in vals.items() if v), flush=True)


@section
def baseline_tally(analyses):
    banner(f'G. {ad.REF} minus every train-time baseline except CE, paired over seeds')
    keys = ('nll', 'nll_ts', 'ece_ts')
    tally = {k: {'win': 0, 'tie': 0, 'loss': 0, 'n/a': 0} for k in keys}
    print('\n| setting | against | ΔNLL | ΔNLL+TS | ΔECE+TS |\n|---|---|---|---|---|', flush=True)
    for n, (r, d) in analyses.items():
        for lab, row in d.get('paired', {}).items():
            if lab == 'XE_baseline':
                continue
            print(f'| {n} | {lab} | ' + ' | '.join(shown(row[k], k) for k in keys) + ' |', flush=True)
            for k in keys:
                tally[k][verdict(row[k])] += 1
    print('\nTotals:', flush=True)
    for k, c in tally.items():
        print(f'  Δ{k}: {c["win"]} win / {c["tie"]} tie / {c["loss"]} loss' + (f' / {c["n/a"]} n/a' if c['n/a'] else ''), flush=True)
    print('\nRunner-up after TS per setting:', flush=True)
    for n, (r, d) in analyses.items():
        u = d.get('runner_up_nll_ts')
        if u:
            print(f'  {n:10s} {u["label"]:16s} {shown(u, "nll_ts")}', flush=True)


@section
def cache_bench(heads_base):
    banner('H. Distributed-cache benchmark')
    found = sorted(glob.glob(os.path.join(heads_base, 'cache_bench', '*', 'cache_bench.json')))
    for p in found:
        print(f'  {p}: {json.dumps(ad.load_json(p))}', flush=True)
    if not found:
        print('  not run yet', flush=True)


def arm_manifest(root):
    return os.path.join(os.path.dirname(root.rstrip('/')), rnc.OUT_TAG, 'additional_manifest.json')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='*', default=None, help='name=path pairs; default experiments.ncr.ROOTS + experiments.additional.ANALYSIS_EXTRA_ROOTS')
    ap.add_argument('--heads', default='./runs', help='base_out of run_backbones')
    a = ap.parse_args(argv)
    roots = dict(x.split('=', 1) for x in a.roots) if a.roots else {**rnc.ROOTS, **ad.ANALYSIS_EXTRA_ROOTS}
    roots = {n: r for n, r in roots.items() if os.path.isdir(r)}

    analyses = {}
    for n, r in roots.items():
        p = os.path.join(r, 'additional_analysis.json')
        if os.path.exists(p):
            analyses[n] = (r, ad.load_json(p))
        else:
            print(f'  [missing] {n}: {p} (run scripts/run_additional.py --stages analysis)', flush=True)
    heads = [(p, ad.load_json(p)) for p in sorted(glob.glob(os.path.join(a.heads, '*', rnc.OUT_TAG, 'head_summary.json')))]
    print(f'Settings: {list(analyses)} | heads: {[p for p, _ in heads]}', flush=True)

    winners(roots, heads)
    per_setting(analyses)
    versus_ncr(analyses, heads)
    label_fixed(analyses)
    arms(roots, analyses)
    head_tables(heads)
    baseline_tally(analyses)
    cache_bench(a.heads)
    print('\nVerdicts are from NeSCal\'s side and use 95% t intervals over seeds: "tie" means not separated at these seeds.', flush=True)
