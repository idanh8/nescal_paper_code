import os, sys, csv, argparse
import numpy as np
from scipy import stats
from experiments.common import timestamp, load_json, load_pkl
import experiments.nescal as run_nescal
import experiments.competitors as run_competitors
import experiments.posthoc as run_posthoc
import experiments.robustness as run_extras


MAIN_METRICS = ['accuracy', 'nll', 'brier_score', 'ece', 'equal_mass_ece', 'avg_cw_ece', 'mce', 'lce']
def _reportable(run_root):
    out = {}
    for mf in ('run_manifest.json', 'competitors_manifest.json'):
        p = os.path.join(run_root, mf)
        if os.path.exists(p):
            for label, per_seed in load_json(p).get('reportable', {}).items():
                out.setdefault(label, {}).update({int(k): v for k, v in per_seed.items()})
    return out

def _collect(run_root, source, extract):
    runs = _reportable(run_root); out = {}
    for label, per_seed in runs.items():
        vals = []
        for seed, rp in per_seed.items():
            p = os.path.join(rp, source)
            if not os.path.exists(p): continue
            obj = load_json(p) if source.endswith('.json') else load_pkl(p)
            try:
                v = extract(obj)
                if v is not None: vals.append(v)
            except Exception:
                pass
        if vals: out[label] = vals
    return out

def holm_bonferroni(pvals, alpha=0.05):
    valid = sorted([(k, v) for k, v in pvals.items() if v == v], key=lambda x: x[1])
    m = len(valid); out = {}; still = True
    for i, (k, p) in enumerate(valid):
        thr = alpha / (m - i)
        rej = still and p <= thr
        if not rej: still = False
        out[k] = (p, rej, thr)
    for k, v in pvals.items():
        if v != v: out[k] = (float('nan'), False, float('nan'))
    return out

def _fmt(vals):
    a = np.array(vals, float); return f'{a.mean():.4f}±{a.std():.4f}' if len(a) else '—'

def _table(title, rows, headers):
    print(f'\n### {title}\n'); print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for r in rows: print('| ' + ' | '.join(str(x) for x in r) + ' |')

def _write_csv(path, headers, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)

def _paired_p(ref_vals_by_seed, other_vals_by_seed):
    ks = [k for k in ref_vals_by_seed if k in other_vals_by_seed]
    if len(ks) < 2: return float('nan'), float('nan')
    a = np.array([ref_vals_by_seed[k] for k in ks]); b = np.array([other_vals_by_seed[k] for k in ks])
    return float((b - a).mean()), float(stats.ttest_rel(a, b).pvalue)

def _per_seed(run_root, source, extract):
    runs = _reportable(run_root); out = {}
    for label, per_seed in runs.items():
        d = {}
        for seed, rp in per_seed.items():
            p = os.path.join(rp, source)
            if os.path.exists(p):
                obj = load_json(p) if source.endswith('.json') else load_pkl(p)
                try:
                    d[seed] = extract(obj)
                except Exception:
                    pass
        if d: out[label] = d
    return out

def assemble_report(run_root):
    ds_arch = run_root.rstrip('/').split(os.sep)[-3:]
    print(f'\n{"="*70}\nREPORT: {"/".join(ds_arch)}\n{"="*70}', flush=True)
    csv_dir = os.path.join(run_root, 'report_csvs'); os.makedirs(csv_dir, exist_ok=True)


    src = 'test_results.pkl'
    def base_metric(mk): return lambda o: o.get(mk)
    methods = sorted(_reportable(run_root))
    rows = []
    for meth in methods:
        row = [meth] + [_fmt(list(_collect(run_root, src, base_metric(mk)).get(meth, []))) for mk in MAIN_METRICS]
        rows.append(row)
    _table('Main calibration (test, base model)', rows, ['method'] + MAIN_METRICS)
    _write_csv(os.path.join(csv_dir, 'main_calibration.csv'), ['method'] + MAIN_METRICS, rows)


    ref = 'ncl_FINAL_cosine'
    for mk in ('nll', 'ece'):
        ps = _per_seed(run_root, src, base_metric(mk))
        if ref not in ps: continue
        pvals = {meth: _paired_p(ps[ref], ps[meth])[1] for meth in methods if meth != ref and meth in ps}
        holm = holm_bonferroni(pvals, alpha=0.05)
        srows = []
        for meth in methods:
            if meth == ref or meth not in ps: continue
            md, _ = _paired_p(ps[ref], ps[meth])
            p, rej, thr = holm[meth]
            srows.append([meth, f'{md:+.4f}', f'{p:.3f}' if p == p else '—', '✓' if rej else '✗'])
        _table(f'Significance vs {ref} — test {mk.upper()} (paired t + Holm, N seeds)',
               srows, ['method', 'Δmean', 'p', 'sig@0.05'])
        _write_csv(os.path.join(csv_dir, f'significance_{mk}.csv'), ['method', 'delta_mean', 'p', 'sig@0.05'], srows)


    dc = [[mk] + [_fmt(list(_collect(run_root, src, base_metric(mk)).get(m, [])))
                  for m in ('ncl_FINAL_cosine', 'ncl_FINAL_constant')] for mk in ('nll', 'ece', 'accuracy')]
    _table('Decay (cosine) vs constant λ', dc, ['metric', 'cosine', 'constant'])
    _write_csv(os.path.join(csv_dir, 'decay_vs_constant.csv'), ['metric', 'cosine', 'constant'], dc)


    for arm_metric in ('ece', 'nll', 'lfe_js'):
        rows = []
        for meth in methods:
            per_seed = _reportable(run_root).get(meth, {})
            arms = {}
            for seed, rp in per_seed.items():
                p = os.path.join(rp, 'posthoc_results.pkl')
                if not os.path.exists(p): continue
                o = load_pkl(p)
                arms.setdefault('base', []).append(o['pre'].get(arm_metric))
                for a in ('ts', 'local', 'dc', 'dac', 'lore'):
                    if a in o.get('post', {}): arms.setdefault(a, []).append(o['post'][a].get(arm_metric))
            if arms:
                rows.append([meth] + [_fmt([v for v in arms.get(a, []) if v is not None])
                                      for a in ('base', 'ts', 'local', 'dc', 'dac', 'lore')])
        if rows:
            _table(f'Post-hoc suite — test {arm_metric}', rows,
                   ['method', 'base', 'TS', 'LoCal', 'Dirichlet', 'DAC', 'LoRe'])
            _write_csv(os.path.join(csv_dir, f'posthoc_{arm_metric}.csv'),
                       ['method', 'base', 'TS', 'LoCal', 'Dirichlet', 'DAC', 'LoRe'], rows)


    def corr_mean(key):
        def f(o):
            vals = [m[key] for c in o.values() for m in c.values() if key in m]
            return float(np.mean(vals)) if vals else None
        return f
    crows = [[mk] + [_fmt(list(_collect(run_root, 'corruption_results.pkl', corr_mean(mk)).get(m, [])))
                     for m in methods] for mk in ('ece', 'nll', 'accuracy')]
    if any(r[1:] != ['—'] * len(methods) for r in crows):
        _table('CIFAR-C (mean over corruptions×severities)', crows, ['metric'] + methods)
        _write_csv(os.path.join(csv_dir, 'corruption.csv'), ['metric'] + methods, crows)


    orows = []
    for mk, path in (('auroc', lambda o: o['msp']['auroc']), ('fpr_at_95_tpr', lambda o: o['msp']['fpr_at_95_tpr'])):
        orows.append([mk] + [_fmt(list(_collect(run_root, 'ood_results.pkl', path).get(m, []))) for m in methods])
    if any(r[1:] != ['—'] * len(methods) for r in orows):
        _table('SVHN OOD separation (MSP)', orows, ['metric'] + methods)
        _write_csv(os.path.join(csv_dir, 'ood.csv'), ['metric'] + methods, orows)


    def cost(k): return lambda o: o.get('cost', {}).get(k)
    crows = []
    for k, unit in (('train_only_s', 's'), ('graph_build_s', 's'), ('peak_mem_bytes', 'GB')):
        vals = _collect(run_root, 'train_time_diagnostics.pkl', cost(k))
        row = [k]
        for m in methods:
            vv = [v for v in vals.get(m, []) if v is not None]
            if unit == 'GB': vv = [v / 1e9 for v in vv]
            row.append(_fmt(vv))
        crows.append(row)
    _table('Cost / scalability', [['metric'] + methods] + crows, [''] * (len(methods) + 1))
    _write_csv(os.path.join(csv_dir, 'cost.csv'), ['metric'] + methods, crows)


    abl_mf = os.path.join(run_root, 'ablations_manifest.json')
    if os.path.exists(abl_mf):
        arms = list(load_json(abl_mf)['reportable'])
        def abl_collect(label, mk):
            per = load_json(abl_mf)['reportable'][label]; vals = []
            for seed, rp in per.items():
                p = os.path.join(rp, 'test_results.pkl')
                if os.path.exists(p): vals.append(load_pkl(p).get(mk))
            return [v for v in vals if v is not None]
        rows = [['headline (js/hard/k=10)'] + [_fmt(list(_collect(run_root, 'test_results.pkl',
                 base_metric(mk)).get('ncl_FINAL_cosine', []))) for mk in ('nll', 'ece', 'accuracy')]]
        rows += [[a] + [_fmt(abl_collect(a, mk)) for mk in ('nll', 'ece', 'accuracy')] for a in arms]
        _table('Ablations (test)', rows, ['arm', 'nll', 'ece', 'accuracy'])
        _write_csv(os.path.join(csv_dir, 'ablations.csv'), ['arm', 'nll', 'ece', 'accuracy'], rows)


    h_mf = os.path.join(run_root, 'heuristic_manifest.json')
    if os.path.exists(h_mf):
        hm = load_json(h_mf); grid_nll = np.mean(list(_collect(run_root, 'test_results.pkl',
                                    base_metric('nll')).get('ncl_FINAL_cosine', []) or [np.nan]))
        rows = []
        for label, per in hm['reportable'].items():
            vals = [load_pkl(os.path.join(rp, 'test_results.pkl')).get('nll')
                    for rp in per.values() if os.path.exists(os.path.join(rp, 'test_results.pkl'))]
            vals = [v for v in vals if v is not None]
            gates = {s: g for s, g in hm['gate_epochs'].items()} if 'entropy' in label or 'probe' in label or 'cwcc' in label else {}
            rows.append([label, _fmt(vals), f'{np.mean(vals) - grid_nll:+.4f}' if vals else '—'])
        rows = [['grid winner (E*,λ*)', f'{grid_nll:.4f}', '0.0000']] + rows
        realized = [load_pkl(os.path.join(rp, 'train_time_diagnostics.pkl'))['theory'].get('lambda_star')
                    for rp in hm['reportable'].get('hreg_lambda', {}).values()
                    if os.path.exists(os.path.join(rp, 'train_time_diagnostics.pkl'))]
        realized = [r for r in realized if r is not None]
        lam_h_str = _fmt(realized) if realized else 'n/a'
        _table(f'Heuristic-regret (test NLL; grid E*={hm["best_onset"]}, λ*={hm["lambda"]}, '
               f'js_target={hm["js_target"]}, realized λ_h={lam_h_str})',
               rows, ['config', 'test_nll', 'Δ vs grid'])
        _write_csv(os.path.join(csv_dir, 'heuristic_regret.csv'), ['config', 'test_nll', 'delta_vs_grid'], rows)


    pareto = [[m, _fmt(list(_collect(run_root, 'test_results.pkl', base_metric('accuracy')).get(m, []))),
                  _fmt(list(_collect(run_root, 'test_results.pkl', base_metric('ece')).get(m, [])))]
              for m in methods]
    _table('Accuracy–ECE (Pareto view)', pareto, ['method', 'accuracy', 'ece'])
    _write_csv(os.path.join(csv_dir, 'pareto_acc_ece.csv'), ['method', 'accuracy', 'ece'], pareto)


    def conf_incorrect(o):
        p = np.asarray(o['test_conf_vectors']); y = np.asarray(o['test_labels'])
        wrong = p.argmax(1) != y
        return float(p.max(1)[wrong].mean()) if wrong.any() else float('nan')
    ci = [[m, _fmt(list(_collect(run_root, 'test_data.pkl', conf_incorrect).get(m, [])))] for m in methods]
    _table('Mean confidence on incorrect predictions', ci, ['method', 'conf_on_wrong'])
    _write_csv(os.path.join(csv_dir, 'conf_on_incorrect.csv'), ['method', 'conf_on_wrong'], ci)


    def keepfrac(o):
        h = [r.get('boundary_keep_frac') for r in o.get('heuristics_history', []) if r.get('boundary_keep_frac') is not None]
        return float(np.mean(h)) if h else None
    kf = list(_collect(run_root, 'train_time_diagnostics.pkl', keepfrac).get('ncl_FINAL_cosine', []))
    if kf:
        _table('Boundary masking (NeSCal, mean same-argmax edge fraction retained)',
               [['ncl_FINAL_cosine', _fmt(kf)]], ['method', 'keep_fraction'])


    def th(path):
        def f(o):
            t = o.get('theory', {}); b = (t.get('bound_diagnostics') or {})
            return {'rn_rel_err_median': b.get('rn_relative_error_median'),
                    'confsmooth_violation': b.get('confsmooth_violation_rate'),
                    'mean_edge_js': t.get('mean_edge_js_at_activation'),
                    'lambda_star': t.get('lambda_star')}
        return f
    thv = _collect(run_root, 'train_time_diagnostics.pkl', th(''))
    if 'ncl_FINAL_cosine' in thv:
        d = thv['ncl_FINAL_cosine']
        agg = {kk: _fmt([x[kk] for x in d if x.get(kk) is not None]) for kk in
               ('rn_rel_err_median', 'confsmooth_violation', 'mean_edge_js', 'lambda_star')}
        mj = np.mean([x['mean_edge_js'] for x in d if x.get('mean_edge_js')])
        ls = np.mean([x['lambda_star'] for x in d if x.get('lambda_star')])
        _table('Theory (NeSCal cosine)', [[k, v] for k, v in agg.items()]
               + [['lambda_star × mean_edge_js', f'{ls * mj:.2f}  (compare ≈27)']], ['quantity', 'value'])


    w5_mf = os.path.join(run_root, 'wave5_manifest.json')
    if os.path.exists(w5_mf):
        rep = load_json(w5_mf)['reportable']
        def w5c(label, mk):
            return [load_pkl(os.path.join(rp, 'test_results.pkl')).get(mk)
                    for rp in rep[label].values() if os.path.exists(os.path.join(rp, 'test_results.pkl'))]
        rows = [[l] + [_fmt([v for v in w5c(l, mk) if v is not None])
                       for mk in ('nll', 'ece', 'avg_cw_ece', 'accuracy')] for l in sorted(rep)]
        _table('Wave 5 — perturbation robustness (test; avg_cw_ece is the Fisher-test signal)',
               rows, ['config', 'nll', 'ece', 'avg_cw_ece', 'accuracy'])
        _write_csv(os.path.join(csv_dir, 'wave5_robustness.csv'),
                   ['config', 'nll', 'ece', 'avg_cw_ece', 'accuracy'], rows)


    w5_mf = os.path.join(run_root, 'wave5_manifest.json')
    if os.path.exists(w5_mf):
        rep = load_json(w5_mf)['reportable']; C = run_nescal.CONFIG['model_params']['num_classes']
        def ht(o, key):
            p = np.asarray(o['test_conf_vectors']); y = np.asarray(o['test_labels']); ts_ = int(C * 2 / 3)
            head, tail = y < ts_, y >= ts_
            def _acc(m): return float((p[m].argmax(1) == y[m]).mean()) if m.any() else float('nan')
            def _ece(m):
                if not m.any(): return float('nan')
                conf = p[m].max(1); corr = (p[m].argmax(1) == y[m]).astype(float)
                idx = np.clip(np.digitize(conf, np.linspace(0, 1, 16)) - 1, 0, 14); e = 0.0
                for b in range(15):
                    bm = idx == b
                    if bm.sum(): e += bm.sum() / len(conf) * abs(conf[bm].mean() - corr[bm].mean())
                return float(e)
            return {'head_acc': _acc(head), 'tail_acc': _acc(tail), 'head_ece': _ece(head), 'tail_ece': _ece(tail)}[key]
        rows = []
        for l in sorted(x for x in rep if '_lt' in x):
            vals = {k: [] for k in ('head_acc', 'tail_acc', 'head_ece', 'tail_ece')}
            for rp in rep[l].values():
                dp = os.path.join(rp, 'test_data.pkl')
                if os.path.exists(dp):
                    o = load_pkl(dp)
                    for k in vals: vals[k].append(ht(o, k))
            rows.append([l] + [_fmt([v for v in vals[k] if v == v]) for k in ('head_acc', 'tail_acc', 'head_ece', 'tail_ece')])
        if rows:
            _table('Fisher test — head vs tail (tail = least-frequent third; compare ncl_fisher vs ncl_uniform on tail_ece)',
                   rows, ['config', 'head_acc', 'tail_acc', 'head_ece', 'tail_ece'])
            _write_csv(os.path.join(csv_dir, 'fisher_head_tail.csv'),
                       ['config', 'head_acc', 'tail_acc', 'head_ece', 'tail_ece'], rows)


    def _ref(k): return lambda o: o['pre'].get(k)
    rrows = [[m, _fmt(list(_collect(run_root, 'posthoc_results.pkl', _ref('brier_toplabel_cal')).get(m, []))),
                 _fmt(list(_collect(run_root, 'posthoc_results.pkl', _ref('brier_toplabel_ref')).get(m, [])))]
             for m in methods]
    if any(r[1] != '—' for r in rrows):
        _table('Refinement decomposition (top-label Brier = calibration + refinement)',
               rrows, ['method', 'brier_cal', 'brier_ref'])
        _write_csv(os.path.join(csv_dir, 'refinement.csv'), ['method', 'brier_cal', 'brier_ref'], rrows)

    print(f'\nCSVs written to {csv_dir}', flush=True)

def run_full(nescal_cfg, competitors_cfg=None, do_ablations=True, do_hreg=True, do_wave5=False):
    import traceback
    pass
    ds, arch, base = nescal_cfg['dataset'], nescal_cfg['arch_name'], nescal_cfg['base_out']
    ts = timestamp()
    run_root = os.path.join(base, ds, arch, ts)
    try:
        run_nescal.run(nescal_cfg, ts)
        run_competitors.run_root_ref[0] = run_root
        run_competitors.run(ts, cfg=nescal_cfg, reuse_baseline_ts=ts)
        if do_ablations:
            import experiments.ablations as run_ablations; run_ablations.run(run_root, nescal_cfg)
        if do_hreg:
            import experiments.heuristics as run_heuristic_regret; run_heuristic_regret.run(run_root, nescal_cfg)
        if do_wave5:
            import experiments.perturbations as run_wave5; run_wave5.run(run_root, nescal_cfg)
        run_posthoc.run(run_root, ds, arch, data_root=nescal_cfg['data_root'],
                        loader_fn=nescal_cfg.get('loader_fn'))
        if ds in ('cifar10', 'cifar100', 'tinyimagenet'):
            run_extras.run(run_root, ds, data_root=nescal_cfg['data_root'])
        else:
            print(f'  extras (corruptions/OOD) skipped: image-only, dataset={ds}', flush=True)
        assemble_report(run_root)
        pass
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        pass
        raise

def rerun_nescal_and_finish(nescal_cfg, reuse_competitors_from,
                            do_ablations=True, do_hreg=True, do_wave5=False, ts=None):
    import shutil, traceback
    pass
    ds, arch, base = nescal_cfg['dataset'], nescal_cfg['arch_name'], nescal_cfg['base_out']
    ts = ts or timestamp()
    run_root = os.path.join(base, ds, arch, ts)
    try:
        run_nescal.run(nescal_cfg, ts)

        src = os.path.join(base, ds, arch, reuse_competitors_from, 'competitors_manifest.json')
        os.makedirs(run_root, exist_ok=True)
        shutil.copy(src, os.path.join(run_root, 'competitors_manifest.json'))

        if do_ablations:
            import experiments.ablations as ra; ra.run(run_root, nescal_cfg)
        if do_hreg:
            import experiments.heuristics as rh; rh.run(run_root, nescal_cfg)
        if do_wave5:
            import experiments.perturbations as rw; rw.run(run_root, nescal_cfg)

        run_posthoc.run(run_root, ds, arch, data_root=nescal_cfg['data_root'],
                        loader_fn=nescal_cfg.get('loader_fn'))
        if ds in ('cifar10', 'cifar100', 'tinyimagenet'):
            run_extras.run(run_root, ds, data_root=nescal_cfg['data_root'])
        else:
            print(f'  extras (corruptions/OOD) skipped: image-only, dataset={ds}', flush=True)
        assemble_report(run_root)
        pass
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        pass
        raise
