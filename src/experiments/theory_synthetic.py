'Theory synthetic.'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv, json, time, argparse, traceback
import numpy as np
import torch
from scipy.stats import spearmanr

import experiments.theory_real as R
from experiments.common import load_json, load_pkl
from data.synthetic_datasets import load_synthetic, make_synthetic
from metrics.eta_metrics import all_eta_metrics
from handlers.trainer import Trainer
from utils.general_utils import make_lambda_schedule
pass

LN2 = R.LN2
log = R.log
PRINT_EPOCHS = (0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 250)


def _sp(x, y):
    return float(spearmanr(x, y).correlation) if len(x) > 10 and np.std(x) > 0 and np.std(y) > 0 else float('nan')


def _mean(x):
    return float(np.mean(x)) if len(x) else float('nan')


def eta_block(P, logP, neighbors, PI, noisy, y_true, rng):
    n = P.shape[0]
    src, dst = R.graph_edges(neighbors)
    e_arr = R.edge_arrays(P, logP, src, dst, need_quad=False)
    js, dc = e_arr['js'], e_arr['dc']
    c, yhat = P.max(1), P.argmax(1)
    a = PI[np.arange(n), yhat]
    E, e = np.abs(a - c), c - a
    sm, gap = np.sqrt(2 * LN2 * js), np.abs(a[src] - a[dst])
    same = yhat[src] == yhat[dst]
    out = {}
    for name, m in (('all', np.ones(n, bool)), ('clean', ~noisy), ('noisy', noisy)):
        out[f'Ecal_{name}'] = _mean(E[m]); out[f'bias_{name}'] = _mean(e[m])
    out['label_validity'] = float((y_true[src] == y_true[dst]).mean())

    rhs = E[src] + sm + gap
    out['prop'] = R.ineq(E[dst], rhs)
    out['prop_same_pred'] = R.ineq(E[dst], rhs, mask=same)
    out['prop_share'] = {'Ecal_i': float(E[src].sum() / rhs.sum()), 'smooth': float(sm.sum() / rhs.sum()),
                         'gap': float(gap.sum() / rhs.sum())}
    out['prop_nonvacuous'] = float(np.mean(rhs < 1.0))

    f1, fD, EE = gap - dc, gap - sm, E[src] + E[dst]
    ob1, obD = f1 > R.ABS_TOL, fD > R.ABS_TOL
    out['obs'] = {'frac_c_form': float(ob1.mean()), 'frac_D_form': float(obD.mean()),
                  'floorD_mean_obstructed': _mean(fD[obD]),
                  'c_form': R.ineq(f1, EE, mask=ob1), 'D_form': R.ineq(fD, EE, mask=obD)}

    rs, rd = rng.integers(0, n, len(src)), rng.integers(0, n, len(src))
    ok = rs != rd
    tv = 0.5 * np.abs(PI[src] - PI[dst]).sum(1)
    out['fid'] = {'gap_edges': _mean(gap), 'gap_random': _mean(np.abs(a[rs[ok]] - a[rd[ok]])),
                  'tv_edges': _mean(tv), 'tv_random': _mean(0.5 * np.abs(PI[rs[ok]] - PI[rd[ok]]).sum(1)),
                  'frac_gap_lt_0p05': float(np.mean(gap < 0.05)),
                  'sp_js_vs_tv': _sp(js, tv), 'sp_dc_vs_gap': _sp(dc, gap), 'sp_dc_vs_tv': _sp(dc, tv)}

    sh, df = ((e[src] + e[dst]) / 2) ** 2, ((e[src] - e[dst]) / 2) ** 2
    out['dec'] = {'shared': _mean(sh), 'diff': _mean(df), 'shared_same': _mean(sh[same]), 'diff_same': _mean(df[same])}

    reg = {}
    for name, m in (('clean-clean', ~noisy[src] & ~noisy[dst]), ('noisy-noisy', noisy[src] & noisy[dst]),
                    ('mixed', noisy[src] != noisy[dst])):
        reg[name] = {'frac_edges': float(m.mean()), 'gap': _mean(gap[m]), 'obs_D': _mean(obD[m]),
                     'shared': _mean(sh[m]), 'diff': _mean(df[m]), 'js': _mean(js[m])}
    out['region'] = reg

    W = R.random_simple_walks(neighbors, R.N_WALKS, R.WALK_R, rng)
    D = R.walk_steps(P, logP, W); cumD = np.cumsum(D, 1)
    pw = {}
    for r in range(1, R.WALK_R + 1):
        okw = W[:, r] >= 0
        if okw.sum() < 50:
            break
        s_, v_ = W[okw, 0], W[okw, r]
        smt = np.sqrt(2 * LN2 * r * cumD[okw, r - 1])
        rh = E[s_] + smt + np.abs(a[v_] - a[s_])
        st = R.ineq(E[v_], rh)
        st['smooth_share'] = float(smt.sum() / rh.sum())
        pw[r] = st
    out['pathwise_full'] = pw
    return out


def test_eta(rp, sidecar):
    td = load_pkl(os.path.join(rp, 'test_data.pkl'))
    probs, logits = np.asarray(td['test_conf_vectors'], np.float64), np.asarray(td['test_logits'], np.float64)
    te = sidecar['test_idx']
    PI, reg = sidecar['PI_eval'][te], sidecar['point_is_noisy'][te]
    if PI.shape[0] != probs.shape[0]:
        return None
    m = all_eta_metrics(probs, logits, PI, region_mask=reg, run_oracle=True)
    m['mse_minus_oracle'] = m['mse_to_eta'] - m['oracle_mse_to_eta']
    return m


def train_record(ctor, optimizer, loaders, seed, ncl, onset, schedule, cand_layers, emb_layer):
    kw = dict(ctor); kw.update(seed=seed, use_amp=False, optimizer=optimizer)
    if not ncl:
        kw['loss_params'] = {'primary_loss': 'XE', 'secondary_loss': None}
    m = Trainer.from_saved_params(kw)
    tr, emb = loaders[0], loaders[3]
    N, n = kw['num_epochs'], len(emb.dataset)
    C, Y = np.full((N, n), np.nan, np.float32), np.zeros((N, n), np.int16)
    stale, sched, diverged = np.full(N, np.nan), None, False
    for ep in range(N):
        if ncl and not m.secondary_active and ep >= onset:
            m.activate_secondary_loss(emb, cand_layers, emb_layer)
            sched = make_lambda_schedule(schedule, m.criterion.lambda_reg, onset=ep, E=N, decay='cosine')
        if sched is not None:
            m.criterion.lambda_reg = sched(ep)
        m.train_epoch(tr)
        logits, _ = R.forward_logits(m.model, emb)
        if not np.isfinite(logits).all():
            diverged = True; break
        P, _ = R.softmax64(logits)
        C[ep], Y[ep] = P.max(1), P.argmax(1)
        if m.secondary_active:
            stale[ep] = float(np.abs(m._inner_secondary.global_probs.detach().cpu().numpy() - P).sum(1).mean())
    nb = m._inner_secondary.neighbors.cpu().numpy() if (ncl and m.secondary_active) else np.zeros((0, 0), np.int64)
    return C, Y, stale, nb, diverged


def seed_experiment(a, man, runs, out_dir, rng):
    cfg = man['config']
    ncl_label = next(l for l in runs if l.startswith('ncl'))
    ref_rp = runs[ncl_label][sorted(runs[ncl_label])[0]]
    ctor = load_pkl(os.path.join(ref_rp, 'model_params.pkl'))
    onset = int(man.get('best_onset', load_pkl(os.path.join(ref_rp, 'train_time_diagnostics.pkl'))['activation_epoch']))
    schedule = cfg.get('schedule', 'constant')
    data_seed = a.data_seed if a.data_seed is not None else sorted(runs[ncl_label])[0]
    exp_dir = os.path.join(out_dir, 'seed_experiment'); os.makedirs(exp_dir, exist_ok=True)
    loaders = load_synthetic(val_split=0.2, batch_size=cfg['batch_size'], seed=data_seed,
                             data_root=cfg.get('data_root', './data'), num_workers=0, sidecar_dir=exp_dir,
                             **cfg['blob'])
    sc = dict(np.load(os.path.join(exp_dir, f'eta_sidecar_seed{data_seed}.npz'), allow_pickle=True))
    PI, noisy = sc['PI_eval'][sc['train_idx']], sc['point_is_noisy'][sc['train_idx']]
    seeds = list(range(a.exp_seed_start, a.exp_seed_start + a.exp_seeds))
    cand = tuple(cfg.get('candidate_layers', ('final', 'stage_minus_1', 'stage_minus_2')))
    emb_layer = ctor.get('embedding_layer', 'final')
    log(f'--- Part 2: {a.exp_seeds} training seeds x (XE, NeSCal) on data_seed={data_seed}; onset={onset}, '
        f'lambda={ctor["loss_params"].get("lambda_reg")}, schedule={schedule}, epochs={ctor["num_epochs"]} ---')
    store = {}
    for method, ncl in (('XE', False), ('NeSCal', True)):
        for s in seeds:
            fp = os.path.join(exp_dir, f'{method}_seed{s}.npz')
            if not os.path.exists(fp):
                t0 = time.time()
                C, Y, st, nb, dv = train_record(ctor, cfg.get('optimizer', 'sgd'), loaders, s, ncl, onset,
                                                schedule, cand, emb_layer)
                np.savez(fp[:-4] + '.tmp.npz', C=C, Y=Y, stale=st, nb=nb, diverged=dv)
                os.replace(fp[:-4] + '.tmp.npz', fp)
                log(f'   {method} seed={s}: {time.time() - t0:.0f}s{" DIVERGED" if dv else ""}')
            d = np.load(fp)
            if not bool(d['diverged']):
                store.setdefault(method, {})[s] = d
    ref = next((d['nb'] for d in store.get('NeSCal', {}).values() if d['nb'].size), None)
    if ref is None:
        log('!! Part 2: no successful NeSCal run to supply the reference graph'); return None
    src, dst = R.graph_edges(ref)
    out = {'data_seed': data_seed, 'onset': onset, 'seeds': seeds, 'curves': {}}
    for method, runs_m in store.items():
        Cs = np.stack([d['C'] for d in runs_m.values()]); Ys = np.stack([d['Y'] for d in runs_m.values()])
        St = np.stack([d['stale'] for d in runs_m.values()])
        curve = []
        for ep in range(Cs.shape[1]):
            rec = {'epoch': ep, **R.across_seed_block(Cs[:, ep].astype(np.float64), Ys[:, ep].astype(np.int64),
                                                      src, dst, A=PI)}
            a_s = PI[np.arange(PI.shape[0])[None, :], Ys[:, ep].astype(np.int64)]
            Ecal = np.abs(Cs[:, ep] - a_s)
            rec.update({'Ecal_train': float(Ecal.mean()), 'Ecal_clean': float(Ecal[:, ~noisy].mean()),
                        'Ecal_noisy': float(Ecal[:, noisy].mean()),
                        'flip_rate': float((Ys[:, ep] != Ys[:, ep - 1]).mean()) if ep else float('nan'),
                        'staleness_l1': float(np.nanmean(St[:, ep])) if np.isfinite(St[:, ep]).any() else float('nan')})
            curve.append(rec)
        out['curves'][method] = curve
        out.setdefault('n_runs', {})[method] = len(runs_m)
    _plot(out, os.path.join(out_dir, 'seed_experiment.png'))
    return out


def _plot(out, fp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('sigma_c_agree', 'sigma of c across seeds (agreeing edges)'),
              ('rho_c_agree', 'pooled rho across seeds (agreeing edges)'),
              ('thr_profitable_frac_gap', 'profitable frac, agreeing edges with |a_i-a_j|>0.01'),
              ('flip_rate', 'label flip rate'), ('staleness_l1', 'cache staleness ||p_hat-p||_1'),
              ('Ecal_train', 'train E_cal = |a-c|')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, t) in zip(axes.ravel(), panels):
        for method, col in (('XE', 'tab:gray'), ('NeSCal', 'tab:red')):
            cur = out['curves'].get(method, [])
            ax.plot([r['epoch'] for r in cur], [r.get(k, np.nan) for r in cur], color=col, label=method)
        ax.axvline(out['onset'], color='k', ls=':', lw=1)
        ax.set_title(t, fontsize=9); ax.set_xlabel('epoch')
        if k in ('flip_rate', 'staleness_l1'):
            ax.set_yscale('log')
    axes[0, 0].legend()
    fig.suptitle(f"Fixed-data across-seed experiment (data_seed={out['data_seed']}, onset={out['onset']})")
    fig.tight_layout(); fig.savefig(fp, dpi=130); plt.close(fig)


def synth_lines(res, L, rows):
    per = res.get('eta_test', {})
    if per:
        R.table('S1 Test-set eta metrics (oracle = best-T on TEST eta; mse-oracle >= 0 by construction)', per, [
            ('mse_to_eta', ('mse_to_eta',)), ('oracle mse', ('oracle_mse_to_eta',)),
            ('mse - oracle', ('mse_minus_oracle',)), ('oracle T', ('oracle_T',)), ('l1_to_eta', ('l1_to_eta',)),
            ('KL(eta||p)', ('kl_eta_to_p',)), ('bias all (c-a)', ('signed_bias',)),
            ('bias clean', ('signed_bias_clean',)), ('bias noisy', ('signed_bias_noisy',)),
            ('mse clean', ('mse_to_eta_clean',)), ('mse noisy', ('mse_to_eta_noisy',))], L, rows)
    eb = res.get('eta', {})
    if not eb:
        return
    R.table('S2 Train nodes: E_cal=|a-c| and bias c-a by region; graph label validity', eb, [
        ('Ecal all', ('Ecal_all',)), ('Ecal clean', ('Ecal_clean',)), ('Ecal noisy', ('Ecal_noisy',)),
        ('bias all', ('bias_all',)), ('bias clean', ('bias_clean',)), ('bias noisy', ('bias_noisy',)),
        ('edge label validity', ('label_validity',))], L, rows)
    cols = [('viol', ('prop', 'viol')), ('max ratio', ('prop', 'max_ratio')), ('tight med', ('prop', 'tight_med')),
            ('tight med (same pred)', ('prop_same_pred', 'tight_med')), ('share E_i', ('prop_share', 'Ecal_i')),
            ('share smooth', ('prop_share', 'smooth')), ('share gap', ('prop_share', 'gap')),
            ('frac rhs<1', ('prop_nonvacuous',))]
    for r in R.R_PRINT:
        cols += [(f'path r{r} viol', ('pathwise_full', r, 'viol')), (f'r{r} tight', ('pathwise_full', r, 'tight_med')),
                 (f'r{r} smooth share', ('pathwise_full', r, 'smooth_share'))]
    R.table('S3 Cor. 8 propagation per edge and full Thm 9 on random simple paths', eb, cols, L, rows)
    R.table('S4 Prop. 10 obstruction', eb, [
        ('frac obstructed (|a gap|>|c gap|)', ('obs', 'frac_c_form')),
        ('frac obstructed (|a gap|>sqrt(2ln2 JS))', ('obs', 'frac_D_form')),
        ('mean floor if obstructed', ('obs', 'floorD_mean_obstructed')),
        ('c-form viol', ('obs', 'c_form', 'viol')), ('c-form tight med', ('obs', 'c_form', 'tight_med')),
        ('D-form viol', ('obs', 'D_form', 'viol')), ('D-form tight med', ('obs', 'D_form', 'tight_med'))], L, rows)
    R.table('S5a Graph fidelity and tracking (Draft-3 bound is monotone in JS)', eb, [
        ('|a gap| edges', ('fid', 'gap_edges')), ('|a gap| random', ('fid', 'gap_random')),
        ('TV edges', ('fid', 'tv_edges')), ('TV random', ('fid', 'tv_random')),
        ('frac |a gap|<0.05', ('fid', 'frac_gap_lt_0p05')), ('rho(JS, TV)', ('fid', 'sp_js_vs_tv')),
        ('rho(|dc|, |a gap|)', ('fid', 'sp_dc_vs_gap')), ('rho(|dc|, TV)', ('fid', 'sp_dc_vs_tv'))], L, rows)
    R.table('S5b Error decomposition per edge: shared ((e_i+e_j)/2)^2 vs differential ((e_i-e_j)/2)^2', eb, [
        ('shared', ('dec', 'shared')), ('diff', ('dec', 'diff')),
        ('shared (same pred)', ('dec', 'shared_same')), ('diff (same pred)', ('dec', 'diff_same'))], L, rows)
    for reg in ('clean-clean', 'noisy-noisy', 'mixed'):
        R.table(f'S5c By edge region: {reg}', eb, [
            ('frac edges', ('region', reg, 'frac_edges')), ('edge JS', ('region', reg, 'js')),
            ('|a gap|', ('region', reg, 'gap')), ('frac obstructed', ('region', reg, 'obs_D')),
            ('shared', ('region', reg, 'shared')), ('diff', ('region', reg, 'diff'))], L, rows)
    if res.get('paired'):
        keys = list(next(iter(res['paired'].values())).keys())
        R.table('S6 Paired deltas per seed (NeSCal - XE; XE on the same seed\'s graph and data)',
                {'ncl - XE': res['paired']}, [(k, (k,)) for k in keys], L, rows)
    exp = res.get('seed_exp')
    if exp:
        L.append(f"\nPart 2: data_seed={exp['data_seed']}, onset={exp['onset']}, runs={exp.get('n_runs')}; "
                 f"figure: seed_experiment.png; full curves in summary.csv")
        for method in ('XE', 'NeSCal'):
            cur = exp['curves'].get(method, [])
            sel = [r for r in cur if r['epoch'] in PRINT_EPOCHS or r['epoch'] == exp['onset'] or r['epoch'] == len(cur) - 1]
            R.flat_table(f'S7 Fixed-data across-seed trajectory, {method}', [(r['epoch'], r) for r in sel], [
                ('flip rate', 'flip_rate'), ('staleness', 'staleness_l1'), ('Ecal', 'Ecal_train'),
                ('sigma_c', 'sigma_c'), ('rho pooled', 'rho_c_pooled'), ('agree frac', 'agree_frac'),
                ('sigma agree', 'sigma_c_agree'), ('rho agree', 'rho_c_agree'),
                ('ceiling sigma*sqrt(2(1-rho))', 'sym_ceiling_agree'), ('mean |a gap|', 'gap_mean_agree'),
                ('profitable frac', 'thr_profitable_frac'), (f'frac |a gap|>{R.GAP_MIN}', 'thr_gap_edge_frac'),
                ('profitable frac given gap', 'thr_profitable_frac_gap'), ('mean (a_i-a_j)^2', 'thr_lhs_mean'),
                ('mean E[(e_i-e_j)^2]', 'thr_rhs_mean'), ('bias share of rhs', 'thr_rhs_bias_share')],
                L, [], key='epoch')
            for r in cur:
                rows.append({'table': f'S7 curve {method}', **r})


PAIRED_KEYS = {'Ecal_all': ('Ecal_all',), 'bias_clean': ('bias_clean',), 'bias_noisy': ('bias_noisy',),
               'diff': ('dec', 'diff'), 'shared': ('dec', 'shared'), 'frac_obstructed': ('obs', 'frac_D_form'),
               'edge_js': None, 'test_mse_to_eta': None, 'test_bias_clean': None, 'test_bias_noisy': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_root', required=True)
    ap.add_argument('--threads', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument('--skip_part1', action='store_true')
    ap.add_argument('--skip_exp', action='store_true')
    ap.add_argument('--exp_seeds', type=int, default=20)
    ap.add_argument('--exp_seed_start', type=int, default=1000)
    ap.add_argument('--data_seed', type=int, default=None, help='default: first ncl_FINAL seed')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run_root = a.run_root.rstrip('/')
    out_dir = os.path.join(run_root, 'theory_audit'); os.makedirs(out_dir, exist_ok=True)
    man = load_json(os.path.join(run_root, 'run_manifest.json'))
    cfg = man['config']
    runs = R.scope(R.reportable(run_root))
    ref_label = next((l for l in runs if l.startswith('ncl')), None)
    rng = np.random.default_rng(a.seed)
    side_dir = os.path.join(run_root, 'eta_sidecars')
    _truth = {}

    def side(s):
        if s not in _truth:
            d = make_synthetic(val_split=0.2, seed=s, **cfg['blob'])
            te = d['test_idx']
            for label, per_seed in runs.items():
                if s not in per_seed:
                    continue
                td = load_pkl(os.path.join(per_seed[s], 'test_data.pkl'))
                if not np.array_equal(np.asarray(td['test_labels']), d['y'][te]):
                    raise RuntimeError(f'{label} seed={s}: regenerated test labels != saved ones -- the manifest '
                                       f'blob config does not reproduce this run')
                with torch.inference_mode():
                    lg = R.build_model(per_seed[s])(torch.tensor(d['X'][te], dtype=torch.float32))
                lg = (lg[0] if isinstance(lg, tuple) else lg).numpy()
                saved = np.asarray(td['test_logits'], np.float32)
                err = float(np.abs(lg - saved).max())
                if err > 1e-3 * (1.0 + float(np.abs(saved).max())):
                    raise RuntimeError(f'{label} seed={s}: logits on regenerated test inputs differ from saved ones '
                                       f'(max |diff| {err:.3g}) -- the manifest blob config does not reproduce this run')
            fp = os.path.join(side_dir, f'eta_sidecar_seed{s}.npz')
            if os.path.exists(fp):
                sc = np.load(fp, allow_pickle=True)
                if not (np.array_equal(sc['train_idx'], d['train_idx']) and np.array_equal(sc['y'], d['y'])):
                    log(f'!! {fp} is STALE (not the headline data for seed {s}) -- using regenerated truth')
            _truth[s] = d
        return _truth[s]
    header = (f'# Synthetic theory audit: {run_root}\n\nmethods: { {l: sorted(s) for l, s in runs.items()} } | '
              f'onset={man.get("best_onset")} lambda={man.get("lambda_final")} schedule={cfg.get("schedule")} | '
              f'blob={cfg.get("blob")}\n"viol" = fraction beyond float64 tolerance; "tight" = lhs/rhs; a = eta_eval at the '
              f'predicted label.')
    log(f'=== synthetic theory audit {run_root} ===')
    try:
        res = {}
        if not a.skip_part1:
            seeds_all = sorted({s for p in runs.values() for s in p})
            pis = [side(s)['PI_eval'] for s in seeds_all[:2]]
            data_fixed = len(pis) == 2 and pis[0].shape == pis[1].shape and np.allclose(pis[0], pis[1])
            header += f'\ndata identical across run seeds: {data_fixed}'
            log(f'data identical across run seeds: {data_fixed}')
            res = R.cpu_stage(run_root, runs, ref_label)
            loader_fn = lambda s: load_synthetic(val_split=0.2, batch_size=cfg['batch_size'], seed=s,
                                                 data_root=cfg.get('data_root', './data'), num_workers=0, **cfg['blob'])
            passes = R.Passes(os.path.join(out_dir, 'cache'), loader_fn)
            R.pass_best(run_root, runs, ref_label, passes, res, rng)
            res['eta'], res['eta_test'] = {}, {}
            for label, per_seed in runs.items():
                for s, rp in per_seed.items():
                    sc = side(s)
                    te = test_eta(rp, sc)
                    if te is not None:
                        res['eta_test'].setdefault(label, {})[s] = te
                    gpath = runs.get(ref_label, {}).get(s) if label == 'XE_baseline' else rp
                    if gpath is None:
                        continue
                    nb = load_pkl(os.path.join(gpath, 'graph_data.pkl'))['neighbors']
                    logits, y = passes.get(label, s, 'best', rp)
                    tr = sc['train_idx']
                    assert np.array_equal(y, sc['y'][tr]), f'{label} seed={s}: loader labels != regenerated labels'
                    P, logP = R.softmax64(logits)
                    res['eta'].setdefault(label, {})[s] = eta_block(P, logP, nb, sc['PI_eval'][tr],
                                                                   sc['point_is_noisy'][tr].astype(bool),
                                                                   sc['y_true'][tr], rng)
                    log(f'   eta block {label} seed={s} done')
            res['paired'] = {}
            xe, nc = res['eta'].get('XE_baseline', {}), res['eta'].get(ref_label, {})
            for s in sorted(set(xe) & set(nc)):
                d = {}
                for k, path in PAIRED_KEYS.items():
                    if path is not None:
                        d[k] = R._get(nc[s], path) - R._get(xe[s], path)
                d['edge_js'] = res['pass'][ref_label][s]['edge_js_mean'] - res['pass']['XE_baseline'][s]['edge_js_mean']
                tx, tn = res['eta_test'].get('XE_baseline', {}).get(s), res['eta_test'].get(ref_label, {}).get(s)
                if tx and tn:
                    d.update({'test_mse_to_eta': tn['mse_to_eta'] - tx['mse_to_eta'],
                              'test_bias_clean': tn['signed_bias_clean'] - tx['signed_bias_clean'],
                              'test_bias_noisy': tn['signed_bias_noisy'] - tx['signed_bias_noisy']})
                res['paired'][s] = d
            if data_fixed:
                R.across_seed_best(runs, ref_label, passes, res)
                R.trajectory(runs, ref_label, passes, res)
            R.save(res, out_dir)
            L, rows = R.report_lines(res, header); synth_lines(res, L, rows); R.write_report(L, rows, out_dir)
        if not a.skip_exp:
            res['seed_exp'] = seed_experiment(a, man, runs, out_dir, rng)
            R.save(res, out_dir)
            L, rows = R.report_lines(res, header); synth_lines(res, L, rows); R.write_report(L, rows, out_dir)
        log(f'=== DONE: {out_dir}/summary.md ===')
        pass
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        pass
        raise
