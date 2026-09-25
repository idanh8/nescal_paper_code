import os, copy, json, argparse, pickle
import numpy as np

import experiments.nescal as run_nescal
import experiments.competitors as run_competitors
import experiments.posthoc as run_posthoc
from experiments.common import (timestamp, run_dir, RunTracker, train_one, select_by_val_nll,
                             write_manifest, load_pkl)
from data.synthetic_datasets import load_synthetic
from metrics.eta_metrics import (all_eta_metrics, point_eta_metrics, bound_predictiveness,
                                 best_temperature_oracle)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONFIG = __import__('reproduction.recipes', fromlist=['source_config']).source_config('synthetic')


def _blob_kwargs(cfg, hetero_strength=None):
    b = dict(cfg['blob'])
    if hetero_strength is not None:
        b['heterogeneity_strength'] = hetero_strength
    return b


def _mlp_base_kwargs(cfg, loaders, seed, root, **extra):
    kw = run_nescal._base_kwargs(cfg, loaders, seed, root, **extra)
    kw['optimizer'] = cfg.get('optimizer', 'sgd')
    kw['use_amp'] = cfg.get('use_amp', False)
    return kw


def _load(cfg, seed, hetero_strength, sidecar_dir, plot_path=None):
    return load_synthetic(val_split=0.2, batch_size=cfg['batch_size'], seed=seed,
                          data_root=cfg['data_root'], num_workers=cfg['num_workers'],
                          sidecar_dir=sidecar_dir, plot_path=plot_path,
                          **_blob_kwargs(cfg, hetero_strength))


def _sidecar(sidecar_dir, seed):
    return dict(np.load(os.path.join(sidecar_dir, f'eta_sidecar_seed{seed}.npz'), allow_pickle=True))


def _split_eta(sidecar, split):

    idx = sidecar[f'{split}_idx']
    return sidecar['PI_eval'][idx], sidecar['point_is_noisy'][idx]


def _run_one_nescal(cfg, ts, seed, loaders, onset, lam, method, sidecar_dir):
    root = run_dir(cfg['base_out'], cfg['dataset'], cfg['arch_name'], ts, method, seed)
    lp = run_nescal._ncl_params(cfg, lam)
    extra = dict(loss_params=lp, resume_from=None,
                 activation_params={'strategy': 'fixed', 'fixed_epoch': onset},
                 lambda_kind=cfg['schedule'], lambda_decay='cosine')
    val, test, div, rp = train_one(RunTracker(cfg['num_epochs'], label=method),
                                   label=f'{method} seed={seed} onset={onset} lam={lam} {cfg["schedule"]}',
                                   epochs_to_train=cfg['num_epochs'],
                                   run_kwargs=_mlp_base_kwargs(cfg, loaders, seed, root, **extra))
    return (None if div else val), test, rp, div


def _val_eta_score(rp, sidecar_dir, seed):
    vp = os.path.join(rp, 'val_data.pkl')
    if not os.path.exists(vp):
        return None
    vd = load_pkl(vp)
    probs = np.asarray(vd.get('val_conf_vectors'))
    if probs is None or probs.ndim != 2:
        return None
    PI_val, region = _split_eta(_sidecar(sidecar_dir, seed), 'val')
    if PI_val.shape[0] != probs.shape[0]:
        return None
    return point_eta_metrics(probs, PI_val)['mse_to_eta']


def _sel_score(cfg, val_results, rp, sidecar_dir, seed, div):
    if div:
        return None
    if cfg['select_metric'] == 'val_eta':
        return _val_eta_score(rp, sidecar_dir, seed)
    return val_results['nll'] if val_results else None


def run_headline(cfg, ts, include_followups=True):
    ds, arch, base = cfg['dataset'], cfg['arch_name'], cfg['base_out']
    run_root = os.path.join(base, ds, arch, ts)
    os.makedirs(run_root, exist_ok=True)
    sidecar_dir = os.path.join(run_root, 'eta_sidecars')
    sb, cf = cfg['stage_b_seeds'], cfg['confirmatory_seeds']
    all_seeds = list(dict.fromkeys(sb + cf))
    headline_H = cfg['blob']['heterogeneity_strength']
    manifest = {'timestamp': ts, 'config': cfg, 'reportable': {}, 'best': {}}

    loaders_by_seed = {}
    scores = {}
    for si, seed in enumerate(sb):
        plot_path = os.path.join(run_root, 'plots', f'blobs_seed{seed}.png') if si == 0 else None
        loaders_by_seed[seed] = _load(cfg, seed, headline_H, sidecar_dir, plot_path=plot_path)
        for onset in cfg['onset_sweep']:
            for lam in cfg['lambda_grid_constant']:
                val, test, rp, div = _run_one_nescal(cfg, ts, seed, loaders_by_seed[seed], onset, lam,
                                                     f'ncl_o{onset}_l{lam:g}', sidecar_dir)
                scores.setdefault((onset, lam), {})[seed] = _sel_score(cfg, val, rp, sidecar_dir, seed, div)
    (best_onset, best_lam), table = select_by_val_nll(scores)
    manifest['best'] = {'onset': best_onset, 'lambda': best_lam, 'headline_H': headline_H,
                        'select_metric': cfg['select_metric']}
    print(f'  -> winner: onset={best_onset}, lambda={best_lam} (by {cfg["select_metric"]})', flush=True)

    for seed in cf:
        loaders_by_seed[seed] = _load(cfg, seed, headline_H, sidecar_dir)
    for seed in all_seeds:
        _, test, rp = _run_one_nescal(cfg, ts, seed, loaders_by_seed[seed], best_onset, best_lam,
                                      'ncl_FINAL', sidecar_dir)[:3]
        manifest['reportable'].setdefault('ncl_FINAL', {})[seed] = rp
    write_manifest(os.path.join(run_root, 'run_manifest.json'),
                   {**manifest, 'best_onset': best_onset, 'lambda_final': best_lam})

    if not include_followups:
        return run_root, sidecar_dir, best_onset, best_lam

    run_competitors.run_root_ref[0] = run_root
    cfg_comp = copy.deepcopy(cfg)
    cfg_comp['loader_fn'] = (lambda s: _load(cfg, s, headline_H, sidecar_dir))
    try:
        run_competitors.run(ts, cfg=cfg_comp, reuse_baseline_ts=None)
    except Exception as e:
        print(f'  !! competitors errored ({type(e).__name__}: {e}) — continuing.', flush=True)

    abl = {'reportable': {}, 'diverged': [], 'errored': []}
    for dvg in cfg['ablations']['divergence']:
        for seed in cfg['ablation_seeds']:
            label = f'abl_div_{dvg}'
            root = run_dir(base, ds, arch, ts, label, seed)
            lp = run_nescal._ncl_params(cfg, best_lam); lp['secondary_params'].update({'divergence_type': dvg})
            extra = dict(loss_params=lp, resume_from=None,
                         activation_params={'strategy': 'fixed', 'fixed_epoch': best_onset},
                         lambda_kind=cfg['schedule'], lambda_decay='cosine',
                         lambda_from_penalty_target=cfg['ablations']['penalty_target'])
            try:
                _, test, div, rp = train_one(RunTracker(cfg['num_epochs'], label=label),
                                             label=f'{label} seed={seed}', epochs_to_train=cfg['num_epochs'],
                                             run_kwargs=_mlp_base_kwargs(cfg, loaders_by_seed[seed], seed, root, **extra))
                abl['reportable'].setdefault(label, {})[seed] = rp
                if div: abl['diverged'].append({'arm': label, 'seed': seed})
            except Exception as e:
                abl['errored'].append({'arm': label, 'seed': seed, 'error': f'{type(e).__name__}: {e}'})
            write_manifest(os.path.join(run_root, 'ablations_manifest.json'), abl)

    try:
        run_posthoc.POSTHOC = ('ts', 'local', 'dc', 'dac', 'lore')
        run_posthoc.run(run_root, ds, arch, data_root=cfg['data_root'], num_bins=cfg['num_bins'],
                        loader_fn=lambda s: _load(cfg, s, headline_H, sidecar_dir))
    except Exception as e:
        print(f'  !! posthoc errored ({type(e).__name__}: {e}).', flush=True)

    return run_root, sidecar_dir, best_onset, best_lam


def compute_eta_report(cfg, run_root, sidecar_dir):
    print('=== Wave6: eta-metrics on FINAL runs ===', flush=True)
    man = json.load(open(os.path.join(run_root, 'run_manifest.json')))
    rep = man.get('reportable', {}).get('ncl_FINAL', {})
    rows = {}
    for seed, rp in rep.items():
        seed = int(seed)
        td = load_pkl(os.path.join(rp, 'test_data.pkl'))
        probs = np.asarray(td['test_conf_vectors']); logits = td.get('test_logits')
        logits = np.asarray(logits) if (logits is not None and np.asarray(logits).ndim == 2) \
            else np.log(np.clip(probs, 1e-12, 1))
        PI_te, region = _split_eta(_sidecar(sidecar_dir, seed), 'test')
        if PI_te.shape[0] != probs.shape[0]:
            print(f'  !! seed {seed}: eta/test length mismatch — skipping', flush=True); continue
        m = all_eta_metrics(probs, logits, PI_te, region_mask=region, run_oracle=True)
        m['nescal_minus_oracle_mse'] = m['mse_to_eta'] - m['oracle_mse_to_eta']
        rows[seed] = m
    if rows:
        keys = sorted({k for r in rows.values() for k in r})
        agg = {k: (float(np.mean([r[k] for r in rows.values() if k in r])),
                   float(np.std([r[k] for r in rows.values() if k in r]))) for k in keys}
        with open(os.path.join(run_root, 'eta_metrics.json'), 'w') as f:
            json.dump({'per_seed': rows, 'aggregate': agg}, f, indent=2, default=str)
        print('  eta-metrics (mean +/- std):', flush=True)
        for k in ('mse_to_eta', 'oracle_mse_to_eta', 'nescal_minus_oracle_mse', 'l1_to_eta',
                  'signed_bias_clean', 'signed_bias_noisy'):
            if k in agg: print(f'    {k}: {agg[k][0]:.5f} +/- {agg[k][1]:.5f}', flush=True)
    return rows


def figure_bound_predictiveness(cfg, ts, best_onset):
    ds, arch, base = cfg['dataset'], cfg['arch_name'], cfg['base_out']
    run_root = os.path.join(base, ds, arch, ts)
    sidecar_dir = os.path.join(run_root, 'eta_sidecars')
    seed = cfg['stage_b_seeds'][0]; H = cfg['blob']['heterogeneity_strength']
    loaders = _load(cfg, seed, H, sidecar_dir)
    sc = _sidecar(sidecar_dir, seed)

    lambdas = cfg['bound_lambda_sweep']
    fig, axes = plt.subplots(1, len(lambdas) + 1, figsize=(4.6 * (len(lambdas) + 1), 4.4), squeeze=False)
    summary, rho_pts = {}, []
    from metrics.graph_metrics import _embed_probs_labels
    from handlers.trainer import Trainer
    for ax, lam in zip(axes[0][:len(lambdas)], lambdas):
        method = f'boundfig_l{lam:g}'
        root = run_dir(base, ds, arch, ts, method, seed)
        lp = run_nescal._ncl_params(cfg, lam)
        extra = dict(loss_params=lp, resume_from=None,
                     activation_params={'strategy': 'fixed', 'fixed_epoch': best_onset},
                     lambda_kind=cfg['schedule'], lambda_decay='cosine')
        _, test, div, rp = train_one(RunTracker(cfg['num_epochs'], label=method),
                                     label=f'{method} seed={seed}', epochs_to_train=cfg['num_epochs'],
                                     run_kwargs=_mlp_base_kwargs(cfg, loaders, seed, root, **extra))
        if div:
            ax.set_title(f'lam={lam:g}: diverged'); continue
        m = Trainer.load_model(rp)
        Z, P, _ = _embed_probs_labels(m.model, loaders[3], m.device, 'final')
        neigh = m._inner_secondary.neighbors.cpu().numpy() if getattr(m._inner_secondary, 'neighbors', None) is not None else None
        PI_tr, _ = _split_eta(sc, 'train')
        bp = bound_predictiveness(P, PI_tr, neighbors=neigh, embeddings=Z, k=cfg['nescal']['k'])
        summary[lam] = {kk: bp[kk] for kk in ('violation_rate', 'spearman_pred_vs_realised',
                                              'spearman_pred_vs_truegap', 'n_edges')}
        rho_pts.append((lam, bp['spearman_pred_vs_truegap']))
        ax.scatter(bp['pred_gap'], bp['realised_gap'], s=6, alpha=0.3, color='tab:blue')
        lim = float(max(bp['pred_gap'].max(), bp['realised_gap'].max(), 1e-6))
        ax.plot([0, lim], [0, lim], 'k:', lw=1, label='y=x (bound tight)')
        ax.set_xlabel('predicted gap  sqrt(8ln2*c_i*JS)'); ax.set_ylabel('realised |c_i-c_j|')
        ax.set_title(f'lam={lam:g}  (viol={bp["violation_rate"]:.2f})'); ax.legend(fontsize=7)
    axt = axes[0][-1]
    if rho_pts:
        xs, ys = zip(*sorted(rho_pts))
        axt.plot(xs, ys, 'o-', color='tab:red')
        axt.set_xlabel('lambda (constant)'); axt.set_ylabel('rho(pred, true posterior gap)')
        axt.set_title('prediction-truth tracking vs lambda'); axt.grid(alpha=0.3)
    fig.suptitle('Figure A - Confidence-smoothness bound holds (blue below y=x); '
                 'predictions track true posterior gap more tightly as lambda grows')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(run_root, 'plots', 'figA_bound_predictiveness.png')
    os.makedirs(os.path.dirname(fp), exist_ok=True); fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    with open(os.path.join(run_root, 'figA_bound_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Figure A -> {fp}', flush=True)


def figure_activation_criterion(cfg, ts, best_onset, best_lam):
    ds, arch, base = cfg['dataset'], cfg['arch_name'], cfg['base_out']
    run_root = os.path.join(base, ds, arch, ts)
    sidecar_dir = os.path.join(run_root, 'eta_sidecars')
    seed = cfg['stage_b_seeds'][0]
    from handlers.trainer import Trainer

    Hs, ns_mse, oracle_mse, xe_mse, V_vals = [], [], [], [], []
    for H in cfg['hetero_sweep']:
        sd_H = os.path.join(run_root, 'eta_sidecars_figB', f'H{H}')
        loaders = _load(cfg, seed, H, sd_H)
        sc = _sidecar(sd_H, seed)
        PI_te, region = _split_eta(sc, 'test')

        root_xe = run_dir(base, ds, arch, ts, f'figB_xe_H{H}', seed)
        _, test_xe, div_xe, rp_xe = train_one(RunTracker(cfg['num_epochs'], label=f'figB_xe_H{H}'),
            label=f'figB_xe H={H}', epochs_to_train=cfg['num_epochs'],
            run_kwargs=_mlp_base_kwargs(cfg, loaders, seed, root_xe,
                loss_params={'primary_loss': 'XE', 'secondary_loss': None}, resume_from=None))

        root_ns = run_dir(base, ds, arch, ts, f'figB_ncl_H{H}', seed)
        lp = run_nescal._ncl_params(cfg, best_lam)
        extra = dict(loss_params=lp, resume_from=None,
                     activation_params={'strategy': 'fixed', 'fixed_epoch': best_onset},
                     lambda_kind=cfg['schedule'], lambda_decay='cosine')
        _, test_ns, div_ns, rp_ns = train_one(RunTracker(cfg['num_epochs'], label=f'figB_ncl_H{H}'),
            label=f'figB_ncl H={H}', epochs_to_train=cfg['num_epochs'],
            run_kwargs=_mlp_base_kwargs(cfg, loaders, seed, root_ns, **extra))
        if div_xe or div_ns:
            print(f'  H={H}: diverged (xe={div_xe}, ncl={div_ns}) — skipping', flush=True); continue

        def _pl(rp):
            td = load_pkl(os.path.join(rp, 'test_data.pkl'))
            p = np.asarray(td['test_conf_vectors']); lg = td.get('test_logits')
            lg = np.asarray(lg) if (lg is not None and np.asarray(lg).ndim == 2) else np.log(np.clip(p, 1e-12, 1))
            return p, lg
        p_ns, _ = _pl(rp_ns); p_xe, lg_xe = _pl(rp_xe)
        mse_ns = point_eta_metrics(p_ns, PI_te)['mse_to_eta']
        mse_xe = point_eta_metrics(p_xe, PI_te)['mse_to_eta']
        _, mse_or, _ = best_temperature_oracle(lg_xe, PI_te)


        V = float('nan')
        try:
            from metrics.graph_metrics import _embed_probs_labels
            from sklearn.neighbors import NearestNeighbors
            mdl = Trainer.load_model(rp_ns)
            Zg, _, _ = _embed_probs_labels(mdl.model, loaders[3], mdl.device, 'final')
            Zg = np.asarray(Zg); k = cfg['nescal']['k']
            nn_idx = NearestNeighbors(n_neighbors=min(k + 1, len(Zg)), metric='cosine').fit(Zg).kneighbors(Zg)[1][:, 1:]
            ytr = np.asarray(sc['y_true'])[np.asarray(sc['train_idx'])]
            if len(ytr) == len(Zg):
                src = np.repeat(np.arange(nn_idx.shape[0]), nn_idx.shape[1]); dst = nn_idx.reshape(-1)
                V = 1.0 - float(np.mean(ytr[src] != ytr[dst]))
            else:
                print(f'  H={H}: V length mismatch (ytr={len(ytr)} Zg={len(Zg)})', flush=True)
        except Exception as e:
            print(f'  H={H}: V computation failed ({type(e).__name__}: {e})', flush=True)
        print(f'  H={H}: mse_ns={mse_ns:.5f} mse_oracle={mse_or:.5f} mse_xe={mse_xe:.5f} V={V:.3f}', flush=True)
        Hs.append(H); ns_mse.append(mse_ns); oracle_mse.append(mse_or); xe_mse.append(mse_xe); V_vals.append(V)

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.plot(Hs, xe_mse, 'o-', color='tab:gray', label='XE baseline')
    ax.plot(Hs, oracle_mse, 's--', color='tab:blue', label='best-T oracle (global rescale)')
    ax.plot(Hs, ns_mse, 'o-', color='tab:red', lw=2, label=f'NeSCal (onset {best_onset}, lam {best_lam:g}, const)')
    ax.set_xlabel('heterogeneity strength H'); ax.set_ylabel('MSE-to-eta (test)')
    if any(np.isfinite(V_vals)):
        ax2 = ax.twinx()
        ax2.plot(Hs, V_vals, '^:', color='tab:green', alpha=0.7, label='graph validity V')
        ax2.set_ylabel('graph validity V', color='tab:green'); ax2.set_ylim(0, 1.02)
        l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='best')
    else:
        ax.legend(fontsize=8, loc='best')
    ax.set_title('Figure B - NeSCal vs the global-rescale oracle across heterogeneity\n'
                 '(V high = valid graph; a local edge shows as NeSCal beating the oracle)')
    fig.tight_layout()
    fp = os.path.join(run_root, 'plots', 'figB_activation_criterion.png')
    os.makedirs(os.path.dirname(fp), exist_ok=True); fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    with open(os.path.join(run_root, 'figB_summary.json'), 'w') as f:
        json.dump({'H': Hs, 'nescal_mse': ns_mse, 'oracle_mse': oracle_mse, 'xe_mse': xe_mse, 'V': V_vals}, f,
                  indent=2, default=str)
    print(f'  Figure B -> {fp}', flush=True)
