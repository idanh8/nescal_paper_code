import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import re, csv, glob, math, time, pickle, argparse, importlib, traceback
import numpy as np
import torch
from torch.utils.data import Subset

from models.factory import create_model
from metrics.calibration_metrics import brier_score
from experiments.common import load_json, load_pkl
pass

LN2 = math.log(2.0)
C_JS = 1.0 / (8.0 * LN2)
REL_TOL, ABS_TOL = 1e-9, 1e-7
LOG_FLOOR = math.log(1e-150)
T_GRID = (0.01, 0.05, 0.1, 0.25, 0.5)
T_PRINT, R_PRINT = (0.05, 0.1, 0.25), (1, 2, 3, 5, 10)
WALK_R, N_WALKS, CHUNK = 10, 20000, 20000
GAP_MIN = 0.01
_T0 = time.time()


def log(msg):
    print(f'[{(time.time() - _T0) / 3600:6.2f}h] {msg}', flush=True)


def softmax64(logits):
    z = np.asarray(logits, np.float64)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return np.exp(logp), logp


def edge_arrays(P, logP, src, dst, need_quad=True):
    E = len(src)
    keys = ('js', 'l1', 'l2', 'quad', 'maxrel') if need_quad else ('js', 'l1', 'l2')
    out = {k: np.empty(E) for k in keys}
    for a in range(0, E, CHUNK):
        b = min(a + CHUNK, E); i, j = src[a:b], dst[a:b]
        pi, pj, li, lj = P[i], P[j], logP[i], logP[j]
        lm = np.logaddexp(li, lj) - LN2
        js = 0.5 * ((pi * (li - lm)).sum(1) + (pj * (lj - lm)).sum(1)) / LN2
        d = pj - pi
        out['js'][a:b] = np.maximum(js, 0.0)
        out['l1'][a:b] = np.abs(d).sum(1)
        out['l2'][a:b] = np.sqrt((d * d).sum(1))
        if need_quad:
            with np.errstate(over='ignore'):
                inv = np.exp(-np.maximum(li, LOG_FLOOR))
            out['quad'][a:b] = (d * d * inv).sum(1) / (8.0 * LN2)
            out['maxrel'][a:b] = (np.abs(d) * inv).max(1)
    c = P.max(1)
    out['dc'] = np.abs(c[src] - c[dst])
    return out


def ineq(lhs, rhs, mask=None):
    if mask is not None:
        lhs, rhs = lhs[mask], rhs[mask]
    if len(lhs) == 0:
        return {'viol': float('nan'), 'max_ratio': float('nan'), 'tight_med': float('nan'),
                'tight_p90': float('nan'), 'n': 0}
    pos = rhs > ABS_TOL
    r = lhs[pos] / rhs[pos] if pos.any() else np.array([np.nan])
    return {'viol': float(np.mean(lhs > rhs * (1 + REL_TOL) + ABS_TOL)), 'max_ratio': float(np.nanmax(r)),
            'tight_med': float(np.nanmedian(r)), 'tight_p90': float(np.nanpercentile(r, 90)), 'n': int(len(lhs))}


def graph_edges(neighbors):
    nb = np.asarray(neighbors, np.int64)
    return np.repeat(np.arange(nb.shape[0]), nb.shape[1]), nb.reshape(-1)


def reciprocity(neighbors):
    src, dst = graph_edges(neighbors)
    N = neighbors.shape[0]
    fwd = np.unique(src * N + dst)
    return float(np.isin(dst * N + src, fwd).mean())


def random_simple_walks(neighbors, n_walks, r_max, rng):
    nb = np.asarray(neighbors, np.int64); N, k = nb.shape
    W = np.full((n_walks, r_max + 1), -1, np.int64)
    W[:, 0] = rng.integers(0, N, n_walks)
    alive = np.ones(n_walks, bool)
    for r in range(1, r_max + 1):
        nxt = np.full(n_walks, -1, np.int64); todo = alive.copy()
        for _ in range(3 * k):
            if not todo.any():
                break
            idx = np.where(todo)[0]
            cand = nb[W[idx, r - 1], rng.integers(0, k, len(idx))]
            ok = ~(W[idx, :r] == cand[:, None]).any(1)
            nxt[idx[ok]] = cand[ok]; todo[idx[ok]] = False
        alive &= nxt >= 0
        W[:, r] = np.where(alive, nxt, -1)
    return W


def walk_steps(P, logP, W):
    n, R1 = W.shape
    D = np.full((n, R1 - 1), np.nan)
    for l in range(1, R1):
        ok = W[:, l] >= 0
        if ok.any():
            D[ok, l - 1] = edge_arrays(P, logP, W[ok, l - 1], W[ok, l], need_quad=False)['js']
    return D


def pathwise_conf(P, logP, neighbors, rng, r_max=WALK_R, n_walks=N_WALKS):
    W = random_simple_walks(neighbors, n_walks, r_max, rng)
    D = walk_steps(P, logP, W); c = P.max(1)
    cumD, cumS = np.cumsum(D, 1), np.cumsum(np.sqrt(np.maximum(D, 0)), 1)
    rows = {}
    for r in range(1, r_max + 1):
        ok = W[:, r] >= 0
        if ok.sum() < 50:
            break
        lhs = np.abs(c[W[ok, 0]] - c[W[ok, r]])
        rhs = np.sqrt(2 * LN2 * r * cumD[ok, r - 1])
        tele = np.sqrt(2 * LN2) * cumS[ok, r - 1]
        st = ineq(lhs, rhs)
        pos = rhs > ABS_TOL
        st.update({'cs_slack_med': float(np.median(tele[pos] / rhs[pos])) if pos.any() else float('nan'),
                   'lhs_med': float(np.median(lhs)), 'rhs_med': float(np.median(rhs))})
        rows[r] = st
    return rows


def eta_free_block(P, logP, labels, neighbors, lam=None, rng=None, pathwise=True):
    src, dst = graph_edges(neighbors)
    e = edge_arrays(P, logP, src, dst)
    js, c, yhat = e['js'], P.max(1), P.argmax(1)
    out = {'n_nodes': int(P.shape[0]), 'n_edges': int(len(src)),
           'edge_js_mean': float(js.mean()), 'edge_js_median': float(np.median(js)),
           'same_pred_frac': float((yhat[src] == yhat[dst]).mean()),
           'train_acc': float((yhat == labels).mean()),
           'train_ce_full': float(-logP[np.arange(len(labels)), labels].mean())}

    out['lemma_l1'] = ineq(e['l1'], np.sqrt(js / C_JS))
    out['cor_conf'] = ineq(e['dc'], np.sqrt(2 * LN2 * js))
    out['cor_conf_same_pred'] = ineq(e['dc'], np.sqrt(2 * LN2 * js), mask=yhat[src] == yhat[dst])
    out['step_half_l1'] = ineq(e['dc'], 0.5 * e['l1'])
    legacy_rhs = np.sqrt(8 * LN2 * c[src] * js)
    out['legacy_conf_viol_raw'] = float(np.mean(e['dc'] > legacy_rhs))

    rel = np.abs(js - e['quad']) / np.maximum(js, 1e-300)
    eps = float(P.min())
    rem_actual = float(np.abs(js - e['quad']).sum())
    rem_bound_log10 = float(np.log10(max(np.sum(e['l2'] ** 3), 1e-300)) - np.log10(12 * LN2)
                            - 2 * np.log10(max(eps, 1e-300)))
    out['laplace'] = {'rel_err_med': float(np.median(rel)), 'rel_err_p90': float(np.percentile(rel, 90)),
                      'frac_edges_rel_lt_0p1': float(np.mean(rel < 0.1)),
                      'frac_edges_in_regime': float(np.mean(e['maxrel'] <= 0.1)),
                      'log10_sum_quad_over_exact': float(np.log10(max(e['quad'].sum(), 1e-300)) - np.log10(max(js.sum(), 1e-300))),
                      'log10_eps_floor': float(np.log10(max(eps, 1e-300))),
                      'log10_min_p_median_node': float(np.median(np.log10(np.maximum(P.min(1), 1e-300)))),
                      'log10_remainder_bound_over_actual': rem_bound_log10 - float(np.log10(max(rem_actual, 1e-300)))}
    out['js_tail'] = {t: float(np.mean(js >= t)) for t in T_GRID}
    if lam is not None and lam > 0:
        L = out['train_ce_full'] + lam * out['edge_js_mean']
        out['markov'] = {t: {'emp': float(np.mean(js >= t)), 'R_over_t': out['edge_js_mean'] / t,
                             'L_over_lam_t': L / (lam * t),
                             'prob_conf_emp': float(np.mean(e['dc'] > np.sqrt(2 * LN2 * t)))} for t in T_GRID}
        out['markov_L'] = L
    if pathwise:
        out['pathwise'] = pathwise_conf(P, logP, neighbors, rng)
    return out


def across_seed_block(C, Y, src, dst, A=None):
    S = C.shape[0]
    out = {'n_seeds': int(S), 'n_nodes': int(C.shape[1]), 'n_edges': int(len(src))}
    if S < 2 or len(src) == 0:
        return out
    X = C - C.mean(0, keepdims=True)
    out['sigma_c'] = float(np.sqrt(np.mean(C.var(0, ddof=1))))
    xi, xj = X[:, src], X[:, dst]
    out['rho_c_pooled'] = float((xi * xj).sum() / np.sqrt((xi ** 2).sum() * (xj ** 2).sum() + 1e-300))
    kstar = Y[0, src]
    agree = (Y[:, src] == kstar).all(0) & (Y[:, dst] == kstar).all(0)
    out['agree_frac'] = float(agree.mean())
    d = C[:, src] - C[:, dst]
    vd = d.var(0, ddof=1)
    out['sqrt_var_dc_all'] = float(np.sqrt(vd.mean()))
    out['mean_dc_all'] = float(np.abs(d).mean())
    if agree.any():
        s_a, d_a = src[agree], dst[agree]
        out['sqrt_var_dc_agree'] = float(np.sqrt(vd[agree].mean()))
        xa_i, xa_j = X[:, s_a], X[:, d_a]
        rho_a = float((xa_i * xa_j).sum() / np.sqrt((xa_i ** 2).sum() * (xa_j ** 2).sum() + 1e-300))
        sig_a = float(np.sqrt(np.mean(C[:, np.unique(np.r_[s_a, d_a])].var(0, ddof=1))))
        out.update({'rho_c_agree': rho_a, 'sigma_c_agree': sig_a,
                    'sym_ceiling_agree': float(sig_a * np.sqrt(max(2 * (1 - rho_a), 0.0)))})
        if A is not None:
            k = kstar[agree]
            ai, aj = A[s_a, k], A[d_a, k]
            ei, ej = C[:, s_a] - ai, C[:, d_a] - aj
            diff = ei - ej
            rhs = (diff ** 2).mean(0)
            bias2 = diff.mean(0) ** 2
            lhs = (ai - aj) ** 2
            big = np.abs(ai - aj) > GAP_MIN
            out.update({'thr_profitable_frac': float(np.mean(lhs < rhs)),
                        'thr_gap_edge_frac': float(big.mean()),
                        'thr_profitable_frac_gap': float(np.mean(lhs[big] < rhs[big])) if big.any() else float('nan'),
                        'thr_lhs_mean': float(lhs.mean()), 'thr_rhs_mean': float(rhs.mean()),
                        'thr_rhs_bias_share': float(bias2.sum() / max(rhs.sum(), 1e-300)),
                        'gap_mean_agree': float(np.abs(ai - aj).mean()),
                        'gap_med_agree': float(np.median(np.abs(ai - aj)))})
    return out


def flip_rate(Y_prev, Y_now):
    return float(np.mean(Y_prev != Y_now))


def temp_oracle(logits, labels):
    grid = np.unique(np.r_[1.0, np.exp(np.linspace(np.log(0.1), np.log(10.0), 201))])
    z = np.asarray(logits, np.float64); y = np.asarray(labels, np.int64); n = len(y)
    nll, bri = [], []
    for T in grid:
        p, lp = softmax64(z / T)
        nll.append(-lp[np.arange(n), y].mean()); bri.append(brier_score(p, y))
    nll, bri = np.array(nll), np.array(bri); i1 = int(np.where(grid == 1.0)[0][0])
    return {'nll_T1': float(nll[i1]), 'nll_T_opt': float(grid[nll.argmin()]), 'nll_oracle': float(nll.min()),
            'nll_gap': float(nll[i1] - nll.min()), 'brier_T1': float(bri[i1]),
            'brier_T_opt': float(grid[bri.argmin()]), 'brier_oracle': float(bri.min()),
            'brier_gap': float(bri[i1] - bri.min())}


def reportable(run_root):
    out = {}
    for mf in ('run_manifest.json', 'competitors_manifest.json'):
        p = os.path.join(run_root, mf)
        if os.path.exists(p):
            for label, per_seed in load_json(p).get('reportable', {}).items():
                out.setdefault(label, {}).update({int(k): v for k, v in per_seed.items()})
    return out


def scope(rep):
    labels = (['XE_baseline'] if 'XE_baseline' in rep else []) + sorted(l for l in rep if l.startswith('ncl_FINAL'))
    return {l: {s: p for s, p in sorted(rep[l].items()) if os.path.exists(os.path.join(p, 'model_state.pth'))}
            for l in labels}


def ckpt_epochs(run_path):
    out = {}
    for f in glob.glob(os.path.join(run_path, 'checkpoints', 'epoch_*.pt')):
        m = re.fullmatch(r'epoch_(\d+)\.pt', os.path.basename(f))
        if m:
            out[int(m.group(1))] = f
    return out


def build_model(run_path, ckpt=None):
    ctor = load_pkl(os.path.join(run_path, 'model_params.pkl'))
    net = create_model(**ctor['model_params'])
    if ckpt is None:
        sd = torch.load(os.path.join(run_path, 'model_state.pth'), map_location='cpu', weights_only=True)
    else:
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state']
    net.load_state_dict(sd)
    return net.eval()


@torch.inference_mode()
def forward_logits(net, loader):
    L, Y, I = [], [], []
    for batch in loader:
        o = net(batch[0])
        L.append((o[0] if isinstance(o, tuple) else o).float()); Y.append(batch[1]); I.append(batch[2])
    idx = torch.cat(I).numpy()
    assert np.array_equal(idx, np.arange(len(idx))), 'embedding loader is not in node order'
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def base_indices(loader):
    ds, idx = loader.dataset, None
    while True:
        if hasattr(ds, 'base_dataset'):
            ds = ds.base_dataset; continue
        if isinstance(ds, Subset):
            cur = np.asarray(ds.indices, np.int64)
            idx = cur if idx is None else cur[idx]
            ds = ds.dataset; continue
        return idx


def make_loader_fn(man_cfg, loader_module, batch_size, num_workers):
    if loader_module:
        cfg = importlib.import_module(loader_module).CONFIG
        assert cfg['dataset'] == man_cfg.get('dataset'), \
            f"{loader_module}.CONFIG dataset={cfg['dataset']} but suite is {man_cfg.get('dataset')}"
        return cfg['loader_fn']
    from data.image_datasets import load_dataset
    ds, root = man_cfg['dataset'], man_cfg.get('data_root', './data')
    return lambda seed: load_dataset(ds, val_split=0.1, batch_size=batch_size, seed=seed, data_root=root,
                                     num_workers=num_workers)


class Passes:
    def __init__(self, cache_dir, loader_fn):
        self.dir, self.loader_fn, self.loaders, self.bidx = cache_dir, loader_fn, {}, {}
        os.makedirs(cache_dir, exist_ok=True)

    def emb_loader(self, seed):
        if seed not in self.loaders:
            self.loaders[seed] = self.loader_fn(seed)[3]
            self.bidx[seed] = base_indices(self.loaders[seed])
        return self.loaders[seed]

    def path(self, label, seed, tag):
        return os.path.join(self.dir, f'{label}__seed{seed}__{tag}.npz')

    def have(self, label, seed, tag):
        return os.path.exists(self.path(label, seed, tag))

    def get(self, label, seed, tag, run_path, ckpt=None):
        fp = self.path(label, seed, tag)
        if not os.path.exists(fp):
            t0 = time.time()
            logits, y = forward_logits(build_model(run_path, ckpt), self.emb_loader(seed))
            tmp = fp[:-4] + '.tmp.npz'
            np.savez(tmp, logits=logits, labels=y); os.replace(tmp, fp)
            log(f'   pass {label} seed={seed} {tag}: {len(y)} nodes in {time.time() - t0:.0f}s')
        d = np.load(fp)
        return d['logits'], d['labels']


def common_nodes(bidx, seeds, ref_seed, neighbors_ref):
    if any(bidx.get(s) is None for s in seeds) or ref_seed not in seeds:
        return None
    common = bidx[seeds[0]]
    for s in seeds[1:]:
        common = np.intersect1d(common, bidx[s])
    common = np.sort(common)
    pos = {}
    for s in seeds:
        order = np.argsort(bidx[s])
        pos[s] = order[np.searchsorted(bidx[s][order], common)]
    src, dst = graph_edges(neighbors_ref)
    bs, bd = bidx[ref_seed][src], bidx[ref_seed][dst]
    keep = np.isin(bs, common) & np.isin(bd, common)
    return pos, np.searchsorted(common, bs[keep]), np.searchsorted(common, bd[keep])


def cpu_stage(run_root, runs, ref_label):
    res = {'stored': {}, 'traj_stored': {}, 'test': {}, 'graph': {}}
    xe_runs = runs.get('XE_baseline', {})
    for label, per_seed in runs.items():
        for seed, rp in per_seed.items():
            td = load_pkl(os.path.join(rp, 'test_data.pkl'))
            P, _ = softmax64(td['test_logits'])
            t = temp_oracle(td['test_logits'], td['test_labels'])
            t.update({'test_log10_min_p_median': float(np.median(np.log10(np.maximum(P.min(1), 1e-300)))),
                      'test_log10_min_p': float(np.log10(max(P.min(), 1e-300)))})
            res['test'].setdefault(label, {})[seed] = t
            diag = load_pkl(os.path.join(rp, 'train_time_diagnostics.pkl'))
            hh = diag.get('heuristics_history') or []
            res['traj_stored'].setdefault(label, {})[seed] = [
                {'epoch': h['epoch'], 'knn_probe_acc': h.get('knn_probe_acc'),
                 'mean_entropy_train': h.get('mean_entropy_train'), 'same_pred_frac': h.get('boundary_keep_frac'),
                 'edge_js_exact': (h['bound']['rn_exact'] / h['bound']['n_edges']) if h.get('bound') else None}
                for h in hh if 'epoch' in h and not h.get('nonfinite_embeddings')]
            if not label.startswith('ncl'):
                continue
            th, cost = diag.get('theory') or {}, diag.get('cost') or {}
            bd = th.get('bound_diagnostics') or {}
            exact = bd['rn_exact'] / bd['n_edges'] if bd else float('nan')
            act, surr = th.get('mean_edge_js_at_activation'), th.get('R_n_star')
            ctor = load_pkl(os.path.join(rp, 'model_params.pkl'))
            N, onset = ctor['num_epochs'], diag.get('activation_epoch')
            ncl_ep = xe_ep = float('nan')
            if seed in xe_runs and onset is not None:
                sc = os.path.join(xe_runs[seed], 'checkpoints', f'epoch_{onset}.train_only_s.json')
                xd = load_pkl(os.path.join(xe_runs[seed], 'train_time_diagnostics.pkl')).get('cost') or {}
                if os.path.exists(sc) and cost.get('train_only_s') is not None:
                    ncl_ep = (cost['train_only_s'] - float(load_json(sc)['train_only_s'])) / max(N - onset, 1)
                if xd.get('train_only_s') is not None:
                    xe_ep = xd['train_only_s'] / N
                res['stored'].setdefault('XE_baseline', {})[seed] = {
                    'peak_mem_gb': (xd.get('peak_mem_bytes') or float('nan')) / 1e9, 'sec_per_epoch': xe_ep}
            res['stored'].setdefault(label, {})[seed] = {
                'activation_epoch': onset, 'best_epoch': th.get('best_epoch'), 'lambda_star': th.get('lambda_star'),
                'edge_js_act': act, 'edge_js_exact_best': exact, 'R_surrogate_best': surr,
                'surr_minus_exact_rel': (surr - exact) / exact if (surr is not None and exact) else float('nan'),
                'exact_over_act': exact / act if act else float('nan'),
                'L_task_star_minibatch': th.get('L_task_star'),
                'graph_build_s': cost.get('graph_build_s'),
                'peak_mem_gb': (cost.get('peak_mem_bytes') or float('nan')) / 1e9, 'sec_per_epoch': ncl_ep}
            g = load_pkl(os.path.join(rp, 'graph_data.pkl'))
            res['graph'].setdefault(label, {})[seed] = {'reciprocity': reciprocity(g['neighbors']),
                                                        'weights_all_one': bool(np.all(g['neighbor_weights'] == 1))}

    ncl = [l for l in runs if l.startswith('ncl')]
    if len(ncl) > 1:
        for seed in runs[ncl[0]]:
            gs = [load_pkl(os.path.join(runs[l][seed], 'graph_data.pkl'))['neighbors'] for l in ncl if seed in runs[l]]
            if len(gs) > 1:
                res['graph'].setdefault('_cross_arm', {})[seed] = float(np.mean([np.array_equal(gs[0], g) for g in gs[1:]]))
    return res


def pass_best(run_root, runs, ref_label, passes, res, rng):
    res.setdefault('pass', {})
    for label, per_seed in runs.items():
        for seed, rp in per_seed.items():
            gpath = runs.get(ref_label, {}).get(seed) if label == 'XE_baseline' else rp
            if gpath is None or not os.path.exists(os.path.join(gpath, 'graph_data.pkl')):
                log(f'   skip {label} seed={seed}: no graph for this seed'); continue
            nb = load_pkl(os.path.join(gpath, 'graph_data.pkl'))['neighbors']
            logits, y = passes.get(label, seed, 'best', rp)
            if logits.shape[0] != nb.shape[0]:
                log(f'   !! {label} seed={seed}: {logits.shape[0]} nodes vs graph {nb.shape[0]} -- skipped'); continue
            P, logP = softmax64(logits)
            lam = None
            if label.startswith('ncl'):
                th = load_pkl(os.path.join(rp, 'train_time_diagnostics.pkl')).get('theory') or {}
                lam = th.get('lambda_star')
            res['pass'].setdefault(label, {})[seed] = eta_free_block(P, logP, y, nb, lam=lam, rng=rng)
            log(f'   metrics {label} seed={seed} done')


def across_seed_best(runs, ref_label, passes, res):
    res.setdefault('across_best', {})
    ref = runs.get(ref_label, {})
    if not ref:
        return
    ref_seed = sorted(ref)[0]
    nb_ref = load_pkl(os.path.join(ref[ref_seed], 'graph_data.pkl'))['neighbors']
    for label, per_seed in runs.items():
        seeds = [s for s in sorted(per_seed) if passes.have(label, s, 'best')]
        for s in seeds:
            passes.emb_loader(s)
        cn = common_nodes(passes.bidx, seeds, ref_seed, nb_ref) if len(seeds) >= 2 else None
        if cn is None:
            log(f'   across-seed {label}: skipped (need >=2 seeds incl. ref seed {ref_seed} and base indices)'); continue
        pos, src, dst = cn
        C, Y, T = [], [], []
        for s in seeds:
            lg, yl = passes.get(label, s, 'best', per_seed[s])
            P, _ = softmax64(lg)
            C.append(P.max(1)[pos[s]]); Y.append(P.argmax(1)[pos[s]]); T.append(yl[pos[s]])
        if not all(np.array_equal(T[0], t) for t in T[1:]):
            log(f'   across-seed {label}: SKIPPED -- dataset labels differ at shared base indices '
                f'(base dataset is not fixed across seeds)'); continue
        res['across_best'][label] = across_seed_block(np.stack(C), np.stack(Y), src, dst)


def trajectory(runs, ref_label, passes, res):
    res.setdefault('traj', {})
    ref = runs.get(ref_label, {})
    if not ref:
        return
    ref_seed = sorted(ref)[0]
    nb_ref = load_pkl(os.path.join(ref[ref_seed], 'graph_data.pkl'))['neighbors']
    for label, per_seed in runs.items():
        seeds = sorted(per_seed)
        eps = [ckpt_epochs(per_seed[s]) for s in seeds]
        common = sorted(set.intersection(*[set(e) for e in eps])) if eps else []
        if label.startswith('ncl'):
            act = load_pkl(os.path.join(per_seed[seeds[0]], 'train_time_diagnostics.pkl')).get('activation_epoch') or 0
            common = [e for e in common if e > act]
        if not common:
            log(f'   trajectory {label}: no common checkpoints'); continue
        total = len(common) * len(seeds); done = 0
        log(f'   trajectory {label}: epochs {common} x seeds {seeds} ({total} passes, cached ones skipped)')
        prevY = {}
        for e in common:
            for s, ep in zip(seeds, eps):
                passes.get(label, s, f'ep{e:03d}', per_seed[s], ckpt=ep[e]); done += 1
            for s in seeds:
                passes.emb_loader(s)
            rec = {'epoch': e}
            cn = common_nodes(passes.bidx, seeds, ref_seed, nb_ref) if len(seeds) >= 2 else None
            C, Y, T, flips = [], [], [], []
            for s in seeds:
                lg, yl = passes.get(label, s, f'ep{e:03d}', per_seed[s], ckpt=eps[seeds.index(s)][e])
                P, _ = softmax64(lg)
                yh = P.argmax(1)
                if s in prevY:
                    flips.append(flip_rate(prevY[s], yh))
                prevY[s] = yh
                if cn is not None:
                    C.append(P.max(1)[cn[0][s]]); Y.append(yh[cn[0][s]]); T.append(yl[cn[0][s]])
            rec['flip_rate_since_prev'] = float(np.mean(flips)) if flips else float('nan')
            if cn is not None and not all(np.array_equal(T[0], t) for t in T[1:]):
                log(f'   trajectory {label}: across-seed SKIPPED -- base dataset not fixed across seeds'); cn = None
            if cn is not None:
                rec.update(across_seed_block(np.stack(C), np.stack(Y), cn[1], cn[2]))
            res['traj'].setdefault(label, []).append(rec)
            log(f'   trajectory {label} epoch {e}: {done}/{total} passes')


def _ms(vals, fmt='{:.4f}'):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return 'n/a'
    f = (lambda x: f'{x:.2e}') if (0 < np.max(np.abs(v)) < 1e-3) else (lambda x: fmt.format(x))
    return f(v.mean()) if len(v) == 1 else f'{f(v.mean())}±{f(v.std())}'


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _order(labels):
    return sorted(labels, key=lambda l: (0 if l.startswith('XE') else 2 if l.startswith('ncl') else 1, l))


def _h(h):
    return h.replace('|', '\\|')


def table(title, per_label, cols, lines, csv_rows):
    """per_label: {label: {seed: dict}}; cols: [(header, path_tuple)]."""
    hdr = ['method', 'seeds'] + [_h(c[0]) for c in cols]
    lines += [f'\n### {title}\n', '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for label in _order(per_label):
        per_seed = per_label[label]
        cells = [_ms([_get(v, p) for v in per_seed.values()]) for _, p in cols]
        lines.append(f'| {label} | {len(per_seed)} | ' + ' | '.join(cells) + ' |')
        for seed, v in per_seed.items():
            csv_rows.append({'table': title, 'method': label, 'seed': seed,
                             **{h: _get(v, p) for h, p in cols}})


def flat_table(title, rows, cols, lines, csv_rows, key='method'):
    hdr = [key] + [_h(c[0]) for c in cols]
    lines += [f'\n### {title}\n', '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for name, r in rows:
        cells = []
        for _, k in cols:
            x = r.get(k) if isinstance(r, dict) else None
            cells.append('n/a' if x is None or (isinstance(x, float) and not np.isfinite(x))
                         else (f'{x:.2e}' if isinstance(x, float) and 0 < abs(x) < 1e-3 else
                               f'{x:.4f}' if isinstance(x, float) else str(x)))
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        csv_rows.append({'table': title, key: name, **{h: (r.get(k) if isinstance(r, dict) else None) for h, k in cols}})


def report(res, out_dir, header):
    L, rows = report_lines(res, header)
    write_report(L, rows, out_dir)


def report_lines(res, header):
    L, rows = [header], []
    if res.get('stored'):
        table('T1 Stored diagnostics at the selected epoch (CPU)', res['stored'], [
            ('onset', ('activation_epoch',)), ('best_ep', ('best_epoch',)), ('lambda*', ('lambda_star',)),
            ('edgeJS@act', ('edge_js_act',)), ('edgeJS exact@best', ('edge_js_exact_best',)),
            ('exact/act', ('exact_over_act',)), ('R surrogate@best', ('R_surrogate_best',)),
            ('(surr-exact)/exact', ('surr_minus_exact_rel',)), ('L_task* (minibatch)', ('L_task_star_minibatch',)),
            ('graph_build_s', ('graph_build_s',)), ('peak_mem_GB', ('peak_mem_gb',)),
            ('s/epoch', ('sec_per_epoch',))], L, rows)
    for label in _order(res.get('traj_stored', {})):
        per_seed = res['traj_stored'][label]
        by_ep = {}
        for recs in per_seed.values():
            for r in recs:
                by_ep.setdefault(r['epoch'], []).append(r)
        out = []
        for e in sorted(by_ep):
            rr = by_ep[e]
            out.append((e, {k: float(np.mean([x[k] for x in rr if x.get(k) is not None]))
                            if any(x.get(k) is not None for x in rr) else None
                            for k in ('knn_probe_acc', 'mean_entropy_train', 'same_pred_frac', 'edge_js_exact')}))
        flat_table(f'T2 Stored per-epoch trajectory, {label} (seed mean; CPU)', out, [
            ('knn_probe_acc', 'knn_probe_acc'), ('train_entropy', 'mean_entropy_train'),
            ('same_pred_frac', 'same_pred_frac'), ('edge_js_exact', 'edge_js_exact')], L, rows, key='epoch')
    if res.get('test'):
        table('T3 Test set: oracle temperature and floor (CPU)', res['test'], [
            ('NLL T=1', ('nll_T1',)), ('T*_nll', ('nll_T_opt',)), ('NLL oracle', ('nll_oracle',)),
            ('NLL gap', ('nll_gap',)), ('Brier T=1', ('brier_T1',)), ('T*_brier', ('brier_T_opt',)),
            ('Brier gap', ('brier_gap',)), ('log10 min_k p (median pt)', ('test_log10_min_p_median',)),
            ('log10 min p', ('test_log10_min_p',))], L, rows)
    if res.get('graph'):
        g = {k: v for k, v in res['graph'].items() if not k.startswith('_')}
        table('T4 Graph facts (CPU)', g, [('reciprocity', ('reciprocity',)),
                                          ('binary weights', ('weights_all_one',))], L, rows)
        if '_cross_arm' in res['graph']:
            L.append(f"\ncosine/constant graphs identical per seed: {res['graph']['_cross_arm']}")
    if res.get('pass'):
        P = res['pass']
        table('T5 Train-graph smoothness at the selected weights (PASS)', P, [
            ('edgeJS mean', ('edge_js_mean',)), ('same_pred', ('same_pred_frac',)),
            ('L1 viol', ('lemma_l1', 'viol')), ('L1 tight med', ('lemma_l1', 'tight_med')),
            ('conf viol', ('cor_conf', 'viol')), ('conf max ratio', ('cor_conf', 'max_ratio')),
            ('conf tight med', ('cor_conf', 'tight_med')), ('conf tight p90', ('cor_conf', 'tight_p90')),
            ('conf tight med (same pred)', ('cor_conf_same_pred', 'tight_med')),
            ('|dc|/(L1/2) med', ('step_half_l1', 'tight_med')),
            ('legacy sqrt(8ln2 c JS) viol', ('legacy_conf_viol_raw',))], L, rows)
        table('T6 Laplacian form, floor, remainder (PASS)', P, [
            ('rel err med', ('laplace', 'rel_err_med')), ('rel err p90', ('laplace', 'rel_err_p90')),
            ('frac rel<0.1', ('laplace', 'frac_edges_rel_lt_0p1')),
            ('frac in regime', ('laplace', 'frac_edges_in_regime')),
            ('log10 sum quad/exact', ('laplace', 'log10_sum_quad_over_exact')),
            ('log10 eps', ('laplace', 'log10_eps_floor')),
            ('log10 min_k p (median node)', ('laplace', 'log10_min_p_median_node')),
            ('log10 remainder bound/actual', ('laplace', 'log10_remainder_bound_over_actual'))], L, rows)
        ncl = {k: v for k, v in P.items() if k.startswith('ncl')}
        if ncl:
            cols = [('train CE (full pass)', ('train_ce_full',)), ('L=CE+lam R', ('markov_L',))]
            for t in T_PRINT:
                cols += [(f'P(JS>={t})', ('markov', t, 'emp')), (f'R/{t}', ('markov', t, 'R_over_t')),
                         (f'L/(lam {t})', ('markov', t, 'L_over_lam_t')),
                         (f'P(dc>sqrt(2ln2 {t}))', ('markov', t, 'prob_conf_emp'))]
            table('T7 Markov (Lemma 5) and probabilistic smoothness (Cor. 6) at the selected epoch (PASS)',
                  ncl, cols, L, rows)
        cols = []
        for r in R_PRINT:
            cols += [(f'r{r} viol', ('pathwise', r, 'viol')), (f'r{r} tight', ('pathwise', r, 'tight_med')),
                     (f'r{r} CS slack', ('pathwise', r, 'cs_slack_med'))]
        table('T8 Pathwise bound, confidence half (Thm 9; random simple paths; PASS)', P, cols, L, rows)
    acol = [('seeds', 'n_seeds'), ('common nodes', 'n_nodes'), ('sigma_c', 'sigma_c'), ('rho pooled', 'rho_c_pooled'),
            ('agree frac', 'agree_frac'), ('sqrt Var(dc) agree', 'sqrt_var_dc_agree'),
            ('rho agree', 'rho_c_agree'), ('sigma agree', 'sigma_c_agree'),
            ('sigma*sqrt(2(1-rho))', 'sym_ceiling_agree'), ('mean |dc|', 'mean_dc_all')]
    if res.get('across_best'):
        flat_table('T9 Across-seed at the selected weights (Prop. 11 inputs; PASS)',
                   [(l, res['across_best'][l]) for l in _order(res['across_best'])], acol, L, rows)
    for label in _order(res.get('traj', {})):
        flat_table(f'T10 Across-seed trajectory over checkpoints, {label} (PASS)',
                   [(r['epoch'], r) for r in res['traj'][label]],
                   [('flip rate', 'flip_rate_since_prev')] + acol[2:], L, rows, key='epoch')
    return L, rows


def write_report(L, rows, out_dir):
    txt = '\n'.join(L)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(txt + '\n')
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(txt, flush=True)


def save(res, out_dir):
    tmp = os.path.join(out_dir, 'results.tmp.pkl')
    with open(tmp, 'wb') as f:
        pickle.dump(res, f)
    os.replace(tmp, os.path.join(out_dir, 'results.pkl'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_root', required=True)
    ap.add_argument('--loader_module', default=None, help='e.g. experiments.run_tabular (non-image suites)')
    ap.add_argument('--threads', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--skip_pass', action='store_true', help='CPU stage only')
    ap.add_argument('--skip_traj', action='store_true', help='no checkpoint trajectory (the long part)')
    ap.add_argument('--seed', type=int, default=0, help='RNG for the random paths')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run_root = a.run_root.rstrip('/')
    out_dir = os.path.join(run_root, 'theory_audit'); os.makedirs(out_dir, exist_ok=True)
    man = load_json(os.path.join(run_root, 'run_manifest.json'))
    runs = scope(reportable(run_root))
    ref_label = 'ncl_FINAL_cosine' if 'ncl_FINAL_cosine' in runs else next((l for l in runs if l.startswith('ncl')), None)
    header = (f'# Theory audit: {run_root}\n\nmethods: { {l: sorted(s) for l, s in runs.items()} } | graph for XE: '
              f'{ref_label} (same seed) | threads={a.threads}\n'
              f'"viol" = fraction beyond float64 tolerance (rel {REL_TOL:g}, abs {ABS_TOL:g}); "tight" = lhs/rhs.')
    log(f'=== theory audit {run_root} | {header.splitlines()[2]} ===')
    rng = np.random.default_rng(a.seed)
    try:
        res = cpu_stage(run_root, runs, ref_label)
        save(res, out_dir); report(res, out_dir, header)
        passes = None
        if not a.skip_pass:
            loader_fn = make_loader_fn(man.get('config', {}), a.loader_module, a.batch_size, a.num_workers)
            passes = Passes(os.path.join(out_dir, 'cache'), loader_fn)
            try:
                passes.emb_loader(sorted(next(iter(runs.values())))[0])
            except ValueError as e:
                log(f'!! PASS stage unavailable ({e}). Non-image suite? pass --loader_module.'); passes = None
        if passes is not None:
            log('--- PASS stage: selected weights ---')
            pass_best(run_root, runs, ref_label, passes, res, rng)
            across_seed_best(runs, ref_label, passes, res)
            save(res, out_dir); report(res, out_dir, header)
            if not a.skip_traj:
                log('--- PASS stage: checkpoint trajectory ---')
                trajectory(runs, ref_label, passes, res)
                save(res, out_dir); report(res, out_dir, header)
        log(f'=== DONE: {out_dir}/summary.md ===')
        pass
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        pass
        raise
