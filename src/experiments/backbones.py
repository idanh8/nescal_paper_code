import os
import copy
import json
import math
import time
import glob
import shutil
import argparse
import statistics
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import experiments.ncr as rnc
import experiments.additional as ad
import experiments.nescal as rn
from experiments.common import RunTracker, run_dir, train_one, write_manifest, load_json
from losses.nescal import PreNeighborhoodConsistencyLoss
from losses.ncr import NCRLoss, NeighborVariantLoss, ann_knn, recall_at_k
from losses.losses import get_primary_loss, get_secondary_loss
from models.factory import create_model
from utils.general_utils import set_seed, make_lambda_schedule

HEAD_JOBS = {
    'imagenet_dinov2': {'base': './data/frozen/imagenet_dinov2/base.pt', 'val_n': 50_000, 'hidden': (1024,), 'lr': 1e-3, 'epochs': 30, 'bs': 1024},
    'c100_vit_b16':    {'base': './data/frozen/cifar100_vit_b16/base.pt', 'val_n': 5_000, 'hidden': (256, 256), 'lr': 1e-3, 'epochs': 30, 'bs': 256},
}
HEAD_ONSETS, HEAD_LAMS, HEAD_ALPHAS = [1, 5, 10], [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], [0.03, 0.1, 0.5, 0.9]
HEAD_BASELINES = {
    'FL':     ('set', [{'fl_variant': 'fixed', 'gamma': 3.0}, {'fl_variant': 'adaptive', 'gamma': 3.0}]),
    'LS':     ('set', [{'alpha': 0.05}, {'alpha': 0.1}]),
    'MMCE':   ('set', [{'lambda_reg': v} for v in (1.0, 7.0, 20.0)]),
    'MDCA':   ('set', [{'lambda_reg': v} for v in (1.0, 5.0, 10.0)]),
    's-AvUC': ('sequential', {'theta': [0.1, 0.5, 0.9], 'temp': [0.5, 1.0, 2.0], 'lambda_reg': [0.1, 1.0, 3.0]}),
    'MbLS':   ('set', [{'margin': m, 'lambda_reg': v} for m in (6.0, 10.0) for v in (0.05, 0.1, 0.2)]),
}
MEM_SEED = 18
K_NN, NCR_T = 10, 2.0
FT = {'dataset': 'cifar100', 'arch_name': 'vit_b16_ft', 'model_params': {'arch': 'pretrained_vit_b16', 'num_classes': 100},
      'num_epochs': 30, 'batch_size': 128, 'lr': 0.003, 'momentum': 0.9, 'weight_decay': 0.0, 'optimizer': 'sgd', 'lr_schedule': 'cosine',
      'lr_milestones_frac': [0.5, 0.75], 'lr_gamma': 0.1, 'num_bins': 15, 'diag_every_n': 10, 'data_root': './data', 'base_out': './runs',
      'num_workers': 8, 'early_stopping': False, 'candidate_layers': ('final',),
      'nescal': {'k': K_NN, 'mask_mode': 'none', 'divergence_type': 'js', 'weight_type': 'topk', 'tau': 1.0}}
FT_ONSETS, FT_LAMS, FT_ALPHAS = [10, 20], [1, 3, 10, 30], [0.1, 0.5, 0.9]
BENCH_N, BENCH_K, BENCH_B, BENCH_STEPS = 1_231_167, 1000, 1024, 200
SMOKE_TRAIN_N = 50_000
BASE_OUT = './runs'

ERRORS = []
COLS = ['nll', 'nll_ts', 'ece', 'ece_ts', 'brier', 'acc']


def _mid(grid):
    return grid[len(grid) // 2:len(grid) // 2 + 1]


def _metrics(val_logits, val_y, test_logits, test_y):
    T = ad.fit_temperature(val_logits, val_y)
    p, p_ts = ad._softmax(test_logits), ad._softmax(test_logits / T)
    m, ts = ad.score(p, test_y), ad.score(p_ts, test_y)
    m.update({'nll_ts': ts['nll'], 'ece_ts': ts['ece'], 'brier_ts': ts['brier'], 'T': T, 'gap': m['nll'] - ts['nll'], 'val_acc': float((np.asarray(val_logits).argmax(1) == val_y).mean())})
    return m


def load_job(job):
    d = torch.load(job['base'], map_location='cpu', weights_only=False)
    X, Y = d['trainval_emb'].float(), d['trainval_lab'].long()
    return {'X': X, 'Y': Y, 'Xte': d['test_emb'].float(), 'Yte': d['test_lab'].long(), 'K': len(d['class_names']), 'd': int(d['enc_dim'])}


def split(data, seed, val_n, dev):
    perm = np.random.RandomState(seed).permutation(data['X'].shape[0])
    va, tr = perm[:val_n], perm[val_n:]
    if rnc.SMOKE:
        tr = tr[:SMOKE_TRAIN_N]
    return {'Xtr': data['X'][tr].to(dev), 'Ytr': data['Y'][tr].to(dev), 'Xva': data['X'][va].to(dev), 'Yva': data['Y'][va],
            'Xte': data['Xte'].to(dev), 'Yte': data['Yte'], 'K': data['K'], 'd': data['d']}


@torch.no_grad()
def _logits(model, X, bs=8192):
    model.eval()
    return torch.cat([model(X[s:s + bs]).float() for s in range(0, X.shape[0], bs)])


def _peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None


def _alloc_gb():
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None


@torch.no_grad()
def _probs_chunked(model, X, K, bs=8192):
    model.eval()
    out = torch.empty(X.shape[0], K, device=X.device)
    for s in range(0, X.shape[0], bs):
        out[s:s + bs] = F.softmax(model(X[s:s + bs]).float(), dim=1)
    return out


def graph_for(root, sp, seed, backend):
    p = os.path.join(root, 'graphs', f'seed_{seed}_{backend}.pt')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        g = torch.load(p, map_location='cpu', weights_only=False)
        return g['neighbors'], g['stats']
    dev = sp['Xtr'].device
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    if backend == 'exact':
        loss = PreNeighborhoodConsistencyLoss(k=K_NN)
        loss.precompute_matrices(sp['Xtr'])
        nb, stats = loss.neighbors, {'backend': 'exact', 'build_search_s': time.time() - t, 'peak_gb': _peak_gb()}
    else:
        x = F.normalize(sp['Xtr'], p=2, dim=1).cpu().numpy().astype('float32')
        I, st = ann_knn(x, K_NN, backend)
        st['recall_at_k'] = recall_at_k(x, I, K_NN)
        nb, stats = torch.as_tensor(I, dtype=torch.long), {**st, 'build_search_s': st['build_s'] + st['search_s']}
    print(f'  [graph] seed {seed} {stats}', flush=True)
    torch.save({'neighbors': nb, 'stats': stats}, p)
    return nb, stats


def train_head(job, sp, seed, spec, neighbors=None):
    dev = sp['Xtr'].device
    cuda = dev.type == 'cuda'
    epochs = 2 if rnc.SMOKE else job['epochs']
    set_seed(seed)
    model = create_model('frozen_enc_mlp', num_classes=sp['K'], in_dim=sp['d'], hidden_dims=job['hidden'], norm_type='layer').to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=job['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    mem = {'resident_start': _alloc_gb()}
    prim = sec = None
    if spec['kind'] == 'baseline':
        prim = get_primary_loss(spec['primary'], {**spec.get('primary_params', {}), 'secondary_loss': spec.get('secondary')}).to(dev)
        sec = get_secondary_loss(spec['secondary'], dict(spec.get('secondary_params', {}))).to(dev) if spec.get('secondary') else None
    N, loss_fn, lam_of = sp['Xtr'].shape[0], None, None
    best, best_state, secs_ce, secs_reg, activate_s = math.inf, None, [], [], None
    onset = min(spec.get('onset', 0), epochs - 1)
    for epoch in range(epochs):
        if spec['kind'] in ('nescal', 'ncr') and epoch == onset:
            mem['peak_before_activation'] = _peak_gb()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            if spec['kind'] == 'nescal':
                loss_fn = NeighborVariantLoss(k=K_NN, divergence_type='js')
                loss_fn.precomputed_neighbors = neighbors
                seed_probs = _probs_chunked(model, sp['Xtr'], sp['K']) if spec.get('chunked_seed') else F.softmax(_logits(model, sp['Xtr']), dim=1)
                loss_fn.activate(None, seed_probs)
                del seed_probs
                lam_of = make_lambda_schedule('cosine', spec['lam'], onset=epoch, E=epochs, decay='cosine')
            else:
                loss_fn = NCRLoss(alpha=spec['alpha'], temperature=NCR_T, k=K_NN)
                loss_fn.active = True
                lam_of = lambda _e: 1.0
            activate_s = time.time() - t
            mem['peak_activation'], mem['resident_after_activation'] = _peak_gb(), _alloc_gb()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
        model.train()
        perm = torch.randperm(N)
        t = time.time()
        lam = lam_of(epoch) if lam_of else 0.0
        for s in range(0, N, job['bs']):
            idx = perm[s:s + job['bs']]
            if idx.shape[0] < 2:
                continue
            dev_idx = idx.to(dev)
            x, y = sp['Xtr'][dev_idx], sp['Ytr'][dev_idx]
            logits = model(x)
            if prim is not None:
                loss = prim(logits, y) + (spec['lam'] * sec(logits, y) if sec is not None else 0.0)
            else:
                loss = F.cross_entropy(logits, y)
                if loss_fn is not None:
                    loss = loss + lam * loss_fn(logits, y, node_indices=idx, features=x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        (secs_reg if (loss_fn is not None or prim is not None) else secs_ce).append(time.time() - t)
        sched.step()
        v = F.cross_entropy(_logits(model, sp['Xva']).cpu(), sp['Yva']).item()
        if not math.isfinite(v):
            return None, None, None
        if v < best and (spec['kind'] in ('ce', 'baseline') or epoch >= onset):
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    vl, tl = _logits(model, sp['Xva']).cpu().numpy(), _logits(model, sp['Xte']).cpu().numpy()
    mem['peak_after_activation' if 'peak_activation' in mem else 'peak_training'] = _peak_gb()
    peaks = [p for k, p in mem.items() if k.startswith('peak') and p is not None]
    m = _metrics(vl, sp['Yva'].numpy(), tl, sp['Yte'].numpy())
    m.update({'val_nll': best, 'spec': spec, 'peak_gb': max(peaks) if peaks else None, 'mem_gb': mem, 'activate_s': activate_s,
              'sec_per_epoch_ce': statistics.mean(secs_ce) if secs_ce else None, 'sec_per_epoch_reg': statistics.mean(secs_reg) if secs_reg else None})
    return m, vl, tl


def head_run(root, job, sp, seed, arm, spec, neighbors=None, save_logits=True):
    d = os.path.join(root, arm, f'seed_{seed}')
    fp = os.path.join(d, 'metrics.json')
    if os.path.exists(fp):
        return load_json(fp)
    os.makedirs(d, exist_ok=True)
    print(f'[{rnc._now()}] {arm} seed={seed}', flush=True)
    m, vl, tl = train_head(job, sp, seed, spec, neighbors)
    if m is None:
        m = {'val_nll': None, 'spec': spec, 'diverged': True}
    elif save_logits:
        np.savez(os.path.join(d, 'logits.npz'), val=vl, test=tl)
    with open(fp, 'w') as f:
        json.dump(m, f, indent=2)
    return m


def _select(metrics):
    table = {}
    for c, per in metrics.items():
        vals = [m.get('val_nll') for m in per.values()]
        table[c] = statistics.mean(vals) if vals and all(v is not None for v in vals) else math.inf
    best = min(table, key=table.get)
    if not math.isfinite(table[best]):
        raise RuntimeError(f'every cell diverged: {table}')
    return best, table


def stage_head(name, job):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(BASE_OUT, name, rnc._out_tag())
    data = load_job(job)
    onsets, lams, alphas = (HEAD_ONSETS[:1], _mid(HEAD_LAMS), _mid(HEAD_ALPHAS)) if rnc.SMOKE else (HEAD_ONSETS, HEAD_LAMS, HEAD_ALPHAS)
    print(f'\n########## [{rnc._now()}] HEAD {name} | N={data["X"].shape[0]} K={data["K"]} d={data["d"]} | {job} ##########', flush=True)
    seeds_sel, seeds_all = rnc.SEL_SEEDS, rnc.SEL_SEEDS + rnc.CONF_SEEDS
    cells_ne = {(e, l): {} for e in onsets for l in lams}
    cells_nc = {(e, a): {} for e in onsets for a in alphas}
    method_ne = lambda c: f'nescal_e{c[0]}_l{rnc._tag(c[1])}'
    method_nc = lambda c: f'ncr_e{c[0]}_a{rnc._tag(c[1])}'
    graphs = {}
    for s in seeds_sel:
        sp = split(data, s, job['val_n'], dev)
        graphs[s], _ = graph_for(root, sp, s, 'exact')
        for c in cells_ne:
            cells_ne[c][s] = head_run(root, job, sp, s, method_ne(c), {'kind': 'nescal', 'onset': c[0], 'lam': c[1]}, graphs[s])
        for c in cells_nc:
            cells_nc[c][s] = head_run(root, job, sp, s, method_nc(c), {'kind': 'ncr', 'onset': c[0], 'alpha': c[1]})
        del sp
        rnc._free()
    win_ne, tab_ne = _select(cells_ne)
    win_nc, tab_nc = _select(cells_nc)
    for label, win, table, grid in (('NeSCal', win_ne, tab_ne, (onsets, lams)), ('NCR', win_nc, tab_nc, (onsets, alphas))):
        print(f'  {label} winner {win}; best cells: ' + ', '.join(f'{c}: {v:.4f}' for c, v in sorted(table.items(), key=lambda kv: kv[1])[:4]), flush=True)
        if win[1] in (grid[1][0], grid[1][-1]) and len(grid[1]) > 1:
            print(f'  NOTE: {label} winner on a grid edge; say so.', flush=True)
    arms = {'XE_baseline': {}, 'ncr_FINAL': {}, 'nescal_ivf_FINAL': {}, 'nescal_FINAL': {}}
    costs = {}
    for s in seeds_all:
        sp = split(data, s, job['val_n'], dev)
        nb, st_exact = graph_for(root, sp, s, 'exact')
        arms['XE_baseline'][s] = head_run(root, job, sp, s, 'ce', {'kind': 'ce'})
        arms['nescal_FINAL'][s] = head_run(root, job, sp, s, method_ne(win_ne), {'kind': 'nescal', 'onset': win_ne[0], 'lam': win_ne[1]}, nb)
        arms['ncr_FINAL'][s] = head_run(root, job, sp, s, method_nc(win_nc), {'kind': 'ncr', 'onset': win_nc[0], 'alpha': win_nc[1]})
        nb_ivf, st_ivf = graph_for(root, sp, s, 'faiss_ivf')
        arms['nescal_ivf_FINAL'][s] = head_run(root, job, sp, s, f'nescal_ivf_e{win_ne[0]}_l{rnc._tag(win_ne[1])}', {'kind': 'nescal', 'onset': win_ne[0], 'lam': win_ne[1]}, nb_ivf)
        costs[s] = {'exact': st_exact, 'ivf': st_ivf}
        del sp, nb, nb_ivf
        rnc._free()
    report_head(name, root, job, data, arms, costs, win_ne, win_nc)
    keep = {method_ne(win_ne), method_nc(win_nc), 'ce'}
    for d in glob.glob(os.path.join(root, '*', 'seed_*')):
        if os.path.basename(os.path.dirname(d)) not in keep and not os.path.basename(os.path.dirname(d)).startswith('nescal_ivf'):
            lp = os.path.join(d, 'logits.npz')
            if os.path.exists(lp):
                os.remove(lp)


def report_head(name, root, job, data, arms, costs, win_ne, win_nc):
    print(f'\n### {name}: test metrics, recomputed from logits; *_ts = validation-fitted TS. NeSCal winner (onset, λ)={win_ne}; NCR winner (e, α)={win_nc}\n', flush=True)
    rows = {lab: {c: [m[c] for m in per.values() if c in m] for c in COLS} for lab, per in arms.items()}
    rnc.md_table(rows, COLS, maximize=('acc',))
    ref = arms['nescal_FINAL']
    print('\nNeSCal minus each arm, paired over seeds (95% t interval):', flush=True)
    summary = {}
    for lab in ('XE_baseline', 'ncr_FINAL', 'nescal_ivf_FINAL'):
        row = {c: ad._ci({s: m[c] for s, m in ref.items() if c in m}, {s: m[c] for s, m in arms[lab].items() if c in m}) for c in ('nll', 'nll_ts', 'ece', 'ece_ts', 'acc')}
        summary[lab] = row
        print(f'  {lab:18s} ' + ' | '.join(f'{c} {ad._fmt_ci(ci)} {ad._verdict(ci, lower_is_better=c != "acc")}' for c, ci in row.items()), flush=True)
    print('\nLabel-fixed, NeSCal minus CE on the test samples both label identically:', flush=True)
    per = {}
    for s in ref:
        try:
            a = np.load(os.path.join(root, f'nescal_e{win_ne[0]}_l{rnc._tag(win_ne[1])}', f'seed_{s}', 'logits.npz'))
            b = np.load(os.path.join(root, 'ce', f'seed_{s}', 'logits.npz'))
        except OSError:
            continue
        y = data['Yte'].numpy()
        ra = {'ty': y, 'p': ad._softmax(a['test']), 'p_ts': ad._softmax(a['test'] / ref[s]['T'])}
        rb = {'ty': y, 'p': ad._softmax(b['test']), 'p_ts': ad._softmax(b['test'] / arms['XE_baseline'][s]['T'])}
        per[s] = (ad.label_fixed(ra, rb), ad.label_fixed(ra, rb, ts=True))
    if per:
        for k in ('ece', 'nll', 'conf'):
            print(f'  Δ{k}: base {ad._fmt_ci(ad._ci({s: v[0][0][k] for s, v in per.items()}, {s: v[0][1][k] for s, v in per.items()}))}  after TS {ad._fmt_ci(ad._ci({s: v[1][0][k] for s, v in per.items()}, {s: v[1][1][k] for s, v in per.items()}))}', flush=True)
        print(f'  agreement {statistics.mean(v[0][2] for v in per.values()):.3f}', flush=True)
    cache_gb = (data['X'].shape[0] - job['val_n']) * data['K'] * 4 / 1e9
    print(f'\nCost at n={data["X"].shape[0] - job["val_n"]}, K={data["K"]}, l={data["d"]}: target cache {cache_gb:.2f} GB, edge list {(data["X"].shape[0] - job["val_n"]) * K_NN * 8 / 1e9:.2f} GB', flush=True)
    ex = [c['exact'] for c in costs.values()]
    iv = [c['ivf'] for c in costs.values()]
    print(f'  exact graph: {ad._pm([e["build_search_s"] for e in ex], 1)} s, peak {ad._pm([e["peak_gb"] for e in ex], 2)} GB', flush=True)
    print(f'  IVF graph:   {ad._pm([e["build_search_s"] for e in iv], 1)} s (CPU), recall@10 {ad._pm([e["recall_at_k"] for e in iv], 3)}', flush=True)
    for lab in arms:
        mm = list(arms[lab].values())
        print(f'  {lab:18s} peak {ad._pm([m.get("peak_gb") for m in mm], 2)} GB | s/epoch CE phase {ad._pm([m.get("sec_per_epoch_ce") for m in mm], 2)} | s/epoch with penalty {ad._pm([m.get("sec_per_epoch_reg") for m in mm], 2)}', flush=True)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'head_summary.json'), 'w') as f:
        json.dump({'arms': {k: {str(s): m for s, m in v.items()} for k, v in arms.items()}, 'paired': summary, 'costs': {str(s): c for s, c in costs.items()},
                   'winners': {'nescal': win_ne, 'ncr': win_nc}}, f, indent=2, default=str)


BL_SECONDARY = {'MMCE': ('mmce', ()), 'MDCA': ('mdca', ()), 's-AvUC': ('s-avuc', ('theta', 'temp')), 'MbLS': ('mbls', ('margin',))}
BL_COLS = [('nll', 'NLL', 1, 4, False), ('nll_ts', 'NLL+TS', 1, 4, False), ('ece', 'ECE %', 100, 2, False),
           ('ece_ts', 'ECE+TS %', 100, 2, False), ('acc', 'Acc %', 100, 2, True)]


def _bl_spec(method, cand):
    if method in ('FL', 'LS'):
        return {'kind': 'baseline', 'primary': method, 'primary_params': dict(cand), 'secondary': None, 'lam': 0.0}
    name, keys = BL_SECONDARY[method]
    return {'kind': 'baseline', 'primary': 'XE', 'primary_params': {}, 'secondary': name,
            'secondary_params': {k: cand[k] for k in keys}, 'lam': float(cand['lambda_reg'])}


def _bl_arm(method, cand):
    return f'bl_{method.replace("-", "")}_' + '_'.join(f'{k}{str(cand[k]).replace(".", "p")}' for k in sorted(cand))


def _bl_search(kind, space, run):
    if kind == 'set':
        best, table = _select({i: run(c) for i, c in enumerate(space)})
        return space[best], {str(space[i]): v for i, v in table.items()}
    best, trace = {k: v[0] for k, v in space.items()}, {}
    for p, values in space.items():
        cands = [{**best, p: v} for v in values]
        try:
            i, table = _select({j: run(c) for j, c in enumerate(cands)})
            best, trace[p] = cands[i], {str(cands[j][p]): v for j, v in table.items()}
        except RuntimeError as e:
            trace[p] = f'every value diverged, kept {best[p]} ({e})'
    return best, trace


def _edge_note(kind, space, cand):
    vals = space if kind == 'sequential' else {k: sorted({c[k] for c in space}) for k in cand if not isinstance(cand[k], str)}
    keys = [k for k, v in vals.items() if len(v) > 1 and cand[k] in (min(v), max(v))]
    return f'  NOTE: winner on a grid edge in {keys}' if keys else ''


def stage_head_baselines(name, job):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(BASE_OUT, name, rnc._out_tag())
    summ = load_json(os.path.join(BASE_OUT, name, rnc.OUT_TAG, 'head_summary.json'))
    data = load_job(job)
    seeds_sel, seeds_all = rnc.SEL_SEEDS, rnc.SEL_SEEDS + rnc.CONF_SEEDS
    print(f'\n########## [{rnc._now()}] HEAD BASELINES {name} | {list(HEAD_BASELINES)} | select {seeds_sel}, report {seeds_all} ##########', flush=True)
    sps = {s: split(data, s, job['val_n'], dev) for s in seeds_sel}
    winners, selection = {}, {}
    for method, (kind, space) in HEAD_BASELINES.items():
        if rnc.SMOKE:
            space = space[:1] if kind == 'set' else {k: v[:1] for k, v in space.items()}
        run = lambda c, m=method: {s: head_run(root, job, sps[s], s, _bl_arm(m, c), _bl_spec(m, c), save_logits=False) for s in seeds_sel}
        try:
            winners[method], selection[method] = _bl_search(kind, space, run)
        except RuntimeError as e:
            print(f'  !! {method}: {e}; dropped from the table', flush=True)
            continue
        print(f'  {method}: winner {winners[method]}{_edge_note(kind, space, winners[method])}', flush=True)
    del sps
    rnc._free()
    arms = {m: {} for m in winners}
    for s in seeds_all:
        sp = split(data, s, job['val_n'], dev)
        for method, cand in winners.items():
            arms[method][s] = head_run(root, job, sp, s, _bl_arm(method, cand), _bl_spec(method, cand), save_logits=False)
        del sp
        rnc._free()
    report_baselines(name, root, summ, arms, winners, selection)


def report_baselines(name, root, summ, arms, winners, selection):
    ref = summ['arms']
    rows = {'CE': ref.get('XE_baseline', {}), **{m: {str(s): v for s, v in per.items()} for m, per in arms.items()},
            'NCR': ref.get('ncr_FINAL', {}), 'NeSCal': ref.get('nescal_FINAL', {})}
    rows = {lab: {s: m for s, m in per.items() if m.get('nll') is not None} for lab, per in rows.items()}
    means = {lab: {c: statistics.mean(m[c] * sc for m in per.values()) for c, _, sc, _, _ in BL_COLS} for lab, per in rows.items() if per}
    top = {c: sorted({round(v[c], nd) for v in means.values()}, reverse=mx)[:2] for c, _, _, nd, mx in BL_COLS}
    print(f'\n### {name}: train-time baselines on the frozen head. Test metrics, mean ± population SD over seeds; TS fitted on validation\n', flush=True)
    print('| method | n | ' + ' | '.join(h for _, h, _, _, _ in BL_COLS) + ' |\n|---|---|' + '---|' * len(BL_COLS), flush=True)
    for lab, per in rows.items():
        cells = []
        for c, _, sc, nd, _ in BL_COLS:
            xs = [m[c] * sc for m in per.values()]
            if not xs:
                cells.append('--')
                continue
            txt, r = f'{statistics.mean(xs):.{nd}f} ± {statistics.pstdev(xs):.{nd}f}', round(statistics.mean(xs), nd)
            cells.append(f'**{txt}**' if r == top[c][0] else (f'*{txt}*' if len(top[c]) > 1 and r == top[c][1] else txt))
        print(f'| {lab} | {len(per)} | ' + ' | '.join(cells) + ' |', flush=True)
    if 'NeSCal' in means:
        print('\n  NeSCal rank by mean: ' + ', '.join(f'{h} {sorted(means, key=lambda lab: means[lab][c], reverse=mx).index("NeSCal") + 1}/{len(means)}'
                                                   for c, h, _, _, mx in BL_COLS), flush=True)
    print('\nNeSCal minus each method, paired over seeds (95% t interval; ECE and accuracy in points):', flush=True)
    paired = {}
    for lab, per in rows.items():
        if lab == 'NeSCal' or not per or not rows.get('NeSCal'):
            continue
        paired[lab] = {c: ad._ci({s: m[c] * sc for s, m in rows['NeSCal'].items()}, {s: m[c] * sc for s, m in per.items()}) for c, _, sc, _, _ in BL_COLS}
        print(f'  {lab:7s} ' + ' | '.join(f'{c} {ad._fmt_ci(paired[lab][c], nd)} {ad._verdict(paired[lab][c], lower_is_better=not mx)}'
                                         for c, _, _, nd, mx in BL_COLS), flush=True)
    print('\nWinners: ' + '; '.join(f'{m} {c}' for m, c in winners.items()), flush=True)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'baselines_summary.json'), 'w') as f:
        json.dump({'winners': winners, 'selection': selection, 'arms': {m: {str(s): v for s, v in per.items()} for m, per in arms.items()},
                   'paired_nescal_minus': paired, 'grids': HEAD_BASELINES}, f, indent=2, default=str)


def stage_head_mem(name, job):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(BASE_OUT, name, rnc._out_tag())
    summ = load_json(os.path.join(BASE_OUT, name, rnc.OUT_TAG, 'head_summary.json'))
    (e_ne, l_ne), (e_nc, a_nc) = summ['winners']['nescal'], summ['winners']['ncr']
    data = load_job(job)
    s = MEM_SEED
    print(f'\n########## [{rnc._now()}] HEAD MEMORY {name} | seed {s} | NeSCal (onset, λ)=({e_ne}, {l_ne}), NCR (e, α)=({e_nc}, {a_nc}) ##########', flush=True)
    sp = split(data, s, job['val_n'], dev)
    nb, _ = graph_for(root if rnc.SMOKE else os.path.join(BASE_OUT, name, rnc.OUT_TAG), sp, s, 'exact')
    specs = {'mem_ce': ({'kind': 'ce'}, None, 'XE_baseline'),
             'mem_ncr': ({'kind': 'ncr', 'onset': e_nc, 'alpha': a_nc}, None, 'ncr_FINAL'),
             'mem_nescal': ({'kind': 'nescal', 'onset': e_ne, 'lam': l_ne}, nb, 'nescal_FINAL'),
             'mem_nescal_chunked_seed': ({'kind': 'nescal', 'onset': e_ne, 'lam': l_ne, 'chunked_seed': True}, nb, 'nescal_FINAL')}
    res = {}
    for arm, (spec, g, _) in specs.items():
        rnc._free()
        res[arm] = head_run(root, job, sp, s, arm, spec, g, save_logits=False)
    on_dev = sum(t.numel() * t.element_size() for t in sp.values() if torch.is_tensor(t) and t.device.type == dev.type) / 1e9
    n, K = sp['Xtr'].shape[0], sp['K']
    keys = [('resident_start', 'resident at start'), ('peak_before_activation', 'peak before activation'), ('peak_activation', 'peak at activation'),
            ('resident_after_activation', 'resident after activation'), ('peak_after_activation', 'peak, penalized training'), ('peak_training', 'peak, training')]
    print(f'\n### {name}: peak GPU memory by phase (GB), seed {s}. Embeddings on the device {on_dev:.2f} GB; prediction cache {n * K * 4 / 1e9:.2f} GB (n={n}, K={K}, fp32)\n', flush=True)
    print('| arm | ' + ' | '.join(h for _, h in keys) + ' | overall peak | val NLL | reported val NLL |\n|---|' + '---|' * (len(keys) + 3), flush=True)
    for arm, m in res.items():
        mg, rep = m.get('mem_gb') or {}, (summ['arms'].get(specs[arm][2], {}).get(str(s)) or {}).get('val_nll')
        fmt = lambda v, nd=2: f'{v:.{nd}f}' if isinstance(v, (int, float)) else '--'
        print(f'| {arm} | ' + ' | '.join(fmt(mg.get(k)) for k, _ in keys) + f' | {fmt(m.get("peak_gb"))} | {fmt(m.get("val_nll"), 4)} | {fmt(rep, 4)} |', flush=True)
    with open(os.path.join(root, 'mem_summary.json'), 'w') as f:
        json.dump({'seed': s, 'embeddings_on_device_gb': on_dev, 'cache_gb': n * K * 4 / 1e9, 'arms': res}, f, indent=2, default=str)
    del sp, nb
    rnc._free()


def ft_context():
    import models.pretrained_vit as pv
    cfg = copy.deepcopy(FT)
    cfg['loader_fn'] = lambda seed: pv.cifar224_loaders(cfg['dataset'], seed, cfg['batch_size'], cfg['data_root'], cfg['num_workers'])
    root = os.path.join(cfg['base_out'], cfg['dataset'], cfg['arch_name'], rnc._out_tag())
    runs = {str(s): os.path.join(root, 'XE_baseline', f'seed_{s}') for s in rnc.SEL_SEEDS + rnc.CONF_SEEDS}
    return {'name': 'c100_vit_ft', 'root': root, 'cfg': cfg, 'ds': cfg['dataset'], 'mf': {'runs': {'XE_baseline': runs}}, 'onset': FT_ONSETS[0], 'lam': FT_LAMS[0]}


def ft_baseline(ctx, seed, onsets):
    c = ctx['cfg']
    root = rnc.run_path(ctx, 'XE_baseline', seed)
    if rnc.finished(root):
        print(f'[reuse] XE_baseline seed={seed}', flush=True)
        return root
    if rnc.SMOKE:
        c = {**c, 'num_epochs': max(onsets) + rnc.SMOKE_EPOCHS}
    root = run_dir(c['base_out'], ctx['ds'], c['arch_name'], rnc._out_tag(), 'XE_baseline', seed)
    kw = rn._base_kwargs(c, rnc._loaders(c, seed), seed, root, loss_params=rn._xe_params(), checkpoint_epochs=sorted(onsets), early_stopping=False, log_graph_maturity=False)
    train_one(rnc.TRACKER, label=f'[{ctx["name"]}] XE_baseline seed={seed}', epochs_to_train=c['num_epochs'], run_kwargs=kw)
    rnc._free()
    return root


def stage_vit_ft():
    ctx = ft_context()
    cfg = ctx['cfg']
    onsets, lams, alphas = (FT_ONSETS[:1], _mid(FT_LAMS), _mid(FT_ALPHAS)) if rnc.SMOKE else (FT_ONSETS, FT_LAMS, FT_ALPHAS)
    print(f'\n########## [{rnc._now()}] ViT-B/16 fine-tune, CIFAR-100 | onsets={onsets} λ={lams} α={alphas} ##########', flush=True)
    cosine = {'lambda_kind': 'cosine', 'lambda_decay': 'cosine'}
    m_ne = lambda c: f'nescal_e{c[0]}_l{rnc._tag(c[1])}'
    m_nc = lambda c: f'ncr_e{c[0]}_a{rnc._tag(c[1])}'
    lp_nc = lambda a: {'primary_loss': 'XE', 'secondary_loss': 'ncr', 'secondary_params': {'alpha': a, 'k': K_NN, 'temperature': NCR_T}}
    seeds_all = rnc.SEL_SEEDS + rnc.CONF_SEEDS
    xe = {s: ft_baseline(ctx, s, onsets) for s in seeds_all}
    cells_ne, cells_nc = [(e, l) for e in onsets for l in lams], [(e, a) for e in onsets for a in alphas]
    for s in rnc.SEL_SEEDS:
        for c in cells_ne:
            rnc.train(ctx, s, m_ne(c), rn._ncl_params(cfg, c[1]), c[0], f'[ft] NeSCal {c} seed={s}', **cosine)
        for c in cells_nc:
            rnc.train(ctx, s, m_nc(c), lp_nc(c[1]), c[0], f'[ft] NCR {c} seed={s}')
    win_ne, tab_ne = rnc.select_finished(ctx, cells_ne, m_ne, rnc.SEL_SEEDS)
    win_nc, tab_nc = rnc.select_finished(ctx, cells_nc, m_nc, rnc.SEL_SEEDS)
    print(f'  NeSCal winner {win_ne} {dict(tab_ne)}\n  NCR winner {win_nc} {dict(tab_nc)}', flush=True)
    rep_ne = {s: rnc.train(ctx, s, m_ne(win_ne), rn._ncl_params(cfg, win_ne[1]), win_ne[0], f'[ft] NeSCal winner seed={s}', **cosine) for s in seeds_all}
    rep_nc = {s: rnc.train(ctx, s, m_nc(win_nc), lp_nc(win_nc[1]), win_nc[0], f'[ft] NCR winner seed={s}') for s in seeds_all}
    for c, name in [(c, m_ne(c)) for c in cells_ne] + [(c, m_nc(c)) for c in cells_nc]:
        if name not in (m_ne(win_ne), m_nc(win_nc)):
            for s in rnc.SEL_SEEDS:
                d = rnc.run_path(ctx, name, s)
                for fn in ('model_state.pth', 'test_data.pkl', 'val_data.pkl', 'graph_data.pkl'):
                    if os.path.exists(os.path.join(d, fn)):
                        os.remove(os.path.join(d, fn))
                shutil.rmtree(os.path.join(d, 'plots'), ignore_errors=True)
    arms = {'XE_baseline': xe, 'ncr_FINAL': {s: r for s, r in rep_nc.items() if r}, 'nescal_FINAL': {s: r for s, r in rep_ne.items() if r}}
    per = {lab: {s: ad.run_metrics(d, stale_check=False) for s, d in dirs.items()} for lab, dirs in arms.items()}
    print(f'\n### CIFAR-100, ViT-B/16 fine-tuned at 224 px: NeSCal winner {win_ne}, NCR winner {win_nc}\n', flush=True)
    rnc.md_table({lab: {c: [m[c] for m in v.values() if m] for c in COLS} for lab, v in per.items()}, COLS, maximize=('acc',))
    print('\nNeSCal minus each arm, paired over seeds:', flush=True)
    for lab in ('XE_baseline', 'ncr_FINAL'):
        print(f'  {lab:14s} ' + ' | '.join(f'{c} {ad._fmt_ci(ad._ci({s: m[c] for s, m in per["nescal_FINAL"].items() if m}, {s: m[c] for s, m in per[lab].items() if m}))}' for c in ('nll', 'nll_ts', 'ece', 'ece_ts', 'acc')), flush=True)
    write_manifest(os.path.join(ctx['root'], 'ft_manifest.json'), {'winners': {'nescal': win_ne, 'ncr': win_nc}, 'selection': {'nescal': {str(k): v for k, v in tab_ne.items()}, 'ncr': {str(k): v for k, v in tab_nc.items()}},
                                                                   'reportable': {lab: {str(s): d for s, d in dirs.items()} for lab, dirs in arms.items()}, 'errors': rnc.ERRORS})


def _gather_worker(rank, world, backend, N, K, B, steps, port, out):
    import torch.distributed as dist
    dist.init_process_group(backend, init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=world)
    dev = torch.device(f'cuda:{rank}') if backend == 'nccl' else torch.device('cpu')
    sync = (lambda: torch.cuda.synchronize(dev)) if backend == 'nccl' else (lambda: None)

    def timed(x, reps):
        buf = [torch.zeros_like(x) for _ in range(world)]
        dist.barrier()
        t = time.time()
        for _ in range(reps):
            dist.all_gather(buf, x)
        sync()
        return time.time() - t
    per_step = timed(torch.rand(B, K, device=dev), steps) / steps
    chunk = 65_536
    shard_rows = N // world
    table_s = timed(torch.rand(chunk, K, device=dev), math.ceil(shard_rows / chunk))
    if rank == 0:
        with open(out, 'w') as f:
            json.dump({'backend': backend, 'world': world, 'n': N, 'K': K, 'batch': B, 'all_gather_per_step_ms': per_step * 1e3,
                       'steps_per_epoch': math.ceil(N / (B * world)), 'per_step_sync_s_per_epoch': per_step * math.ceil(N / (B * world)),
                       'epoch_end_table_sync_s': table_s, 'table_gb': N * K * 4 / 1e9}, f, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def stage_cache_bench():
    import torch.multiprocessing as mp
    root = os.path.join(BASE_OUT, 'cache_bench', rnc._out_tag())
    os.makedirs(root, exist_ok=True)
    out = os.path.abspath(os.path.join(root, 'cache_bench.json'))
    n_dev = torch.cuda.device_count()
    backend = 'nccl' if n_dev >= 2 else 'gloo'
    print(f'\n########## [{rnc._now()}] CACHE BENCH | visible GPUs={n_dev} backend={backend} (one machine) ##########', flush=True)
    N, steps = (BENCH_N // 100, 20) if rnc.SMOKE else (BENCH_N, BENCH_STEPS)
    mp.spawn(_gather_worker, args=(2, backend, N, BENCH_K, BENCH_B, steps, 29500 + os.getpid() % 500, out), nprocs=2, join=True)
    r = load_json(out)
    print(f'  {r}', flush=True)
    if backend == 'gloo':
        print('  NOTE: fewer than 2 GPUs visible, so this ran over CPU shared memory; do not report it as GPU interconnect timing.', flush=True)


def preflight(stages, jobs):
    problems = []
    if 'head' in stages:
        for j in jobs:
            b = HEAD_JOBS[j]['base']
            if not os.path.exists(b):
                problems.append(f'{j}: {b} missing; extract it first, see the docstring, or drop it with --jobs')
            else:
                print(f'[preflight] {j}: {b} {os.path.getsize(b) / 1e9:.1f} GB', flush=True)
        import importlib.util
        if importlib.util.find_spec('faiss') is None:
            problems.append('faiss not importable; pip install faiss-cpu')
    for st in ('head_baselines', 'head_mem'):
        if st in stages:
            for j in jobs:
                if not os.path.exists(HEAD_JOBS[j]['base']):
                    problems.append(f'{st}: {HEAD_JOBS[j]["base"]} missing')
                p = os.path.join(BASE_OUT, j, rnc.OUT_TAG, 'head_summary.json')
                if not os.path.exists(p):
                    problems.append(f'{st}: {p} missing; run --stages head --jobs {j} first')
    if 'vit_ft' in stages:
        import models.pretrained_vit as pv
        m = create_model(**{**FT['model_params'], 'weights': None})
        if not isinstance(m, pv.PretrainedViTB16):
            problems.append("models/resnet_architectures.py: create_model does not dispatch 'pretrained_vit_b16'")
        del m
    print('[preflight] ' + ('PROBLEMS:\n  ' + '\n  '.join(problems) if problems else 'all checks passed'), flush=True)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', nargs='+', default=['head'], choices=['head', 'vit_ft', 'cache_bench', 'head_baselines', 'head_mem'])
    ap.add_argument('--jobs', nargs='+', default=list(HEAD_JOBS), choices=list(HEAD_JOBS))
    ap.add_argument('--preflight_only', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='1 grid cell, 1+1 seeds, 2 epochs, 50k training samples, under <OUT_TAG>_smoke')
    a = ap.parse_args(argv)
    rnc.SMOKE = a.smoke
    if a.smoke:
        rnc.SEL_SEEDS, rnc.CONF_SEEDS = rnc.SEL_SEEDS[:1], rnc.CONF_SEEDS[:1]
    if preflight(a.stages, a.jobs):
        return 1
    if a.preflight_only:
        return 0
    rnc.TRACKER = RunTracker(10 ** 7, label='backbones')

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
            if st == 'head':
                for j in a.jobs:
                    guarded(f'head:{j}', stage_head, j, HEAD_JOBS[j])
            elif st in ('head_baselines', 'head_mem'):
                for j in a.jobs:
                    guarded(f'{st}:{j}', stage_head_baselines if st == 'head_baselines' else stage_head_mem, j, HEAD_JOBS[j])
            elif st == 'vit_ft':
                guarded('vit_ft', stage_vit_ft)
            else:
                guarded('cache_bench', stage_cache_bench)
    except SystemExit as e:
        print(f'!! {e}', flush=True)
        return 2
    print(f'\n=== backbones {a.stages} done in {(time.time() - t0) / 3600:.2f} h | errors: {ERRORS or rnc.ERRORS or "none"} ===', flush=True)
    return 0
