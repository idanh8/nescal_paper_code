import os, sys, glob, pickle
import numpy as np


METHODS = ('ncl_FINAL_cosine', 'XE_baseline')
REF_ARM = 'ncl_FINAL_cosine'


def _load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def _hist(seed_dir):
    p = os.path.join(seed_dir, 'train_time_diagnostics.pkl')
    if not os.path.exists(p):
        return None, None
    d = _load(p)
    return d, sorted(d.get('heuristics_history', []), key=lambda r: r.get('epoch', 0))


def _at(hist, ep):
    if not hist:
        return {}
    later = [r for r in hist if r.get('epoch', 0) >= ep]
    return later[0] if later else hist[-1]


def _ref_epoch(root, ref_arm):
    eps = []
    for sd in sorted(glob.glob(os.path.join(root, ref_arm, 'seed_*'))):
        d, _ = _hist(sd)
        if d and d.get('activation_epoch') is not None:
            eps.append(d['activation_epoch'])
    return int(np.mean(eps)) if eps else None


def _m(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def _f(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '     -'


def main(roots, show_all=False, ref_arm=REF_ARM):
    hdr = (f'{"setting":<40} {"method":<20} {"ref_ep":>6} {"probe":>7} {"pr/acc":>7} {"sil":>7} '
           f'{"edge_js":>8} {"lambda*":>8} {"acc":>7} {"ece":>7}   traj(first/mid/last)')
    print(hdr); print('-' * len(hdr))
    for root in roots:
        label = '/'.join(root.rstrip('/').split(os.sep)[-3:])
        ref_ep = _ref_epoch(root, ref_arm)
        if ref_ep is None:
            print(f'{label:<40} !! no activation_epoch for {ref_arm} -- skipping run', flush=True)
            continue
        mdirs = sorted(glob.glob(os.path.join(root, '*'))) if show_all else \
                [os.path.join(root, m) for m in METHODS]
        for mdir in mdirs:
            if not os.path.isdir(mdir):
                continue
            rows = []
            for sd in sorted(glob.glob(os.path.join(mdir, 'seed_*'))):
                d, h = _hist(sd)
                if d is None:
                    continue
                rec = _at(h, ref_ep)
                th = d.get('theory', {}) or {}
                tp = os.path.join(sd, 'test_results.pkl')
                t = _load(tp) if os.path.exists(tp) else {}
                traj = [r.get('knn_probe_acc') for r in h] if h else []
                rows.append({'probe': rec.get('knn_probe_acc'), 'sil': rec.get('silhouette'),
                             'edge_js': th.get('mean_edge_js_at_activation'), 'lam': th.get('lambda_star'),
                             'acc': t.get('accuracy'), 'ece': t.get('ece'),
                             'f': traj[0] if traj else None,
                             'md': traj[len(traj) // 2] if traj else None,
                             'l': traj[-1] if traj else None})
            if not rows:
                continue
            probe, acc = _m([r['probe'] for r in rows]), _m([r['acc'] for r in rows])
            ratio = probe / acc if (np.isfinite(probe) and np.isfinite(acc) and acc > 0) else float('nan')
            traj = f"{_f(_m([r['f'] for r in rows]),2)}/{_f(_m([r['md'] for r in rows]),2)}/{_f(_m([r['l'] for r in rows]),2)}"
            print(f'{label:<40} {os.path.basename(mdir):<20} {ref_ep:>6} {_f(probe):>7} {_f(ratio,3):>7} '
                  f'{_f(_m([r["sil"] for r in rows]),3):>7} {_f(_m([r["edge_js"] for r in rows])):>8} '
                  f'{_f(_m([r["lam"] for r in rows]),3):>8} {_f(acc):>7} {_f(_m([r["ece"] for r in rows])):>7}   {traj}')
