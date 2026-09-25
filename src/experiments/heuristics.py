import os, argparse, numpy as np
from experiments.common import run_dir, RunTracker, train_one, write_manifest, load_json, load_pkl
from data.image_datasets import load_dataset
import experiments.nescal as run_nescal

def _baseline_dir(run_root, seed): return os.path.join(run_root, 'XE_baseline', f'seed_{seed}')
def _ckpt(run_root, seed, epoch):
    return os.path.join(_baseline_dir(run_root, seed), 'checkpoints', f'epoch_{epoch}.pt')

def _gate_epochs(diag, rules):
    hist = sorted(diag.get('heuristics_history', []), key=lambda r: r['epoch'])
    ent = [(r['epoch'], r.get('mean_entropy_train')) for r in hist if r.get('mean_entropy_train') is not None]
    prb = [(r['epoch'], r.get('knn_probe_acc')) for r in hist if r.get('knn_probe_acc') is not None]
    cwc = [(r['epoch'], r.get('cwcc')) for r in hist if r.get('cwcc') is not None]
    below = lambda seq, thr: next((e for e, v in seq if v <= thr), None)
    above = lambda seq, thr: next((e for e, v in seq if v >= thr), None)
    out = {}
    if ent: out['entropy_drop'] = below(ent, (1 - rules['entropy_drop']) * ent[0][1])
    if prb: out['probe'] = above(prb, rules['probe_frac'] * max(v for _, v in prb))
    if cwc: out['cwcc'] = below(cwc, rules['cwcc_frac'] * cwc[0][1])
    return out

def _snap(seed_run_root, seed, epoch, every, N):
    cands = [e for e in range(0, N, every) if e <= epoch]
    return max(cands) if cands else 0

def run(run_root, cfg=run_nescal.CONFIG):
    ds, arch, base = cfg['dataset'], cfg['arch_name'], cfg['base_out']
    mf = load_json(os.path.join(run_root, 'run_manifest.json'))
    E, lam, N = mf['best_onset'], mf['lambda_cosine'], cfg['num_epochs']
    every, seeds = cfg['baseline_checkpoint_every'], cfg['heuristic_seeds']
    ts = run_root.rstrip('/').split(os.sep)[-1]
    rules, js_target = cfg['heuristics']['onset'], cfg['heuristics']['lambda_js_target']
    combined = cfg.get('run_combined_heuristic', False)

    tr = RunTracker(len(seeds) * (3 + 1 + (3 if combined else 0)) * N, label=f'{ds}/{arch}:hreg')
    manifest = {'run_root': run_root, 'best_onset': E, 'lambda': lam, 'js_target': js_target,
                'gate_epochs': {}, 'reportable': {}, 'diverged': [], 'errored': []}
    print(f'=== Heuristic-regret {ds}/{arch} | grid E*={E} λ*={lam} | js_target={js_target} | combined={combined} ===',
          flush=True)

    for seed in seeds:
        loaders = run_nescal._loaders(cfg, seed)
        diag = load_pkl(os.path.join(_baseline_dir(run_root, seed), 'train_time_diagnostics.pkl'))
        gates = _gate_epochs(diag, rules); manifest['gate_epochs'][seed] = gates

        plan = []
        for h, ep in gates.items():
            if ep is None:
                print(f'  seed={seed}: gate {h} never fired — recorded, skipped.', flush=True); continue
            e_s = _snap(run_root, seed, ep, every, N)
            plan.append((f'hreg_onset_{h}', e_s, lam, None))
            if combined:
                plan.append((f'hreg_combined_{h}', e_s, None, js_target))
        plan.append(('hreg_lambda', E, None, js_target))

        for label, onset, lam_v, js_t in plan:
            try:
                lp = run_nescal._ncl_params(cfg, lam_v if lam_v is not None else 1.0)
                root = run_dir(base, ds, arch, ts, label, seed)
                extra = dict(loss_params=lp, resume_from=_ckpt(run_root, seed, onset),
                             activation_params={'strategy': 'fixed', 'fixed_epoch': onset},
                             lambda_kind='cosine', lambda_decay='cosine', lambda_from_js_target=js_t)
                _, _, div, rp = train_one(tr, label=f'{label} seed={seed} onset={onset} '
                                                f'{("λ=" + str(lam_v)) if lam_v else "auto-λ"}',
                                          epochs_to_train=N - onset,
                                          run_kwargs=run_nescal._base_kwargs(cfg, loaders, seed, root, **extra))
                manifest['reportable'].setdefault(label, {})[seed] = rp
                if div: manifest['diverged'].append({'run': label, 'seed': seed})
            except Exception as e:
                print(f'  !! {label} seed={seed} ERRORED ({type(e).__name__}: {e}) — skipping, continuing.',
                      flush=True)
                manifest['errored'].append({'run': label, 'seed': seed, 'error': f'{type(e).__name__}: {e}'})
            write_manifest(os.path.join(run_root, 'heuristic_manifest.json'), manifest)

    print('=== Heuristic-regret complete ===', flush=True)
