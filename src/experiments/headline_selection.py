import numpy as np
import experiments.nescal as run_nescal
from experiments.common import run_dir, RunTracker, train_one, select_by_val_nll

COSINE_DELTA = 0.005
POSTDROP_ONSET = 100
FALLBACK_ONSET = 95


def _run_nescal_arm(cfg, ts, seed, loaders, onset, lam, schedule='cosine', lr_schedule='multistep',
                    method=None, resume_epoch=None):
    base, ds, arch = cfg['base_out'], cfg['dataset'], cfg['arch_name']
    root = run_dir(base, ds, arch, ts, method, seed)
    resume = run_nescal._ckpt(base, cfg, ts, seed, resume_epoch) if resume_epoch is not None else None
    extra = dict(loss_params=run_nescal._ncl_params(cfg, lam), resume_from=resume,
                 activation_params={'strategy': 'fixed', 'fixed_epoch': onset},
                 lambda_kind=schedule, lambda_decay='cosine', lr_schedule=lr_schedule)
    epochs = cfg['num_epochs'] - (resume_epoch if resume_epoch is not None else 0)
    val, test, div, rp = train_one(RunTracker(epochs, label=method),
        label=f'{method} seed={seed} onset={onset} λ={lam} {lr_schedule}',
        epochs_to_train=epochs, run_kwargs=run_nescal._base_kwargs(cfg, loaders, seed, root, **extra))
    return (None if div else val['nll']), rp, div


def select_headline(cfg, ts):
    base, ds, arch = cfg['base_out'], cfg['dataset'], cfg['arch_name']
    sb = cfg['stage_b_seeds']; N = cfg['num_epochs']; onsets = cfg['candidate_onsets']
    log = {'grid': {}, 'decision': None}

    loaders_by_seed = {}
    for seed in sb:
        loaders_by_seed[seed] = run_nescal._loaders(cfg, seed)
        root = run_dir(base, ds, arch, ts, 'XE_baseline', seed)
        dense = sorted(set(onsets) | set(range(0, N, cfg['baseline_checkpoint_every'])))
        train_one(RunTracker(N, label=f'XE_baseline s{seed}'), label=f'XE_baseline seed={seed}',
            epochs_to_train=N, run_kwargs=run_nescal._base_kwargs(cfg, loaders_by_seed[seed], seed, root,
                loss_params=run_nescal._xe_params(), log_graph_maturity=True, checkpoint_epochs=dense))

    cos_scores = {}
    for seed in sb:
        for e in onsets:
            for lam in cfg['lambda_grid_cosine']:
                v, _, _ = _run_nescal_arm(cfg, ts, seed, loaders_by_seed[seed], e, lam,
                                          schedule='cosine', lr_schedule='multistep',
                                          method=f'ncl_sel_onset{e}_lam{lam}', resume_epoch=e)
                cos_scores.setdefault((e, lam), {})[seed] = v
    best_cos, cos_table = select_by_val_nll(cos_scores)
    best_onset, lam_cos = best_cos
    log['grid'] = {str(k): v for k, v in cos_table.items()}
    print(f'  MultiStep winner: onset={best_onset}, λ={lam_cos}, val_nll={cos_table[best_cos]:.4f}', flush=True)

    lr_schedule = 'multistep'
    if best_onset >= POSTDROP_ONSET:
        print(f'=== onset {best_onset} >= {POSTDROP_ONSET} -> cosine-LR confound check ===', flush=True)
        cos_lr_scores = {}
        for seed in sb:
            v, _, div = _run_nescal_arm(cfg, ts, seed, loaders_by_seed[seed], best_onset, lam_cos,
                                        schedule='cosine', lr_schedule='cosine',
                                        method=f'ncl_cosLR_onset{best_onset}_lam{lam_cos}', resume_epoch=None)
            cos_lr_scores[seed] = v
        finite = [v for v in cos_lr_scores.values() if v is not None]
        ms_val = cos_table[best_cos]
        if not finite:
            print('  cosine-LR diverged on all seeds -> cannot validate; FALLING BACK to onset 95.', flush=True)
            best_onset = FALLBACK_ONSET; delta = None; verdict = 'cosine_diverged_fallback'
        else:
            cos_lr_val = float(np.mean(finite)); delta = cos_lr_val - ms_val
            print(f'  cosine-LR val_nll={cos_lr_val:.4f} vs MultiStep={ms_val:.4f} (Δ={delta:+.4f})', flush=True)
            if delta < COSINE_DELTA:
                verdict = 'multistep_kept_confound_clean'
                print(f'  Δ<{COSINE_DELTA}: onset {best_onset} win is NOT an LR artifact -> KEEP MultiStep headline.', flush=True)
            else:
                verdict = 'fallback_to_95'
                print(f'  Δ>={COSINE_DELTA}: onset-{best_onset} win was schedule-driven -> FALL BACK to onset {FALLBACK_ONSET}.', flush=True)
                best_onset = FALLBACK_ONSET
        log['decision'] = {'verdict': verdict, 'delta_val_nll': delta, 'lambda': lam_cos,
                           'onset_selected': best_cos[0], 'onset_final': best_onset}
    else:
        log['decision'] = {'verdict': 'winner_pre_drop', 'onset_final': best_onset, 'lambda': lam_cos}
        print(f'  winner onset {best_onset} < {POSTDROP_ONSET}: no confound check needed.', flush=True)

    return best_onset, lam_cos, lr_schedule, log
