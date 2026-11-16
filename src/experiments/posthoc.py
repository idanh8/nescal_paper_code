import os, pickle
from experiments.common import load_json
from data.image_datasets import load_dataset
from handlers.trainer import Trainer, _embed_probs_labels
from posthoc.calibrators import apply_posthoc_suite

POSTHOC = ('ts', 'local', 'dc', 'dac', 'lore')


def _reportable_runs(run_root):
    out = {}
    for mf in ('run_manifest.json', 'competitors_manifest.json', 'mixup_manifest.json', 'ncr_manifest.json'):
        p = os.path.join(run_root, mf)
        if os.path.exists(p):
            for label, per_seed in load_json(p).get('reportable', {}).items():
                out.setdefault(label, {}).update({int(k): v for k, v in per_seed.items()})
    return out

def run(run_root, dataset, arch, batch_size=256, data_root='./data', num_workers=4, num_bins=15, loader_fn=None,
        only_labels=None, skip_existing=False):
    runs = _reportable_runs(run_root)
    if only_labels is not None: runs = {k: v for k, v in runs.items() if k in only_labels}
    total = sum(len(v) for v in runs.values()); done = 0
    print(f'=== Posthoc over {total} runs under {run_root} ===', flush=True)
    for label, per_seed in runs.items():
        for seed, rp in per_seed.items():
            done += 1
            if not os.path.exists(os.path.join(rp, 'model_state.pth')):
                print(f'[{done}/{total}] SKIP {label} seed={seed} (no model)', flush=True); continue
            if skip_existing and os.path.exists(os.path.join(rp, 'posthoc_results.pkl')):
                print(f'[{done}/{total}] {label} seed={seed}: posthoc already done', flush=True); continue
            m = Trainer.load_model(rp)

            if loader_fn is not None:
                _, _, _, emb_loader, _ = loader_fn(seed)
            else:
                _, _, _, emb_loader, _ = load_dataset(dataset, val_split=0.1, batch_size=batch_size, seed=seed,
                                                      data_root=data_root, num_workers=num_workers)
            train_feats, _, train_lbls = _embed_probs_labels(m.model, emb_loader, m.device, 'final')
            res = apply_posthoc_suite(
                m.all_val_logits, m.all_val_labels, m.all_val_features,
                m.all_test_logits, m.all_test_labels, m.all_test_features,
                calibrators=POSTHOC, seed=seed, num_bins=num_bins, lce_projection=m.lce_projection,
                train_features=train_feats, train_labels=train_lbls,
                calibrator_kwargs={'dac': {'knn_k': 200 if m.model_params['num_classes'] >= 100 else 50}},
                include_local_metrics=True)
            with open(os.path.join(rp, 'posthoc_results.pkl'), 'wb') as f: pickle.dump(res, f)
            print(f'[{done}/{total}] {label} seed={seed}: '
                  f"pre ECE {res['pre'].get('ece'):.4f} -> "
                  f"{ {a: round(res['post'][a].get('ece', float('nan')), 4) for a in POSTHOC if a in res['post']} }",
                  flush=True)
    print('=== Posthoc complete ===', flush=True)
