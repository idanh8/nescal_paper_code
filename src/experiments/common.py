import os, sys, time, json, datetime


_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC not in sys.path: sys.path.insert(0, _SRC)

from handlers.trainer import train_model
from utils.general_utils import DivergedRun

SELECT_KEY = 'nll'

def timestamp(): return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def run_dir(base, dataset, arch, ts, method, seed):
    d = os.path.join(base, dataset, arch, ts, method, f'seed_{seed}')
    os.makedirs(d, exist_ok=True); return d

class RunTracker:
    def __init__(self, total_epochs_planned, label='suite'):
        self.total = max(total_epochs_planned, 1); self.done_epochs = 0
        self.sec_per_epoch = None; self.t0 = time.time(); self.label = label; self.n_runs = 0

    def _fmt_eta(self, secs):
        return (datetime.datetime.now() + datetime.timedelta(seconds=secs)).strftime('%a %H:%M')

    def record(self, epochs_trained, elapsed_s):
        self.n_runs += 1; self.done_epochs += epochs_trained
        rate = elapsed_s / max(epochs_trained, 1)
        self.sec_per_epoch = rate if self.sec_per_epoch is None else 0.7 * self.sec_per_epoch + 0.3 * rate
        remaining = (self.total - self.done_epochs) * self.sec_per_epoch
        pct = 100.0 * self.done_epochs / self.total
        print(f'[{self.label}] run {self.n_runs} done ({epochs_trained} ep, {elapsed_s/60:.1f} min).')

def train_one(tracker, *, label, epochs_to_train, run_kwargs):
    print(f'\n>>> {label}  (training ~{epochs_to_train} epochs)', flush=True)
    t = time.time()
    try:
        _, val_results, test_results, _ = train_model(**run_kwargs)
    except DivergedRun as e:
        elapsed = time.time() - t
        print(f'  !! DIVERGED at epoch {e.first_epoch + 1} — voiding {label}, continuing.', flush=True)
        tracker.record(min(epochs_to_train, getattr(e, 'first_epoch', epochs_to_train)), elapsed)
        return None, None, True, run_kwargs['root_path']
    tracker.record(epochs_to_train, time.time() - t)

    return val_results, test_results, False, run_kwargs['root_path']

def select_by_val_nll(scores):
    table = {}
    for key, per_seed in scores.items():
        vals = [v for v in per_seed.values() if v is not None]
        table[key] = float('inf') if (len(vals) != len(per_seed) or not vals) else sum(vals) / len(vals)
    best = min(table, key=table.get)
    return best, table

def write_manifest(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2, default=str)
    print(f'[manifest] wrote {path}', flush=True)


import numpy as np, pickle

def _jsonable(o):
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.ndarray): return None
    return o

def save_results(root_path, val_results, test_results):
    """Insurance so the report never hard-depends on posthoc having run."""
    with open(os.path.join(root_path, 'results.json'), 'w') as f:
        json.dump({'val': _jsonable(val_results), 'test': _jsonable(test_results)}, f, indent=2)

def load_json(path):
    with open(path) as f: return json.load(f)

def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)
