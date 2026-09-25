import pytz
import datetime
import os
import math
import random
import torch
import numpy as np

def make_lambda_schedule(kind, lam_peak, onset, E, off_frac=0.5, decay='cosine'):
    """f(epoch) -> lambda, for epochs in [onset+1, E).
       constant: lam_peak throughout.
       window  : lam_peak until off_epoch = onset + round(off_frac*(E-onset)), then 0 (hard suppression).
       decay   : lam_peak at onset decaying to 0 at E ('linear' or 'cosine')."""
    span = max(E - onset, 1)
    off_epoch = onset + int(round(off_frac * span))
    def f(epoch):
        if kind == 'constant':
            return float(lam_peak)
        if kind == 'window':
            return float(lam_peak) if epoch < off_epoch else 0.0
        t = (epoch - onset) / span
        if decay == 'linear':
            return float(lam_peak) * max(0.0, 1.0 - t)
        return float(lam_peak) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    f.off_epoch = off_epoch
    return f


def get_timestamp():
  gmt_plus_3 = pytz.timezone('Etc/GMT-3')
  now = datetime.datetime.now(gmt_plus_3)
  timestamp = now.strftime('%Y-%m-%d-%H%M')
  return timestamp


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class DivergedRun(RuntimeError):
    def __init__(self, first_epoch, last_value, consecutive):
        self.first_epoch = first_epoch
        self.last_value = last_value
        self.consecutive = consecutive
        super().__init__(f'Training diverged: validation loss non-finite for {consecutive} consecutive '
                         f'epochs (first at epoch {first_epoch + 1}, last value {last_value}).')
