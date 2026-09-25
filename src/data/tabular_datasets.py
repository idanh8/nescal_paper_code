import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder

DATASETS = {

    'helena': ('openml', 41169, None),
    'covertype': ('covtype', None, 100_000),
     'jannis': ('openml', 41168, None),
}


def _fetch(name):
    kind, oid, sub = DATASETS[name]
    if kind == 'openml':
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=oid, as_frame=True, parser='auto')
        X = d.data.to_numpy(dtype=np.float32)
        y = LabelEncoder().fit_transform(np.asarray(d.target).ravel())
    else:
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype()
        X, y = d.data.astype(np.float32), (d.target - 1).astype(int)
    if sub is not None and len(y) > sub:
        idx = np.random.RandomState(0).choice(len(y), sub, replace=False)
        X, y = X[idx], y[idx]
        print(f'[tabular] {name}: subsampled to {sub} rows (graph tractability)', flush=True)
    print(f'[tabular] {name}: X={X.shape}, {len(np.unique(y))} classes', flush=True)
    return X, y.astype(np.int64)


def _split(X, y, frac, seed):
    try:
        return train_test_split(np.arange(len(y)), test_size=frac, random_state=seed, stratify=y)
    except ValueError:
        print('[tabular] stratified split not possible (rare class) -- using random split', flush=True)
        return train_test_split(np.arange(len(y)), test_size=frac, random_state=seed)


class _IndexedTabular(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], i


PREPROC = {'helena': 'standard', 'covertype': 'quantile', 'jannis': 'quantile'}

def build_tabular(name='helena', test_frac=0.2, val_frac=0.2, split_seed=0,
                  batch_size=512, num_workers=4):
    X, y = _fetch(name)
    n_classes, in_dim = int(y.max()) + 1, X.shape[1]
    trpool_i, te_i = _split(X, y, test_frac, split_seed)

    def loader_fn(seed):
        sub_tr, sub_va = _split(X[trpool_i], y[trpool_i], val_frac, seed)
        tr_i, va_i = trpool_i[sub_tr], trpool_i[sub_va]
        if PREPROC.get(name, 'quantile') == 'standard':
            qt = StandardScaler().fit(X[tr_i])
        else:
            qt = QuantileTransformer(output_distribution='normal',
                                     n_quantiles=min(1000, len(tr_i)), random_state=seed)
            rng = np.random.RandomState(seed)
            qt.fit(X[tr_i] + rng.normal(0, 1e-3, X[tr_i].shape).astype(np.float32))
        Xtr = qt.transform(X[tr_i]).astype(np.float32)
        Xva, Xte = qt.transform(X[va_i]).astype(np.float32), qt.transform(X[te_i]).astype(np.float32)
        tr_ds = _IndexedTabular(Xtr, y[tr_i])
        va_ds, te_ds = _IndexedTabular(Xva, y[va_i]), _IndexedTabular(Xte, y[te_i])
        mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers)
        names = [str(c) for c in range(n_classes)]
        return mk(tr_ds, True), mk(va_ds, False), mk(te_ds, False), mk(tr_ds, False), names

    return loader_fn, n_classes, in_dim
