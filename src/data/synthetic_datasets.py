import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_datasets import IndexedImageDataset


class TensorVectorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float()
        self.y = y.long()
        self.targets = self.y.tolist()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], int(self.y[idx])


def _class_centers(num_classes, dim, sep, close_frac, close_sep_scale, rng):
    centers = np.zeros((num_classes, dim), dtype=np.float64)
    n_close = int(round(close_frac * num_classes))
    is_close = np.zeros(num_classes, dtype=bool)
    is_close[:n_close] = True


    far_idx = np.where(~is_close)[0]; close_idx = np.where(is_close)[0]
    for j, c in enumerate(far_idx):
        ang = 2 * np.pi * j / max(len(far_idx), 1)
        centers[c, 0] = sep * np.cos(ang); centers[c, 1] = sep * np.sin(ang)


    r_close = sep * close_sep_scale
    offset = np.array([sep * 1.6, 0.0])
    for j, c in enumerate(close_idx):
        ang = 2 * np.pi * j / max(len(close_idx), 1)
        centers[c, 0] = offset[0] + r_close * np.cos(ang)
        centers[c, 1] = offset[1] + r_close * np.sin(ang)
    return centers, is_close


def _make_blobs(num_classes, dim, n, sep, noise_scale, hetero_mode, hetero_strength, close_frac, close_sep_scale, class_temp_hi, rng):
    centers, is_close = _class_centers(num_classes, dim, sep, close_frac, close_sep_scale, rng)


    s = np.ones(num_classes)
    if hetero_mode == 'posterior_temp':
        n_hot = int(round(close_frac * num_classes))
        s[:n_hot] = 1.0 + hetero_strength * (class_temp_hi - 1.0)

    y = rng.integers(0, num_classes, size=n)
    X = centers[y] + noise_scale * rng.standard_normal(size=(n, dim))


    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    logpi = -d2 / (2.0 * (s[None, :] ** 2)) - dim * np.log(s[None, :])
    logpi -= logpi.max(1, keepdims=True)
    PI = np.exp(logpi); PI /= PI.sum(1, keepdims=True)
    return X.astype(np.float64), y.astype(np.int64), PI.astype(np.float64), is_close


def _apply_label_noise(y, PI, is_close, hetero_mode, hetero_strength, rng):
    num_classes = PI.shape[1]
    y_obs = y.copy()

    noisy_pt = is_close[y]
    flip = noisy_pt & (rng.random(len(y)) < hetero_strength)
    y_obs[flip] = rng.integers(0, num_classes, size=int(flip.sum()))


    PI_eval = PI.copy()
    if hetero_strength > 0:
        a = hetero_strength
        unif = np.full(num_classes, 1.0 / num_classes)
        PI_eval[noisy_pt] = (1 - a) * PI[noisy_pt] + a * unif[None, :]
        PI_eval /= PI_eval.sum(1, keepdims=True)
    return y_obs, PI_eval


def _plot_2d(X, y, is_close, save_path, title):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    K = int(y.max()) + 1
    cmap = plt.cm.tab10(np.linspace(0, 1, max(K, 10)))
    for k in range(K):
        m = y == k
        marker = 'x' if is_close[k] else 'o'
        ax.scatter(X[m, 0], X[m, 1], s=8, color=cmap[k % len(cmap)], marker=marker,
                   alpha=0.6, label=f'c{k}{" (confusable)" if is_close[k] else ""}')
    ax.set_title(title); ax.set_xlabel('dim 0'); ax.set_ylabel('dim 1')
    ax.legend(fontsize=6, ncol=2, markerscale=1.5, loc='best')
    ax.set_aspect('equal', 'datalim')
    fig.tight_layout(); fig.savefig(save_path, dpi=140, bbox_inches='tight'); plt.close(fig)


def make_synthetic(num_classes=10, dim=20, n=6000, sep=8.0, noise_scale=1.0,
                   heterogeneity_mode='separation', heterogeneity_strength=0.4,
                   close_frac=0.5, close_sep_scale=0.35, class_temp_hi=3.0,
                   val_split=0.2, test_split=0.2, seed=42, plot_path=None):
    if heterogeneity_mode == 'label_noise':
        return make_synthetic_notebook(num_classes=num_classes, dim=dim, n=n, sep=sep,
                                       noise_scale=noise_scale, heterogeneity_strength=heterogeneity_strength,
                                       val_split=val_split, test_split=test_split, seed=seed, plot_path=plot_path)
    rng = np.random.default_rng(seed)
    X, y_true, PI, is_close = _make_blobs(num_classes, dim, n, sep, noise_scale, heterogeneity_mode,
                                          heterogeneity_strength, close_frac, close_sep_scale,
                                          class_temp_hi, rng)
    y_obs, PI_eval = _apply_label_noise(y_true, PI, is_close, heterogeneity_mode,
                                        heterogeneity_strength, rng)


    idx = rng.permutation(n)
    n_test = int(test_split * n); n_val = int(val_split * n)
    test_idx = idx[:n_test]; val_idx = idx[n_test:n_test + n_val]; train_idx = idx[n_test + n_val:]

    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        _plot_2d(X, y_obs, is_close, plot_path,
                 f'blobs K={num_classes} d={dim} sep={sep} mode={heterogeneity_mode} '
                 f'H={heterogeneity_strength} (seed {seed})')

    return {'X': X, 'y': y_obs, 'y_true': y_true, 'PI': PI, 'PI_eval': PI_eval,
            'is_close': is_close, 'point_is_noisy': is_close[y_true],
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
            'num_classes': num_classes, 'dim': dim, 'seed': seed}


def load_synthetic(val_split=0.2, batch_size=256, seed=42, data_root='./data', num_workers=0, sidecar_dir=None, plot_path=None, **blob_kwargs):
    d = make_synthetic(val_split=val_split, seed=seed, plot_path=plot_path, **blob_kwargs)
    X = torch.from_numpy(d['X']); y = torch.from_numpy(d['y'])
    tr_i, va_i, te_i = d['train_idx'], d['val_idx'], d['test_idx']

    full = TensorVectorDataset(X, y)
    train_set = Subset(full, tr_i.tolist())
    val_set = Subset(full, va_i.tolist())
    test_set = Subset(full, te_i.tolist())
    embed_set = Subset(full, tr_i.tolist())

    common = dict(num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(IndexedImageDataset(train_set), batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(IndexedImageDataset(val_set), batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(IndexedImageDataset(test_set), batch_size=batch_size, shuffle=False, **common)
    embedding_loader = DataLoader(IndexedImageDataset(embed_set), batch_size=batch_size, shuffle=False, **common)

    if sidecar_dir is not None:
        os.makedirs(sidecar_dir, exist_ok=True)
        np.savez(os.path.join(sidecar_dir, f'eta_sidecar_seed{seed}.npz'),
                 PI_eval=d['PI_eval'], PI_clean=d['PI'], y=d['y'], y_true=d['y_true'],
                 point_is_noisy=d['point_is_noisy'], is_close=d['is_close'],
                 train_idx=tr_i, val_idx=va_i, test_idx=te_i,
                 num_classes=d['num_classes'], dim=d['dim'], seed=seed)

    class_names = [f'c{k}' for k in range(d['num_classes'])]
    return train_loader, val_loader, test_loader, embedding_loader, class_names

def make_synthetic_notebook(num_classes=10, dim=20, n=6000, sep=8.0, noise_scale=1.0, heterogeneity_strength=0.3, noisy_frac=0.5, val_split=0.2, test_split=0.2, seed=42, plot_path=None, **_ignored):
    rng = np.random.RandomState(seed)
    K, d = num_classes, dim
    ang = 2 * np.pi * np.arange(K) / K
    mu = np.zeros((K, d))
    mu[:, 0], mu[:, 1] = sep * np.cos(ang), sep * np.sin(ang)
    y_true = rng.randint(0, K, n)
    X = mu[y_true] + noise_scale * rng.randn(n, d)
    rho = np.where(np.arange(K) < int(K * noisy_frac), heterogeneity_strength, 0.0)
    y = y_true.copy()
    flip = rng.rand(n) < rho[y_true]
    y[flip] = rng.randint(0, K, int(flip.sum()))

    logp = -0.5 * ((X[:, None, :] - mu[None]) ** 2).sum(-1)
    pi = np.exp(logp - logp.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)

    PI_eval = pi * (1 - rho)[None, :] + (pi * rho[None, :]).sum(1, keepdims=True) / K
    is_close = np.arange(K) < int(K * noisy_frac)
    point_is_noisy = is_close[y_true]

    idx = rng.permutation(n)
    n_test = int(test_split * n); n_val = int(val_split * n)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        _plot_2d(X, y, is_close, plot_path,
                 f'notebook blobs K={K} d={d} sep={sep} hetero={heterogeneity_strength} (seed {seed})')

    return {'X': X.astype(np.float64), 'y': y.astype(np.int64), 'y_true': y_true.astype(np.int64),
            'PI': pi.astype(np.float64), 'PI_eval': PI_eval.astype(np.float64),
            'is_close': is_close, 'point_is_noisy': point_is_noisy,
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
            'num_classes': K, 'dim': d, 'seed': seed}
