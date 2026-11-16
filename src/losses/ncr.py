import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.nescal import PreNeighborhoodConsistencyLoss, js_divergence_torch


def _drop_self(I, k):
    rows = np.arange(I.shape[0])[:, None]
    valid = (I != rows) & (I >= 0)
    order = np.argsort(~valid, axis=1, kind='stable')[:, :k]
    out = np.take_along_axis(I, order, axis=1)
    kept = np.take_along_axis(valid, order, axis=1)
    if not kept.all():
        out = np.where(kept, out, out[:, :1])
    return out


def recall_at_k(x, I, k, n_sample=2000, chunk=256, seed=0):
    n = x.shape[0]
    idx = np.random.RandomState(seed).choice(n, min(n_sample, n), replace=False)
    hits = 0
    for s in range(0, len(idx), chunk):
        batch = idx[s:s + chunk]
        sims = x[batch] @ x.T
        sims[np.arange(len(batch)), batch] = -np.inf
        exact = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        hits += sum(len(set(exact[i].tolist()) & set(I[batch[i]].tolist())) for i in range(len(batch)))
    return hits / (len(idx) * k)


def ann_knn(x, k, backend, nprobe=None, hnsw_m=32, ef_search=64, query_sample=None):
    import faiss
    n, d = x.shape
    t0 = time.time()
    if backend == 'faiss_ivf':
        nlist = max(1, int(4 * math.sqrt(n)))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        train_ids = np.random.RandomState(0).choice(n, min(n, 64 * nlist), replace=False)
        index.train(x[train_ids])
        index.add(x)
        index.nprobe = nprobe or max(1, nlist // 32)
    elif backend == 'faiss_hnsw':
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 2 * ef_search
        index.add(x)
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"unknown graph_backend '{backend}'")
    build_s = time.time() - t0
    t1 = time.time()
    if query_sample is not None and query_sample < n:
        ids = np.random.RandomState(0).choice(n, query_sample, replace=False)
        index.search(x[ids], k + 1)
        return None, {'backend': backend, 'build_s': build_s, 'search_s': (time.time() - t1) * n / query_sample,
                      'search_extrapolated_from': int(query_sample), 'n': int(n), 'd': int(d)}
    _, I = index.search(x, k + 1)
    return _drop_self(I, k), {'backend': backend, 'build_s': build_s, 'search_s': time.time() - t1, 'n': int(n), 'd': int(d)}


def _minibatch_topk(features, k, grad=False):
    v = F.normalize(features if grad else features.detach(), p=2, dim=1)
    sim = v @ v.T
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return sim.masked_fill(eye, -float('inf')).topk(k, dim=1)


class NCRLoss(PreNeighborhoodConsistencyLoss):
    requires_features = True
    default_lambda_reg = 1.0

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0, k: int = 10, grad_through_features: bool = True,
                 form: str = 'code', **kw):
        super().__init__(lambda_reg=1.0, divergence_type='js', weight_type='topk', k=k, **kw)
        assert 0.0 < alpha <= 1.0, 'alpha must be in (0, 1]'
        assert form in ('code', 'paper'), "form: 'code' = released scenic loss, 'paper' = Eq. 3 as printed"
        self.alpha, self.temperature, self.grad_through_features, self.form = alpha, temperature, grad_through_features, form
        self.last_ncr = None

    def ncr_term(self, logits, features):
        B = logits.shape[0]
        k = min(self.k, B - 1)
        log_p = F.log_softmax(logits / self.temperature, dim=1)
        p = log_p.exp()
        s, idx = _minibatch_topk(features, k, grad=self.grad_through_features)
        s = s.clamp_min(0.0)
        total = s.sum(dim=1, keepdim=True)
        w = torch.where(total > 0, s / total.clamp_min(1e-12), torch.full_like(s, 1.0 / k))
        target = (w.unsqueeze(-1) * p[idx]).sum(dim=1)
        log_t = target.clamp_min(1e-12).log()
        if self.form == 'code':
            return self.temperature ** 2 * (target * (log_t - log_p)).sum(dim=1).mean()
        return (p * (log_p - log_t)).sum(dim=1).mean()

    def forward(self, logits, labels, node_indices=None, features=None, **kwargs):
        if not self.active or logits.shape[0] < 2:
            return torch.tensor(0.0, device=logits.device)
        if features is None:
            raise ValueError('NCRLoss needs features: apply the train_epoch edit in handlers/trainer.py')
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits32, feats32 = logits.float(), features.float()
            ncr = self.ncr_term(logits32, feats32)
            self.last_ncr = float(ncr.detach())
            return self.alpha * (ncr - F.cross_entropy(logits32, labels))


class NeighborVariantLoss(PreNeighborhoodConsistencyLoss):

    def __init__(self, target: str = 'edge', graph: str = 'frozen', graph_backend: str = 'exact', stats_path: str = None,
                 cache_sync_epochs: int = None, drift_path: str = None, **kw):
        super().__init__(**kw)
        assert target in ('edge', 'mixture') and graph in ('frozen', 'minibatch')
        self.target, self.graph, self.graph_backend, self.stats_path = target, graph, graph_backend, stats_path
        self.cache_sync_epochs, self.drift_path = cache_sync_epochs, drift_path
        self.precomputed_neighbors = None
        self._pending, self._seen_sync = None, 0
        self._drift_hist, self._drift_acc = [], [0.0, 0.0, 0, 0]

    @property
    def requires_features(self):
        return self.graph == 'minibatch'

    def _edge_divergence(self, pi_flat, pj_flat):
        if self.divergence_type == 'skl':
            a, b = pi_flat.clamp_min(1e-10), pj_flat.clamp_min(1e-10)
            la, lb = torch.log2(a), torch.log2(b)
            return 0.5 * ((a * (la - lb)).sum(dim=1) + (b * (lb - la)).sum(dim=1))
        if self.divergence_type == 'logit_l2':
            la, lb = pi_flat.clamp_min(1e-10).log(), pj_flat.clamp_min(1e-10).log()
            za, zb = la - la.mean(dim=1, keepdim=True), lb - lb.mean(dim=1, keepdim=True)
            return ((za - zb) ** 2).sum(dim=1)
        return super()._edge_divergence(pi_flat, pj_flat)

    def _penalty(self, probs, pj):
        B, k, C = pj.shape
        if self.target == 'mixture':
            return self._edge_divergence(probs, pj.mean(dim=1)).mean()
        return self._edge_divergence(probs.unsqueeze(1).expand(B, k, C).reshape(-1, C), pj.reshape(-1, C)).mean()

    def forward(self, logits, labels, node_indices=None, features=None, **kwargs):
        if not self.active or logits.shape[0] < 2:
            return torch.tensor(0.0, device=logits.device)
        probs = F.softmax(logits, dim=1)
        if self.graph == 'minibatch':
            if features is None:
                raise ValueError("graph='minibatch' needs features: apply the train_epoch edit")
            with torch.autocast(device_type=logits.device.type, enabled=False):
                _, idx = _minibatch_topk(features.float(), min(self.k, logits.shape[0] - 1))
                return self._penalty(probs.float(), probs.detach().float()[idx])
        if self.drift_path:
            self._log_drift(probs.detach(), node_indices)
        if self.cache_sync_epochs:
            self._write_pending(probs.detach(), node_indices)
        else:
            self.update_global_probs(probs, node_indices)
        if self.target == 'edge':
            return self._space_aware_regularization_loss(probs, node_indices)
        return self._penalty(probs, self.global_probs[self.neighbors[node_indices].to(probs.device)])

    def _write_pending(self, p, idx):
        if self._pending is None:
            self._pending = self.global_probs.clone()
        self._pending[idx] = p
        self._seen_sync += p.shape[0]
        if self._seen_sync >= self.cache_sync_epochs * self.global_probs.shape[0]:
            self.global_probs.copy_(self._pending)
            self._seen_sync = 0

    def _log_drift(self, p, idx):
        nb = self.neighbors[idx].to(p.device)
        B, k = nb.shape
        drift = js_divergence_torch(self.global_probs[idx], p).clamp_min(0.0).sqrt()
        edge = js_divergence_torch(p.repeat_interleave(k, dim=0), self.global_probs[nb].reshape(B * k, -1)).clamp_min(0.0).sqrt()
        a = self._drift_acc
        a[0] += float(edge.sum()); a[1] += float(drift.sum()); a[2] += B * k; a[3] += B
        if a[3] >= self.global_probs.shape[0]:
            self._drift_hist.append({'sqrt_js_edge_to_cache': a[0] / a[2], 'sqrt_js_cache_drift': a[1] / a[3]})
            self._drift_acc = [0.0, 0.0, 0, 0]
            with open(self.drift_path, 'w') as f:
                json.dump(self._drift_hist, f)

    def precompute_matrices(self, features, chunk_size=512):
        if self.precomputed_neighbors is not None:
            self.neighbors = self.precomputed_neighbors
            self.neighbor_weights = torch.ones_like(self.neighbors, dtype=torch.float32)
            return
        if self.graph_backend == 'exact':
            return super().precompute_matrices(features, chunk_size=chunk_size)
        x = F.normalize(features, p=2, dim=1).detach().cpu().numpy().astype('float32')
        I, stats = ann_knn(x, self.k, self.graph_backend)
        stats['recall_at_k'] = recall_at_k(x, I, self.k)
        print(f'[ann] {stats}', flush=True)
        self.neighbors = torch.as_tensor(I, dtype=torch.long)
        self.neighbor_weights = torch.ones_like(self.neighbors, dtype=torch.float32)
        if self.stats_path:
            with open(self.stats_path, 'w') as f:
                json.dump(stats, f, indent=2)


class RankMixupLoss(nn.Module):
    requires_mixup = True
    default_lambda_reg = 0.1

    def __init__(self, alpha: float = 2.0, margin: float = 2.0):
        super().__init__()
        self.alpha, self.margin = alpha, margin

    def forward(self, logits, labels, node_indices=None, mixed_logits=None, **kwargs):
        if mixed_logits is None:
            raise ValueError('RankMixupLoss needs mixed_logits: apply the train_epoch edit in handlers/trainer.py')
        c = F.softmax(logits, dim=1).max(dim=1).values
        c_mix = F.softmax(mixed_logits, dim=1).max(dim=1).values
        return F.relu(c_mix - c + self.margin).mean()
