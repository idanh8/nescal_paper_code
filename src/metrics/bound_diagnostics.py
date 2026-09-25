import numpy as np
EPS = 1e-12

def js_divergence(p, q):
    p = np.clip(p, EPS, 1.0); q = np.clip(q, EPS, 1.0); m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * (np.log2(p) - np.log2(m)), 1) + np.sum(q * (np.log2(q) - np.log2(m)), 1))

def laplacian_approx(p_i, p_j):
    return np.sum((p_i - p_j) ** 2 / (8 * np.log(2) * np.clip(p_i, EPS, 1.0)), axis=1)

def _edges(neighbors=None, embeddings=None, k=10):
    if neighbors is not None:
        neighbors = np.asarray(neighbors); N, kk = neighbors.shape
        return np.repeat(np.arange(N), kk).astype(int), neighbors.reshape(-1).astype(int)
    from sklearn.neighbors import NearestNeighbors
    z = np.asarray(embeddings)
    _, idx = NearestNeighbors(n_neighbors=min(k + 1, z.shape[0]), metric='cosine').fit(z).kneighbors(z)
    src, dst = [], []
    for i in range(z.shape[0]):
        nb = [j for j in idx[i] if j != i][:k]; src += [i] * len(nb); dst += nb
    return np.asarray(src), np.asarray(dst)

def compute_bound_diagnostics(probs_over_nodes, neighbors=None, embeddings=None, k=10,
                              head_weight=None, label='final', split='train'):
    p = np.asarray(probs_over_nodes, dtype=np.float64); c = p.max(1)
    src, dst = _edges(neighbors, embeddings, k)
    p_i, p_j, c_i, c_j = p[src], p[dst], c[src], c[dst]
    js_e = js_divergence(p_i, p_j); js_a = laplacian_approx(p_i, p_j)
    rel = np.abs(js_e - js_a) / np.maximum(js_e, EPS)
    dp = np.linalg.norm(p_i - p_j, axis=1); rem = np.abs(js_e - js_a); m = (dp > EPS) & (rem > 0)
    slope = float(np.polyfit(np.log(dp[m]), np.log(rem[m]), 1)[0]) if m.sum() > 10 else np.nan
    lhs = np.abs(c_i - c_j); rhs = np.sqrt(8 * np.log(2) * c_i * np.clip(js_e, 0, None))
    out = {'label': label, 'split': split, 'n_points': int(p.shape[0]), 'n_edges': int(len(src)), 'k': int(k),
           'rn_exact': float(js_e.sum()), 'rn_approx': float(js_a.sum()), 'dp_norm_median': float(np.median(dp)),
           'rn_relative_error_median': float(np.median(rel)), 'rn_relative_error_p90': float(np.percentile(rel, 90)),
           'remainder_loglog_slope': slope, 'confsmooth_violation_rate': float(np.mean(lhs > rhs)),
           'confsmooth_tightness_median': float(np.median(lhs / np.maximum(rhs, EPS)))}
    if embeddings is not None:
        z = np.asarray(embeddings); dz = np.linalg.norm(z[src] - z[dst], axis=1); dzm = dz > EPS
        out['lipschitz_empirical_ratio'] = float(np.max(dp[dzm] / dz[dzm])) if dzm.any() else np.nan
    if head_weight is not None:
        out['lipschitz_bound_theoretical'] = float(0.5 * np.linalg.svd(np.asarray(head_weight), compute_uv=False)[0])
    return out
