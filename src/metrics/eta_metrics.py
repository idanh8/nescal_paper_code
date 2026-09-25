import numpy as np
EPS = 1e-12


def _eta_of_pred(probs, PI):
    pred = probs.argmax(1)
    return PI[np.arange(PI.shape[0]), pred]


def point_eta_metrics(probs, PI, region_mask=None):
    probs = np.asarray(probs, np.float64); PI = np.asarray(PI, np.float64)
    c = probs.max(1); eta = _eta_of_pred(probs, PI)
    out = {'mse_to_eta': float(np.mean((c - eta) ** 2)),
           'signed_bias': float(np.mean(c - eta))}
    if region_mask is not None:
        m = np.asarray(region_mask, bool)
        for name, sel in (('noisy', m), ('clean', ~m)):
            if sel.any():
                out[f'mse_to_eta_{name}'] = float(np.mean((c[sel] - eta[sel]) ** 2))
                out[f'signed_bias_{name}'] = float(np.mean(c[sel] - eta[sel]))
    return out


def fullvector_eta_metrics(probs, PI, region_mask=None):
    probs = np.asarray(probs, np.float64); PI = np.asarray(PI, np.float64)
    pc = np.clip(probs, EPS, 1.0)
    l1 = np.abs(probs - PI).sum(1)
    kl = (PI * (np.log(np.clip(PI, EPS, 1.0)) - np.log(pc))).sum(1)
    out = {'l1_to_eta': float(l1.mean()), 'kl_eta_to_p': float(kl.mean())}
    if region_mask is not None:
        m = np.asarray(region_mask, bool)
        for name, sel in (('noisy', m), ('clean', ~m)):
            if sel.any():
                out[f'l1_to_eta_{name}'] = float(l1[sel].mean())
                out[f'kl_eta_to_p_{name}'] = float(kl[sel].mean())
    return out


def best_temperature_oracle(logits, PI, grid=None):
    logits = np.asarray(logits, np.float64); PI = np.asarray(PI, np.float64)
    grid = np.asarray(grid) if grid is not None else np.concatenate(
        [np.linspace(0.5, 1.0, 6, endpoint=False), np.linspace(1.0, 5.0, 41)])
    z = logits - logits.max(1, keepdims=True)
    best = (None, np.inf); curve = []
    for T in grid:
        e = np.exp(z / T); p = e / e.sum(1, keepdims=True)
        c = p.max(1); eta = _eta_of_pred(p, PI)
        mse = float(np.mean((c - eta) ** 2)); curve.append((float(T), mse))
        if mse < best[1]: best = (float(T), mse)
    return best[0], best[1], curve


def _cosine_knn_edges(Z, k):
    from sklearn.neighbors import NearestNeighbors
    Z = np.asarray(Z, np.float64)
    _, idx = NearestNeighbors(n_neighbors=min(k + 1, Z.shape[0]), metric='cosine').fit(Z).kneighbors(Z)
    src, dst = [], []
    for i in range(Z.shape[0]):
        nb = [j for j in idx[i] if j != i][:k]; src += [i] * len(nb); dst += nb
    return np.asarray(src), np.asarray(dst)


def bound_predictiveness(probs, PI, neighbors=None, embeddings=None, k=10):
    from scipy.stats import spearmanr
    probs = np.asarray(probs, np.float64); PI = np.asarray(PI, np.float64)
    c = probs.max(1); eta = _eta_of_pred(probs, PI)
    if neighbors is not None:
        nb = np.asarray(neighbors); N, kk = nb.shape
        src = np.repeat(np.arange(N), kk); dst = nb.reshape(-1)
    else:
        src, dst = _cosine_knn_edges(embeddings, k)

    p_i, p_j = probs[src], probs[dst]
    m = 0.5 * (p_i + p_j)
    js = 0.5 * (np.sum(p_i * (np.log2(np.clip(p_i, EPS, 1)) - np.log2(np.clip(m, EPS, 1))), 1)
                + np.sum(p_j * (np.log2(np.clip(p_j, EPS, 1)) - np.log2(np.clip(m, EPS, 1))), 1))
    js = np.clip(js, 0, None)
    c_i, c_j = c[src], c[dst]
    pred_gap = np.sqrt(8 * np.log(2) * c_i * js)
    realised_gap = np.abs(c_i - c_j)
    true_post_gap = 0.5 * np.abs(PI[src] - PI[dst]).sum(1)
    true_conf_gap = np.abs(eta[src] - eta[dst])

    def _sp(a, b):
        if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
            return float(spearmanr(a, b).correlation)
        return float('nan')

    return {'js_edge': js, 'pred_gap': pred_gap, 'realised_gap': realised_gap,
            'true_post_gap': true_post_gap, 'true_conf_gap': true_conf_gap,
            'n_edges': int(len(src)),
            'violation_rate': float(np.mean(realised_gap > pred_gap)),
            'spearman_pred_vs_realised': _sp(pred_gap, realised_gap),
            'spearman_pred_vs_truegap': _sp(pred_gap, true_post_gap),
            'spearman_realised_vs_truegap': _sp(realised_gap, true_post_gap)}


def all_eta_metrics(probs, logits, PI, region_mask=None, run_oracle=True):
    out = {}
    out.update(point_eta_metrics(probs, PI, region_mask))
    out.update(fullvector_eta_metrics(probs, PI, region_mask))
    if run_oracle:
        T, mse, _ = best_temperature_oracle(logits, PI)
        out['oracle_T'] = T; out['oracle_mse_to_eta'] = mse
    return out
