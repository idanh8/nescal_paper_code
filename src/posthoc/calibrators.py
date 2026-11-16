import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from posthoc.local_net import LoCalNet, locals_loss


class BasePostHocCalibrator:
    name = 'base'
    requires_features = False
    requires_seed = False

    def get_params(self):
        return {}

class TemperatureScaling(BasePostHocCalibrator):
    """
    Single-parameter temperature scaling (Guo et al., 2017).
    """

    name = 'ts'

    def __init__(self, lr: float = 0.01, max_iter: int = 50, init_temp: float = 1.5):
        self.lr = lr
        self.max_iter = max_iter
        self.init_temp = init_temp
        self.temperature = None
        self.refit_positive = False

    def fit(self, val_logits, val_labels, val_features=None):


        from scipy.optimize import minimize_scalar
        z = np.asarray(val_logits, np.float64); y = np.asarray(val_labels, np.int64); rows = np.arange(len(y))

        def nll(log_t):
            s = z / np.exp(log_t); s = s - s.max(1, keepdims=True)
            return float(np.mean(np.log(np.exp(s).sum(1)) - s[rows, y]))

        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method='bounded', options={'xatol': 1e-6})
        self.temperature = float(np.exp(res.x))
        self.refit_positive = False
        return self

    def transform(self, logits, features=None):
        if self.temperature is None:
            raise RuntimeError('TemperatureScaling.transform called before fit.')
        logits = torch.tensor(np.asarray(logits), dtype=torch.float32)
        probs = F.softmax(logits / self.temperature, dim=1)
        return probs.numpy()

    def get_params(self):
        return {'temperature': self.temperature, 'ts_refit': self.refit_positive}

class LoCalCalibrator(BasePostHocCalibrator):
    """
    LoCal Nets (Barbera et al., AISTATS 2026). Adapted from the authors's code.
    """

    name = 'local'
    requires_features = True
    requires_seed = True

    def __init__(self, similarity_dim=50, hidden_dim=64, dropout=0.3, bandwidth=10.0,
                 lambda_js=1.0, alpha_ce=1.0, lr=1e-3, weight_decay=0.0, epochs=100,
                 batch_size=1024, patience=10, es_frac=0.2, seed=42, device='cpu', verbose=False):
        self.similarity_dim = similarity_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.bandwidth = bandwidth
        self.lambda_js = lambda_js
        self.alpha_ce = alpha_ce
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.es_frac = es_frac
        self.seed = seed
        self.device = device
        self.verbose = verbose

        self.net = None
        self.scaler = None
        self.pca = None
        self.best_epoch = None
        self.best_val_js = None
        self.history = []


    def _project(self, features):
        return self.pca.transform(self.scaler.transform(np.asarray(features, dtype=np.float32))).astype(np.float32)

    def _eval_js(self, feats, logits, y_onehot):
        self.net.eval()
        sigma = float(self.bandwidth) ** 2
        totals, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, feats.shape[0], self.batch_size):
                end = min(start + self.batch_size, feats.shape[0])
                if end - start < 2:
                    continue
                new_logits, phi = self.net(feats[start:end, :-self.similarity_dim],
                                           logits[start:end], feats[start:end, -self.similarity_dim:])
                _, js, _ = locals_loss(new_logits, phi, y_onehot[start:end], sigma,
                                       self.lambda_js, self.alpha_ce)
                totals += float(js.item())
                n += 1
        self.net.train()
        return totals / max(n, 1)


    def fit(self, val_logits, val_labels, val_features=None):
        if val_features is None:
            raise ValueError("LoCalCalibrator requires val_features (penultimate embeddings).")

        logits_np = np.asarray(val_logits, dtype=np.float32)
        labels_np = np.asarray(val_labels).astype(np.int64)
        feats_np = np.asarray(val_features, dtype=np.float32)
        num_classes, feature_dim = logits_np.shape[1], feats_np.shape[1]
        self.similarity_dim = min(self.similarity_dim, np.asarray(feats_np).shape[1])


        self.scaler = StandardScaler().fit(feats_np)
        self.pca = PCA(n_components=self.similarity_dim, random_state=self.seed).fit(self.scaler.transform(feats_np))
        pca_np = self._project(feats_np)


        idx_fit, idx_es = train_test_split(np.arange(len(labels_np)), test_size=self.es_frac,
                                           random_state=self.seed, stratify=labels_np)

        dev = torch.device(self.device)


        X = torch.tensor(np.concatenate([feats_np, pca_np], axis=1), device=dev)
        L = torch.tensor(logits_np, device=dev)
        Y = F.one_hot(torch.tensor(labels_np, device=dev), num_classes=num_classes).float()

        torch.manual_seed(self.seed)
        self.net = LoCalNet(feature_dim, num_classes, self.similarity_dim, self.hidden_dim, self.dropout).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sigma = float(self.bandwidth) ** 2

        rng = np.random.default_rng(self.seed)
        best_js, best_state, best_epoch, since_best = float('inf'), None, -1, 0

        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(idx_fit)
            ep_js, ep_ce, n_batches = 0.0, 0.0, 0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                if len(batch) < 2:
                    continue
                b = torch.tensor(batch, device=dev, dtype=torch.long)
                new_logits, phi = self.net(X[b, :feature_dim], L[b], X[b, feature_dim:])
                loss, js, ce = locals_loss(new_logits, phi, Y[b], sigma, self.lambda_js, self.alpha_ce)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_js += float(js.item()); ep_ce += float(ce.item()); n_batches += 1

            es = torch.tensor(idx_es, device=dev, dtype=torch.long)
            val_js = self._eval_js(X[es], L[es], Y[es])
            self.history.append({'epoch': epoch, 'val_js': val_js,
                                 'train_js': ep_js / max(n_batches, 1),
                                 'train_ce': ep_ce / max(n_batches, 1)})
            if self.verbose:
                print(f'  [local] epoch {epoch:3d}  val_js {val_js:.6f}')

            if val_js < best_js:
                best_js, best_epoch, since_best = val_js, epoch, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        self.net.load_state_dict(best_state)   # authors restore the best checkpoint, not the last
        self.best_epoch, self.best_val_js = best_epoch, best_js
        self.feature_dim = feature_dim
        return self

    def transform(self, logits, features=None):
        if self.net is None:
            raise RuntimeError('LoCalCalibrator.transform called before fit.')
        if features is None:
            raise ValueError('LoCalCalibrator requires test_features.')

        dev = torch.device(self.device)
        feats = torch.tensor(np.asarray(features, dtype=np.float32), device=dev)
        pcas = torch.tensor(self._project(features), device=dev)
        lgts = torch.tensor(np.asarray(logits, dtype=np.float32), device=dev)

        self.net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, feats.shape[0], self.batch_size):
                end = min(start + self.batch_size, feats.shape[0])
                new_logits, _ = self.net(feats[start:end], lgts[start:end], pcas[start:end])
                out.append(F.softmax(new_logits, dim=1).cpu())
        return torch.cat(out).numpy()

    def get_params(self):
        return {'best_epoch': self.best_epoch, 'best_val_js': self.best_val_js,
                'epochs_run': len(self.history), 'history': self.history,
                'alpha_sim': self.net.alpha_sim.detach().cpu().numpy().tolist(),
                'alpha_cls': self.net.alpha_cls.detach().cpu().numpy().tolist(),
                'bandwidth': self.bandwidth, 'similarity_dim': self.similarity_dim}

class DirichletCalibrator(BasePostHocCalibrator):
    """
    Full Dirichlet calibration with ODIR (Kull et al., NeurIPS 2019).
    Adapted from the authors' code.

    """

    name = 'dc'
    requires_features = False
    requires_seed = True

    def __init__(self, reg_lambda_list=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 150.0, 200.0), reg_mu_list=('tie',),
                 reg_norm=True, ref_row=True, lr=0.1, max_iter=200, eps=1e-12,
                 val_frac=0.2, seed=42, device='cpu', verbose=False):
        self.reg_lambda_list = list(reg_lambda_list)
        self.reg_mu_list = list(reg_mu_list)
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.val_frac = val_frac
        self.seed = seed
        self.device = device
        self.verbose = verbose

        self.W = None
        self.b = None
        self.reg_lambda = None
        self.reg_mu = None
        self.grid = []


    def _log_probs(self, logits):
        p = F.softmax(torch.tensor(np.asarray(logits), dtype=torch.float32, device=self.device), dim=1)
        return torch.log(torch.clamp(p, self.eps, 1.0 - self.eps))

    def _gauge(self, W, b):

        if not self.ref_row:
            return W, b
        return W - W[-1:], b - b[-1]

    def _fit_one(self, logp, labels, reg_lambda, reg_mu):
        C = logp.shape[1]
        W_raw = torch.eye(C, device=self.device).requires_grad_(True)
        b_raw = torch.zeros(C, device=self.device).requires_grad_(True)
        off = ~torch.eye(C, dtype=torch.bool, device=self.device)
        n_off = float(C * (C - 1))

        nll = torch.nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([W_raw, b_raw], lr=self.lr, max_iter=self.max_iter,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            W, b = self._gauge(W_raw, b_raw)
            w_pen, b_pen = W[off].pow(2).sum(), b.pow(2).sum()
            if self.reg_norm:
                w_pen, b_pen = w_pen / n_off, b_pen / C
            loss = nll(logp @ W.t() + b, labels) + reg_lambda * w_pen + reg_mu * b_pen
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            W, b = self._gauge(W_raw, b_raw)
        return W.detach().clone(), b.detach().clone()


    def fit(self, val_logits, val_labels, val_features=None):
        labels_np = np.asarray(val_labels).astype(np.int64)
        logp = self._log_probs(val_logits)
        labels = torch.tensor(labels_np, dtype=torch.long, device=self.device)


        idx_fit, idx_sel = train_test_split(np.arange(len(labels_np)), test_size=self.val_frac,
                                            random_state=self.seed, stratify=labels_np)
        fit_i = torch.tensor(idx_fit, dtype=torch.long, device=self.device)
        sel_i = torch.tensor(idx_sel, dtype=torch.long, device=self.device)

        best = None
        for lam in self.reg_lambda_list:
            for mu in self.reg_mu_list:
                if mu == 'tie':
                    mu_eff = lam
                elif mu is None:
                    raise ValueError("reg_mu=None selects Kull's non-ODIR L2 branch. Use 'tie' or a float.")
                else:
                    mu_eff = float(mu)

                W, b = self._fit_one(logp[fit_i], labels[fit_i], lam, mu_eff)
                with torch.no_grad():
                    sel_ll = float(F.cross_entropy(logp[sel_i] @ W.t() + b, labels[sel_i]).item())
                self.grid.append({'lambda': lam, 'mu': mu_eff, 'val_logloss': sel_ll})
                if self.verbose:
                    print(f'  [dc] lambda={lam:<8g} mu={mu_eff:<8g} val NLL={sel_ll:.5f}')
                if best is None or sel_ll < best[0]:
                    best = (sel_ll, lam, mu_eff, W, b)

        _ll, self.reg_lambda, self.reg_mu, self.W, self.b = best
        lo, hi = min(self.reg_lambda_list), max(self.reg_lambda_list)
        if self.reg_lambda in (lo, hi) and len(self.reg_lambda_list) > 1:
            edge = 'lower' if self.reg_lambda == lo else 'upper'
            print(f'  [dc] WARNING: selected reg_lambda={self.reg_lambda:g} is at the {edge} edge of '
                  f'the grid {self.reg_lambda_list}. The optimum may lie outside it -- widen and refit '
                  f'before reporting this number.')
            self.grid_edge = edge
        else:
            self.grid_edge = None
        if self.verbose:
            print(f'  [dc] selected lambda={self.reg_lambda:g} mu={self.reg_mu:g} (val NLL={_ll:.5f})')
        return self

    def transform(self, logits, features=None):
        if self.W is None:
            raise RuntimeError('DirichletCalibrator.transform called before fit.')
        with torch.no_grad():
            return F.softmax(self._log_probs(logits) @ self.W.t() + self.b, dim=1).cpu().numpy()

    def get_params(self):
        C = self.W.shape[0]
        off = self.W[~torch.eye(C, dtype=torch.bool, device=self.W.device)]
        return {'reg_lambda': self.reg_lambda, 'reg_mu': self.reg_mu, 'reg_norm': self.reg_norm,
                'ref_row': self.ref_row, 'grid': self.grid,
                'offdiag_rms': float(off.pow(2).mean().sqrt().item()),
                'diag_mean': float(self.W.diagonal().mean().item()),
                'W': self.W.cpu().numpy(), 'b': self.b.cpu().numpy()}

class LoReCalibrator(BasePostHocCalibrator):
    """LoRe: local recalibration (Luo et al., UAI 2022, Eq. 6).
        Based on the Barbera et al. LoCal Nets implementation of LCE  
    """
    name = 'lore'
    requires_features = True
    requires_seed = True

    def __init__(self, n_bins=15, gamma=10.0, kernel='gaussian', similarity_dim=50,
                 lce_projection=None, seed=0, device='cpu', chunk=2048, **kw):
        self.n_bins = n_bins; self.gamma = gamma; self.kernel = kernel
        self.similarity_dim = similarity_dim; self.lce_projection = lce_projection
        self.seed = seed; self.device = device; self.chunk = chunk
        self.cal_phi = self.cal_bin = self.cal_correct = self.cal_conf = None

    def _project(self, features):
        from metrics.calibration_metrics import fit_lce_projection, apply_lce_projection
        if self.lce_projection is None:
            self.lce_projection = fit_lce_projection(features, n_components=self.similarity_dim, seed=self.seed)
        return apply_lce_projection(self.lce_projection, features).astype(np.float32)

    def _bin(self, conf):
        return np.clip(np.digitize(conf, np.linspace(0.0, 1.0, self.n_bins + 1)) - 1, 0, self.n_bins - 1)

    def fit(self, val_logits, val_labels, val_features=None):
        if val_features is None:
            raise ValueError("LoReCalibrator requires val_features.")
        pv = _np_softmax(val_logits)
        self.cal_phi = self._project(val_features)
        self.cal_conf = pv.max(1); self.cal_bin = self._bin(self.cal_conf)
        self.cal_correct = (pv.argmax(1) == np.asarray(val_labels)).astype(np.float32)
        return self

    def transform(self, logits, features=None):
        if self.cal_phi is None: raise RuntimeError("LoReCalibrator.transform before fit.")
        if features is None: raise ValueError("LoReCalibrator requires test_features.")
        p = _np_softmax(logits); conf = p.max(1); pred = p.argmax(1); tbin = self._bin(conf)
        phi_t = self._project(features)

        dev = torch.device(self.device)
        Cphi_all = torch.tensor(self.cal_phi, device=dev)
        Ccorr_all = torch.tensor(self.cal_correct, device=dev)
        new_conf = conf.copy(); two_s2 = 2.0 * float(self.gamma) ** 2
        for b in range(self.n_bins):
            t_idx = np.where(tbin == b)[0]; c_idx = np.where(self.cal_bin == b)[0]
            if len(t_idx) == 0 or len(c_idx) == 0: continue
            Cphi = Cphi_all[c_idx]; Ccorr = Ccorr_all[c_idx]; Csq = (Cphi ** 2).sum(1)
            for s in range(0, len(t_idx), self.chunk):
                sel = t_idx[s:s + self.chunk]; T = torch.tensor(phi_t[sel], device=dev)
                if self.kernel == 'laplacian':
                    w = torch.exp(-torch.cdist(T, Cphi, p=1) / (Cphi.shape[1] * float(self.gamma)))
                else:
                    d2 = torch.clamp((T ** 2).sum(1, keepdim=True) + Csq[None, :] - 2 * (T @ Cphi.T), min=0.0)
                    w = torch.exp(-d2 / two_s2)
                denom = w.sum(1); cprime = (w @ Ccorr) / (denom + 1e-12)
                cprime = torch.where(denom > 1e-12, cprime, torch.full_like(denom, float('nan'))).cpu().numpy()
                ok = ~np.isnan(cprime); new_conf[sel[ok]] = cprime[ok]
        new_conf = np.clip(new_conf, 1e-6, 1.0 - 1e-6)

        rows = np.arange(len(p)); out = p.copy(); out[rows, pred] = 0.0
        rest = out.sum(1); scale = np.where(rest > 1e-12, (1.0 - new_conf) / (rest + 1e-12), 0.0)
        out = out * scale[:, None]
        deg = rest <= 1e-12
        if deg.any(): out[deg] = ((1.0 - new_conf[deg]) / (p.shape[1] - 1))[:, None]
        out[rows, pred] = new_conf
        return out

    def get_params(self):
        return {'n_bins': self.n_bins, 'gamma': self.gamma, 'kernel': self.kernel,
                'similarity_dim': self.similarity_dim}

def _np_softmax(x):
    x = np.asarray(x, np.float64); x = x - x.max(1, keepdims=True); e = np.exp(x)
    return e / e.sum(1, keepdims=True)

class DACCalibrator(BasePostHocCalibrator):
    """
    Density-Aware Calibration (Tomani et al. 2023). Ported from the authors' density_aware_calib.py 
    """
    name = 'dac'
    requires_features = True
    requires_train_features = True
    requires_seed = False

    def __init__(self, knn_k=50, bounded=True, tol=1e-12, eps=1e-7, verbose=False, **kw):
        self.knn_k = knn_k; self.bounded = bounded; self.tol = tol; self.eps = eps; self.verbose = verbose
        self.w = None; self._train_norm = None
        self.bnds = ([[0.0, 1e4]] if bounded else [[-1e4, 1e4]]) + [[-100.0, 100.0]]
        self.init = [1.0, 1.0]
        self._index = None

    @staticmethod
    def _norm(x):
        x = np.asarray(x, np.float32); return x / (np.linalg.norm(x, 2, 1, keepdims=True) + 1e-10)

    def _build_index(self):
        import faiss
        self._index = faiss.IndexFlatL2(self._train_norm.shape[1])
        self._index.add(np.ascontiguousarray(self._train_norm))

    def _knn_score(self, query):
        q = np.ascontiguousarray(self._norm(query))
        D, _ = self._index.search(q, self.knn_k)
        return D[:, -1]                              # kth-NN distance; authors' avg_top_k=False default

    def _temperature(self, w, s):
        return np.clip(w[0] * s + w[-1], 1e-20, None)

    def _nll(self, w, logits, onehot, s):
        p = _np_softmax(logits / self._temperature(w, s)[:, None])
        return -np.sum(onehot * np.log(p + 1e-12)) / p.shape[0]

    def fit(self, val_logits, val_labels, val_features=None, train_features=None, train_labels=None):
        if val_features is None or train_features is None:
            raise ValueError("DACCalibrator needs val_features AND train_features.")
        from scipy import optimize
        self._train_norm = self._norm(train_features)
        self._build_index()
        s = self._knn_score(val_features)
        logits = np.asarray(val_logits, np.float64)
        onehot = np.eye(logits.shape[1])[np.asarray(val_labels).astype(int)]
        res = optimize.minimize(self._nll, self.init, args=(logits, onehot, s), method='L-BFGS-B',
                                bounds=self.bnds, tol=self.tol, options={'eps': self.eps, 'disp': self.verbose})
        self.w = res.x
        return self

    def transform(self, logits, features=None):
        if self.w is None: raise RuntimeError('DACCalibrator.transform before fit.')
        if features is None: raise ValueError('DACCalibrator needs test_features.')
        return _np_softmax(np.asarray(logits, np.float64) / self._temperature(self.w, self._knn_score(features))[:, None])

    def get_params(self):
        return {'w_scale': float(self.w[0]), 'w_bias': float(self.w[-1]), 'knn_k': self.knn_k}


CALIBRATOR_REGISTRY = {
    'ts': TemperatureScaling,
    'local': LoCalCalibrator,
    'dc': DirichletCalibrator,
    'lore':  LoReCalibrator,
    'dac':   DACCalibrator,
}

def build_calibrator(name: str, **kwargs) -> BasePostHocCalibrator:
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator '{name}'. "
                         f"Available: {sorted(CALIBRATOR_REGISTRY)}")
    return CALIBRATOR_REGISTRY[name](**kwargs)


def apply_posthoc_suite(val_logits, val_labels, val_features, test_logits, test_labels, test_features,
                        calibrators=('ts', 'local', 'dc'), seed=None, num_bins=15, lce_projection=None,
                        device='cpu', train_features=None, train_labels=None, calibrator_kwargs=None,
                        include_local_metrics=False):
    import numpy as np
    from metrics.calibration_metrics import (calibration_metrics, fit_lce_projection,
                                             proper_score_decomposition, local_freq_error)
    if lce_projection is None:
        lce_projection = fit_lce_projection(val_features, n_components=50, seed=seed or 0)
    calibrator_kwargs = dict(calibrator_kwargs or {})
    calibrator_kwargs.setdefault('lore', {}).setdefault('lce_projection', lce_projection)

    def _softmax(z):
        z = np.asarray(z); z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def _metrics(probs):
        m = calibration_metrics(probs, test_labels, test_features, num_bins=num_bins, plots=False,
                                lce_projection=lce_projection)
        m['accuracy'] = float((probs.argmax(1) == np.asarray(test_labels)).mean())
        if include_local_metrics:
            m.update(proper_score_decomposition(probs, test_labels, num_bins=num_bins))
            m.update(local_freq_error(probs, test_labels, test_features, lce_projection=lce_projection))
        return m

    out = {'pre': _metrics(_softmax(test_logits)), 'post': {}, 'params': {}}
    for name in calibrators:
        cls = CALIBRATOR_REGISTRY[name]
        if getattr(cls, 'requires_features', False) and (val_features is None or test_features is None):
            out['post'][name] = {'error': 'requires features; none saved'}; continue
        kwargs = dict(calibrator_kwargs.get(name, {}))
        if getattr(cls, 'requires_seed', False) and seed is not None: kwargs.setdefault('seed', seed)
        if getattr(cls, 'requires_features', False): kwargs.setdefault('device', device)
        cal = build_calibrator(name, **kwargs)
        fit_kwargs = {}
        if getattr(cls, 'requires_train_features', False):
            if train_features is None:
                out['post'][name] = {'error': 'requires train features; none supplied'}; continue
            fit_kwargs = {'train_features': train_features, 'train_labels': train_labels}
        cal.fit(val_logits, val_labels, val_features, **fit_kwargs)
        out['post'][name] = _metrics(cal.transform(test_logits, test_features)); out['params'][name] = cal.get_params()
    return out
