import os
import copy
import pickle
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from metrics.calibration_metrics import calibration_metrics, ece_mce, brier_score, local_freq_error
from viz.vizlib import basic_metrics_per_epoch_unspec
from utils.general_utils import set_seed, DivergedRun, make_lambda_schedule
from losses.losses import get_loss
from models.factory import create_model
from metrics.graph_metrics import _embed_probs_labels, _cwcc, _knn_jaccard
from metrics.bound_diagnostics import compute_bound_diagnostics


def _nescal_epoch_diagnostics(model, loader, layer, epoch, seed, prev_emb, subset, k=10):
    Z, P, Y = _embed_probs_labels(model.model, loader, model.device, layer)
    if not (np.isfinite(Z).all() and np.isfinite(P).all()):


        return {'epoch': epoch, 'nonfinite_embeddings': True}, prev_emb
    rec = {'epoch': epoch, **compute_embedding_quality(Z, Y, seed=seed),
           'mean_entropy_train': float((-(P * np.log(np.clip(P, 1e-12, 1.0))).sum(1)).mean())}
    Zs, Ps = Z[subset], P[subset]
    rec['cwcc'] = _cwcc(torch.tensor(Zs), torch.tensor(Ps), k)
    rec['graph_settle_jaccard'] = _knn_jaccard(Zs, prev_emb, k)
    if model.secondary_active:
        _nb = model._inner_secondary.neighbors.cpu().numpy()
        _yi = P.argmax(1)[np.repeat(np.arange(len(P)), _nb.shape[1])]
        _yj = P.argmax(1)[_nb.reshape(-1)]
        rec['boundary_keep_frac'] = float((_yi == _yj).mean())
        from metrics.bound_diagnostics import compute_bound_diagnostics
        rec['bound'] = compute_bound_diagnostics(
            P, neighbors=model._inner_secondary.neighbors.cpu().numpy(), embeddings=Z,
            head_weight=model.model.linear.weight.detach().cpu().numpy(), label=f'epoch_{epoch}')
    return rec, Zs

def compute_embedding_quality(embeddings: np.ndarray, labels: np.ndarray, knn_k: int = 5,
                               max_samples: int = 5000, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    n = embeddings.shape[0]
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    knn = KNeighborsClassifier(n_neighbors=knn_k + 1, metric='cosine')
    knn.fit(embeddings, labels)
    neighbors = knn.kneighbors(embeddings, return_distance=False)[:, 1:]
    neighbor_labels = labels[neighbors]
    majority_vote = np.array([np.bincount(row).argmax() for row in neighbor_labels])
    knn_probe_acc = accuracy_score(labels, majority_vote)

    try:
        sil = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)), random_state=seed)
    except ValueError:
        sil = float('nan')

    return {'knn_probe_acc': knn_probe_acc, 'silhouette': sil}

def evaluate_corruptions(model, base_name, corruptions, severities, num_bins, data_root='./data', batch_size=256):
    import torch, torch.nn.functional as F
    from data.image_datasets import load_cifar_c, load_tinyimagenet_c
    from metrics.calibration_metrics import ece_mce, brier_score
    corruption_loader = (load_tinyimagenet_c if base_name == 'tinyimagenet' else load_cifar_c)
    out = {}
    for c in corruptions:
        out[c] = {}
        for s in severities:
            try:
                if base_name == 'tinyimagenet':
                    loader = corruption_loader(c, s, batch_size=batch_size, data_root=data_root)
                else:
                    loader = corruption_loader(base_name, c, s, batch_size=batch_size, data_root=data_root)
                _, probs, labels = _embed_probs_labels(model.model, loader, model.device, 'final')
                if len(labels) == 0:
                    raise FileNotFoundError('no usable images')
                out[c][s] = {'ece': float(ece_mce(probs, labels, num_bins=num_bins)[0]),
                             'nll': float(F.nll_loss(torch.log(torch.tensor(probs).clamp_min(1e-12)),
                                                     torch.tensor(labels)).item()),
                             'brier': float(brier_score(probs, labels)),
                             'accuracy': float((probs.argmax(1) == labels).mean())}
            except Exception as e:
                print(f'  [corruption] skipping {c}/{s}: {type(e).__name__}: {e}', flush=True)
        if not out[c]:
            del out[c]
    return out

def evaluate_ood_separation(model, base_name, data_root='./data', batch_size=256, size=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    from data.image_datasets import load_svhn_ood
    id_probs = np.asarray(model.all_test_conf_vectors)
    _, ood_probs, _ = _embed_probs_labels(model.model,
                        load_svhn_ood(batch_size=batch_size, data_root=data_root, normalize_as=base_name, size=size),
                        model.device, 'final')
    def scores(p): return {'msp': p.max(1), 'neg_entropy': (p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)}
    ids, oods = scores(id_probs), scores(ood_probs); out = {}
    for k in ('msp', 'neg_entropy'):
        s = np.concatenate([ids[k], oods[k]]); y = np.concatenate([np.ones(len(ids[k])), np.zeros(len(oods[k]))])
        thr = np.percentile(ids[k], 5)
        out[k] = {'auroc': float(roc_auc_score(y, s)), 'aupr_id': float(average_precision_score(y, s)),
                  'fpr_at_95_tpr': float((oods[k] >= thr).mean())}
    return out

@torch.no_grad()
def extract_all_embeddings(model, loader, device, candidate_layers):
    model.eval()
    collected = {layer: [] for layer in candidate_layers}
    all_labels = []

    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        _, embeddings = model(images, return_embeddings=True)
        for layer in candidate_layers:
            collected[layer].append(embeddings[layer].cpu())
        all_labels.append(labels)

    for layer in candidate_layers:
        collected[layer] = torch.cat(collected[layer]).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return collected, all_labels


class NCLActivationController:
    def __init__(self, strategy='fixed', fixed_epoch=None, fixed_fraction=None, total_epochs=None,
                 metric_key='knn_probe_acc', patience=5, min_delta=1e-3):
        assert strategy in ('fixed', 'auto')
        self.strategy = strategy
        self.metric_key = metric_key
        self.patience = patience
        self.min_delta = min_delta
        self._activated = False
        self._best = -float('inf')
        self._stall_count = 0

        if strategy == 'fixed':
            if fixed_epoch is not None:
                self.activation_epoch = fixed_epoch
            elif fixed_fraction is not None and total_epochs is not None:
                self.activation_epoch = int(total_epochs * fixed_fraction)
            else:
                raise ValueError("strategy='fixed' requires fixed_epoch, or fixed_fraction + total_epochs")

    def step(self, epoch: int, diagnostics: dict) -> bool:
        if self._activated:
            return True
        if self.strategy == 'fixed':
            self._activated = epoch >= self.activation_epoch
        else:
            value = diagnostics[self.metric_key]
            if value > self._best + self.min_delta:
                self._best, self._stall_count = value, 0
            else:
                self._stall_count += 1
            self._activated = self._stall_count >= self.patience
        return self._activated


class Trainer:
    def __init__(self, model_params: dict, loss_params: dict, seed: int = 42, lr: float = 0.1,
                 momentum: float = 0.9, weight_decay: float = 5e-4, lr_milestones_frac=(0.5, 0.75),
                 lr_gamma: float = 0.1, num_epochs: int = 200, embedding_layer: str = 'final', use_amp: bool = True,
                 lr_schedule: str = 'multistep', optimizer: str = 'sgd', mixup_alpha: float = None):
        set_seed(seed)
        self.model_params = model_params
        self.embedding_layer = embedding_layer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(**model_params).to(self.device)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_milestones_frac = lr_milestones_frac
        self.lr_gamma = lr_gamma
        self.num_epochs = num_epochs
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.optimizer_type = optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        milestones = [int(num_epochs * f) for f in lr_milestones_frac]
        self.lr_schedule = lr_schedule
        if lr_schedule == 'cosine':

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif lr_schedule == 'multistep':
            milestones = [int(num_epochs * f) for f in lr_milestones_frac]
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=lr_gamma)
        elif lr_schedule == 'constant':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[], gamma=1.0)
        else:
            raise ValueError(f"unknown lr_schedule '{lr_schedule}' (expected 'multistep' or 'cosine')")

        self.loss_params = loss_params
        self.seed = seed
        primary_loss = loss_params.get('primary_loss', 'XE')
        secondary_loss = loss_params.get('secondary_loss', None)
        self.criterion = get_loss(primary_loss, secondary_loss, loss_params)
        self.eval_criterion = get_loss('XE', None, loss_params)
        self._inner_secondary = getattr(self.criterion, 'secondary_loss_fn', None)
        self.lce_projection = None
        self.mixup_alpha = mixup_alpha

    @property
    def secondary_active(self):
        inner = self._inner_secondary
        if inner is None:
            return False
        return getattr(inner, 'active', True)

    def activate_secondary_loss(self, embedding_loader, candidate_layers, embedding_layer):
        """No-op for secondary losses that don't declare requires_activation.
        Kept full-precision — see module docstring."""
        inner = self._inner_secondary
        if inner is None or not getattr(inner, 'requires_activation', False):
            return None

        embeddings_dict, _ = extract_all_embeddings(self.model, embedding_loader, self.device, candidate_layers)
        chosen_embeddings = torch.tensor(embeddings_dict[embedding_layer], device=self.device)

        self.model.eval()
        N = len(embedding_loader.dataset)
        num_classes = self.model_params['num_classes']
        initial_probs = torch.zeros(N, num_classes)
        with torch.no_grad():
            for images, labels, idx in embedding_loader:
                images = images.to(self.device, non_blocking=True)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                initial_probs[idx] = probs.cpu()

        inner.activate(chosen_embeddings, initial_probs.to(self.device))
        return embeddings_dict

    def fit_lce_projection(self, n_components: int = 50, seed: int = 0):
        """Fit the LCE feature space on the VAL split. Must be called after validate()
        and before evaluate(); test features are projected with this val-fitted basis."""
        from metrics.calibration_metrics import fit_lce_projection as _fit
        self.lce_projection = _fit(self.all_val_features, n_components=n_components, seed=seed)
        return self.lce_projection

    def train_epoch(self, dataloader, verbose=False):
        self.model.train()
        total_loss_sum = primary_loss_sum = secondary_loss_sum = 0
        correct = total = 0

        pbar = tqdm(dataloader, desc='Training') if verbose else dataloader
        for images, labels, idx in pbar:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=self.use_amp):
                if getattr(self._inner_secondary, 'requires_mixup', False):
                    _a = self._inner_secondary.alpha
                    _lam = float(np.random.beta(_a, _a))
                    _perm = torch.randperm(images.size(0), device=images.device)
                    outputs = self.model(images)
                    _mixed = self.model(_lam * images + (1.0 - _lam) * images[_perm])
                    total_loss, primary_loss, secondary_loss = self.criterion(outputs, labels, node_indices=idx, mixed_logits=_mixed)
                elif self.mixup_alpha and self._inner_secondary is not None:
                    _lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    _perm = torch.randperm(images.size(0), device=images.device)
                    outputs = self.model(_lam * images + (1.0 - _lam) * images[_perm])
                    _prim = self.criterion.primary_loss_fn
                    primary_loss = _lam * _prim(outputs, labels) + (1.0 - _lam) * _prim(outputs, labels[_perm])
                    secondary_loss = self._inner_secondary(self.model(images), labels, node_indices=idx) if self.secondary_active \
                        else torch.zeros((), device=images.device)
                    total_loss = primary_loss + self.criterion.lambda_reg * secondary_loss
                elif self.mixup_alpha:
                    _lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    _perm = torch.randperm(images.size(0), device=images.device)
                    outputs = self.model(_lam * images + (1.0 - _lam) * images[_perm])
                    _la = self.criterion(outputs, labels, node_indices=idx)
                    _lb = self.criterion(outputs, labels[_perm], node_indices=idx)
                    total_loss, primary_loss, secondary_loss = (_lam * a + (1.0 - _lam) * b for a, b in zip(_la, _lb))
                elif getattr(self._inner_secondary, 'requires_features', False):
                    outputs, _feats = self.model(images, embedding_layer=self.embedding_layer)
                    total_loss, primary_loss, secondary_loss = self.criterion(outputs, labels, node_indices=idx, features=_feats)
                else:
                    outputs = self.model(images)
                    total_loss, primary_loss, secondary_loss = self.criterion(outputs, labels, node_indices=idx)

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_sum += total_loss.detach()
            primary_loss_sum += primary_loss.detach()
            secondary_loss_sum += secondary_loss.detach()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().detach()

        self.scheduler.step()
        n_batches = len(dataloader)
        accuracy = 100. * correct.item() / total
        return (total_loss_sum.item() / n_batches, primary_loss_sum.item() / n_batches,
                secondary_loss_sum.item() / n_batches, accuracy)

    def _eval_pass(self, dataloader, verbose, desc):
        torch.cuda.empty_cache()
        self.model.eval()
        loss_sum = correct = total = 0
        all_preds, all_labels, all_logits, all_features = [], [], [], []

        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=self.use_amp):
            pbar = tqdm(dataloader, desc=desc) if verbose else dataloader
            for images, labels, idx in pbar:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs, embeddings = self.model(images, return_embeddings=True)
                loss_out = self.eval_criterion(outputs, labels)
                loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
                _, predicted = outputs.max(1)

                loss_sum += loss.item()
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
                all_logits.append(outputs.float().cpu())
                all_features.append(embeddings[self.embedding_layer].float().cpu())

        logits = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        preds_cat = torch.cat(all_preds)
        features_cat = torch.cat(all_features)
        conf_vectors = torch.softmax(logits, dim=1)

        return {
            'loss': loss_sum / len(dataloader), 'accuracy': 100. * correct / total,
            'labels': labels_cat.numpy(), 'preds': preds_cat.numpy(), 'logits': logits.numpy(),
            'conf_vectors': conf_vectors.numpy(), 'features': features_cat.numpy(),
            'top_confs': np.max(conf_vectors.numpy(), axis=1),
        }

    def validate(self, dataloader, verbose=False):
        result = self._eval_pass(dataloader, verbose, 'Validation')
        self.all_val_labels = result['labels']
        self.all_val_preds = result['preds']
        self.all_val_logits = result['logits']
        self.all_val_conf_vectors = result['conf_vectors']
        self.all_val_features = result['features']
        self.all_val_top_confs = result['top_confs']
        return result['loss'], result['accuracy']

    def test(self, dataloader, verbose=False):
        result = self._eval_pass(dataloader, verbose, 'Testing')
        self.all_test_labels = result['labels']
        self.all_test_preds = result['preds']
        self.all_test_logits = result['logits']
        self.all_test_conf_vectors = result['conf_vectors']
        self.all_test_features = result['features']
        self.all_test_top_confs = result['top_confs']

    def evaluate(self, calibration_bins=50, verbose=False, plots_params=None, split='test'):
        torch.cuda.empty_cache()
        if split == 'test':
            labels, preds, confs, features = self.all_test_labels, self.all_test_preds, self.all_test_conf_vectors, self.all_test_features
            plot = True
        elif split == 'val':
            labels, preds, confs, features = self.all_val_labels, self.all_val_preds, self.all_val_conf_vectors, self.all_val_features
            plot = False
        else:
            raise ValueError(f"split must be 'test' or 'val', got '{split}'")

        f1 = f1_score(labels, preds, average="macro")
        accuracy = accuracy_score(labels, preds)
        results_dict = {'accuracy': accuracy, 'f1_score': f1}

        if isinstance(calibration_bins, list):
            for nb in calibration_bins:
                if plots_params is not None and plot:
                    plots_params['savename'] = f'{nb}_bins'
                results_dict[f'calibration_metrics_{nb}'] = calibration_metrics(
                    confs, labels, features, num_bins=nb, plots=plot, plot_params=plots_params if plot else None, lce_projection=self.lce_projection)
        else:
            if plots_params is not None and plot:
                plots_params['savename'] = f'{calibration_bins}_bins'
            results_dict.update(calibration_metrics(
                confs, labels, features, num_bins=calibration_bins, plots=plot,
                plot_params=plots_params if plot else None, lce_projection=self.lce_projection))

        if verbose:
            print(f"[{split}] Accuracy: {accuracy:.4f}, F1: {f1:.4f}, "
                  f"ECE: {results_dict.get('ece', 'N/A')}, MCE: {results_dict.get('mce', 'N/A')}")
        return results_dict

    def _constructor_kwargs(self):
        return {
            'model_params': self.model_params, 'loss_params': self.loss_params, 'seed': self.seed,
            'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.weight_decay,
            'lr_milestones_frac': self.lr_milestones_frac, 'lr_gamma': self.lr_gamma,
            'num_epochs': self.num_epochs, 'embedding_layer': self.embedding_layer, 'use_amp': self.use_amp,
            'lr_schedule': self.lr_schedule,
            'mixup_alpha': self.mixup_alpha,
        }

    @classmethod
    def from_saved_params(cls, params):
        """Construct from saved settings, ignoring obsolete fields only when set to None."""
        import inspect
        supported = inspect.signature(cls).parameters
        kwargs = {key: value for key, value in params.items() if key in supported or value is not None}
        return cls(**kwargs)

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model_state.pth'))
        test_data_dict = {
            'test_labels': self.all_test_labels, 'test_preds': self.all_test_preds,
            'test_conf_vectors': self.all_test_conf_vectors, 'test_top_confs': self.all_test_top_confs,
            'test_features': self.all_test_features, 'test_logits': self.all_test_logits
        }
        val_data_dict = {
            'val_labels': self.all_val_labels, 'val_preds': self.all_val_preds,
            'val_conf_vectors': self.all_val_conf_vectors, 'val_top_confs': self.all_val_top_confs,
            'val_features': self.all_val_features, 'val_logits': self.all_val_logits, 'lce_projection': self.lce_projection
        }
        with open(os.path.join(path, 'model_params.pkl'), 'wb') as f:
            pickle.dump(self._constructor_kwargs(), f)
        with open(os.path.join(path, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data_dict, f)
        with open(os.path.join(path, 'val_data.pkl'), 'wb') as f:
            pickle.dump(val_data_dict, f)
        inner = getattr(self.criterion, 'secondary_loss_fn', None)
        if inner is not None and getattr(inner, 'neighbors', None) is not None:
            with open(os.path.join(path, 'graph_data.pkl'), 'wb') as f:
                pickle.dump({'neighbors': inner.neighbors.cpu().numpy(),
                             'neighbor_weights': inner.neighbor_weights.cpu().numpy(),
                             'mean_edge_js_at_activation': getattr(inner, 'mean_edge_js_at_activation', None)}, f)
        print(f'Model and test data saved to {path}')

    @staticmethod
    def load_model(path):
        """Reconstructs a Trainer exactly as trained (same
        optimizer/scheduler config, not just the same weights) and restores
        its saved test-set predictions. New — see module docstring."""
        with open(os.path.join(path, 'model_params.pkl'), 'rb') as f:
            ctor_kwargs = pickle.load(f)
        with open(os.path.join(path, 'test_data.pkl'), 'rb') as f:
            test_data_dict = pickle.load(f)
        with open(os.path.join(path, 'val_data.pkl'), 'rb') as f:
            val_data_dict = pickle.load(f)

        loaded = Trainer.from_saved_params(ctor_kwargs)
        loaded.model.load_state_dict(torch.load(os.path.join(path, 'model_state.pth'), map_location=loaded.device))
        loaded.all_test_labels = test_data_dict['test_labels']
        loaded.all_test_preds = test_data_dict['test_preds']
        loaded.all_test_conf_vectors = test_data_dict['test_conf_vectors']
        loaded.all_test_top_confs = test_data_dict['test_top_confs']
        loaded.all_test_features = test_data_dict['test_features']
        loaded.all_test_logits = test_data_dict['test_logits']
        loaded.all_val_labels = val_data_dict['val_labels']
        loaded.all_val_preds = val_data_dict['val_preds']
        loaded.all_val_conf_vectors = val_data_dict['val_conf_vectors']
        loaded.all_val_top_confs = val_data_dict['val_top_confs']
        loaded.all_val_features = val_data_dict['val_features']
        loaded.all_val_logits = val_data_dict['val_logits']
        loaded.lce_projection = val_data_dict['lce_projection']
        print(f'Model and test data loaded from {path}')
        return loaded

    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch, 'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(), 'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict() if self.use_amp else None,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        if self.use_amp and ckpt.get('scaler_state') is not None:
            self.scaler.load_state_dict(ckpt['scaler_state'])
        torch.set_rng_state(ckpt['torch_rng_state'].cpu())
        if ckpt['cuda_rng_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in ckpt['cuda_rng_state']])
        np.random.set_state(ckpt['numpy_rng_state'])
        return ckpt['epoch']


def train_model(train_loader, val_loader, test_loader, embedding_loader, label_list: list,
                                  model_params: dict, loss_params: dict, activation_params: dict = None,
                                  seed: int = 42, num_epochs: int = 200, num_bins=20, lr: float = 0.1, lr_schedule='multistep',
                                  momentum: float = 0.9, weight_decay: float = 5e-4, use_amp: bool = True,
                                  candidate_layers=('final', 'stage_minus_1', 'stage_minus_2'), embedding_layer='final',
                                  graph_recompute_every=None, early_stopping: bool = False, early_stopping_patience: int = 10,
                                  lambda_kind='constant', lambda_decay='cosine', lambda_off_frac=0.5,
                                  diag_every_n=10, checkpoint_epochs=None, resume_from=None, log_graph_maturity=False,
                                  root_path='./ncl_resnet', verbose=False, lambda_from_penalty_target=None,
                                  corruptions=None, corruption_severities=(1, 2, 3, 4, 5), corruption_base=None, data_root='./data',
                                  run_ood=False, ood_base=None, optimizer='sgd', lr_milestones_frac=(0.5, 0.75), lr_gamma=0.1,
                                  mixup_alpha=None, save_selection_ckpts=False):
    
    os.makedirs(root_path, exist_ok=True)
    plots_path = os.path.join(root_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    ckpt_dir = os.path.join(root_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    if checkpoint_epochs is None:
        checkpoint_epochs = sorted({int(num_epochs * f) for f in (0.5, 0.75, 0.99)})
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t_run0 = time.time()
    train_only_s = 0.0
    graph_build_s = None
    lam_sched = None
    lambda_history = []
    val_cal_history = []
    best_reg_star = best_task_star = best_lambda_star = None
    best_epoch_star = None
    diag_k = 10
    _n_emb = len(embedding_loader.dataset)
    diag_subset = np.random.RandomState(seed).choice(_n_emb, min(4096, _n_emb), replace=False)
    prev_diag_emb = None

    model = Trainer(model_params, loss_params, seed=seed, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay, num_epochs=num_epochs, embedding_layer=embedding_layer,
                                    use_amp=use_amp, lr_schedule=lr_schedule, optimizer=optimizer, lr_gamma=lr_gamma, lr_milestones_frac=lr_milestones_frac,
                                    mixup_alpha=mixup_alpha)
    inner_secondary = getattr(model.criterion, 'secondary_loss_fn', None)
    needs_activation = getattr(inner_secondary, 'requires_activation', False)

    if needs_activation and activation_params is None:
        raise ValueError("loss_params specifies a secondary loss that requires activation "
                          "(e.g. NCL), but activation_params was not provided.")

    controller = NCLActivationController(**activation_params) if needs_activation else None
    start_epoch = 0
    if resume_from is not None:
        start_epoch = model.load_checkpoint(resume_from)
        _sidecar = resume_from.rsplit('.pt', 1)[0] + '.train_only_s.json'
        if os.path.exists(_sidecar):
            with open(_sidecar) as _f:
                train_only_s = float(json.load(_f).get('train_only_s', 0.0))


    diagnostics_history = []
    activation_epoch = None
    best_val_ce_loss, improvement_counter = float('inf'), 0
    best_sel = {'nll': (float('inf'), None, None), 'ece': (float('inf'), None, None)}
    best_model_state = None
    divergence_patience = 5
    nonfinite_streak = 0
    first_nonfinite_epoch = None
    num_epochs_actual = num_epochs

    train_total_losses, train_ce_losses, train_reg_losses, train_accs = [], [], [], []
    val_losses, val_accs = [], []

    _iter = range(start_epoch, num_epochs)

    def _save_ckpt_with_time(path, ep):
        model.save_checkpoint(path, ep)
        with open(path.rsplit('.pt', 1)[0] + '.train_only_s.json', 'w') as _f:
            json.dump({'train_only_s': train_only_s, 'epoch': ep}, _f)

    pbar = tqdm(_iter, desc='Training', initial=start_epoch, total=num_epochs) if not verbose else _iter
    for epoch in pbar:
        if epoch in checkpoint_epochs:
            _save_ckpt_with_time(os.path.join(ckpt_dir, f'epoch_{epoch}.pt'), epoch)
        if verbose:
            print(f'\nEpoch {epoch + 1}/{num_epochs}')

        activated = model.secondary_active
        if needs_activation and not activated:
            if controller.strategy == 'auto':
                embeddings_dict, epoch_labels = extract_all_embeddings(model.model, embedding_loader, model.device, candidate_layers)
                diagnostics = {layer: compute_embedding_quality(embeddings_dict[layer], epoch_labels, seed=seed)
                                for layer in candidate_layers}
                diagnostics_history.append({'epoch': epoch, **diagnostics})
                should_activate = controller.step(epoch, diagnostics[embedding_layer])
            else:
                should_activate = controller.step(epoch, {})

            if should_activate:
                _t0 = time.time()
                model.activate_secondary_loss(embedding_loader, candidate_layers, embedding_layer)
                graph_build_s = time.time() - _t0
                activation_epoch = epoch
                lam_peak = model.criterion.lambda_reg
                if lambda_from_penalty_target is not None:
                    _mep = getattr(model._inner_secondary, 'mean_edge_penalty_at_activation', None)
                    if _mep and _mep > 0:
                        lam_peak = float(lambda_from_penalty_target) / _mep
                        model.criterion.lambda_reg = lam_peak
                        if verbose: print(f'  auto-λ from penalty: λ={lam_peak:.2f} (mean_edge_penalty={_mep:.4f})')
                lam_sched = make_lambda_schedule(lambda_kind, lam_peak, onset=epoch, E=num_epochs,
                                                 off_frac=lambda_off_frac, decay=lambda_decay)
                _save_ckpt_with_time(os.path.join(ckpt_dir, f'epoch_{epoch}_ncl_activated.pt'), epoch)
                if verbose:
                    print(f'NCL activated at epoch {epoch + 1}; lam_peak={lam_peak}, '
                          f'mean_edge_penalty={getattr(model._inner_secondary, "mean_edge_penalty_at_activation", None)}')

        elif needs_activation and activated and graph_recompute_every and \
                (epoch - activation_epoch) % graph_recompute_every == 0 and epoch != activation_epoch:
            model.activate_secondary_loss(embedding_loader, candidate_layers, embedding_layer)
            if verbose:
                print(f'NCL graph recomputed at epoch {epoch + 1}.')


        if needs_activation and lam_sched is not None and model.secondary_active:
            model.criterion.lambda_reg = lam_sched(epoch)
        lambda_history.append({'epoch': epoch, 'lambda': model.criterion.lambda_reg if needs_activation else 0.0})
        _t_tr = time.time()
        train_total_loss, train_ce_loss, train_reg_loss, train_acc = model.train_epoch(train_loader, verbose=verbose)
        train_only_s += time.time() - _t_tr
        train_total_losses.append(train_total_loss)
        train_ce_losses.append(train_ce_loss)
        train_reg_losses.append(train_reg_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = model.validate(val_loader, verbose=verbose)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        _vc, _vl = model.all_val_conf_vectors, model.all_val_labels
        _vnll = float(F.nll_loss(torch.log(torch.tensor(_vc).clamp_min(1e-12)),
                                 torch.tensor(_vl)).item())
        val_cal_history.append({
            'epoch': epoch, 'val_nll': _vnll,
            'val_ece': float(ece_mce(_vc, _vl, num_bins=num_bins)[0]),
            'val_brier': float(brier_score(_vc, _vl)),
            'mean_entropy': float((-(_vc * np.log(np.clip(_vc, 1e-12, 1.0))).sum(1)).mean()),
        })
        if save_selection_ckpts and math.isfinite(_vnll):
            for _m, _v in (('nll', _vnll), ('ece', val_cal_history[-1]['val_ece'])):
                if math.isfinite(_v) and _v < best_sel[_m][0]:
                    best_sel[_m] = (_v, epoch, {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()})

        if (needs_activation or log_graph_maturity) and (epoch % diag_every_n == 0 or epoch == num_epochs - 1):
            _rec, prev_diag_emb = _nescal_epoch_diagnostics(
                model, embedding_loader, embedding_layer, epoch, seed, prev_diag_emb, diag_subset, k=diag_k)
            diagnostics_history.append(_rec)

        if verbose:
            print(f'Train Loss: {train_total_loss:.4f} (CE: {train_ce_loss:.4f}, Reg: {train_reg_loss:.4f}), '
                  f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if not math.isfinite(val_loss):
            if first_nonfinite_epoch is None:
                first_nonfinite_epoch = epoch
            nonfinite_streak += 1
            print(f'  [epoch {epoch + 1}] NON-FINITE val loss ({val_loss}) — '
                  f'{nonfinite_streak}/{divergence_patience} consecutive')
            if nonfinite_streak >= divergence_patience:
                raise DivergedRun(first_nonfinite_epoch, val_loss, nonfinite_streak)
        else:
            nonfinite_streak = 0
            if val_loss < best_val_ce_loss:
                improvement_counter = 0
                best_model_state = copy.deepcopy(model.model.state_dict())
                best_val_ce_loss = val_loss
                best_reg_star = train_reg_loss
                best_task_star = train_ce_loss
                best_lambda_star = model.criterion.lambda_reg if needs_activation else 0.0
                best_epoch_star = epoch
            else:
                improvement_counter += 1

        if improvement_counter >= early_stopping_patience and early_stopping:
            if verbose:
                print(f'Early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs).')
            num_epochs_actual = epoch + 1
            break

    num_epochs_actual = len(train_accs)
    basic_metrics_per_epoch_unspec([train_accs, val_accs], ['Train Accuracy', 'Validation Accuracy'],
                                    num_epochs_actual, plots_path, savename='accuracy')
    basic_metrics_per_epoch_unspec([train_total_losses, val_losses], ['Train Total Loss', 'Val Total Loss'],
                                    num_epochs_actual, plots_path, savename='total_losses')
    basic_metrics_per_epoch_unspec([train_ce_losses, train_reg_losses], ['Train CE Loss', 'Train Reg Loss'],
                                    num_epochs_actual, plots_path, savename='train_losses')

    if best_model_state is None:
        raise DivergedRun(first_nonfinite_epoch if first_nonfinite_epoch is not None else 0,
                          best_val_ce_loss, num_epochs)
    if save_selection_ckpts:
        for _m, (_v, _e, _s) in best_sel.items():
            if _s is not None:
                torch.save({'model_state': _s, 'epoch': _e, 'val_score': _v, 'selected_by': f'val_{_m}'},
                           os.path.join(ckpt_dir, f'best_val_{_m}.pt'))
    model.model.load_state_dict(best_model_state)
    model.validate(val_loader, verbose=verbose)
    model.fit_lce_projection(n_components=min(50, model.model.feat_dim))


    local_freq = local_freq_error(model.all_val_conf_vectors, model.all_val_labels,
                                   model.all_val_features, lce_projection=model.lce_projection)
    bound_diag = None
    if needs_activation and model.secondary_active:

        _Z, _P, _ = _embed_probs_labels(model.model, embedding_loader, model.device, embedding_layer)
        bound_diag = compute_bound_diagnostics(
            _P, neighbors=model._inner_secondary.neighbors.cpu().numpy(), embeddings=_Z,
            head_weight=model.model.linear.weight.detach().cpu().numpy())
    train_time_diag = {
        'activation_epoch': activation_epoch,
        'heuristics_history': diagnostics_history, 'val_cal_history': val_cal_history,
        'train_acc_history': train_accs, 'val_acc_history': val_accs,
        'lambda_history': lambda_history,
        'theory': {'R_n_star': best_reg_star, 'L_task_star': best_task_star,
                   'lambda_star': best_lambda_star, 'best_epoch': best_epoch_star,
                   'mean_edge_js_at_activation': getattr(model._inner_secondary, 'mean_edge_js_at_activation', None) if (needs_activation and model.secondary_active) else None,
                   'bound_diagnostics': bound_diag, 'local_freq_error': local_freq},
        'cost': {'wall_clock_s': time.time() - t_run0, 'graph_build_s': graph_build_s,
                 'peak_mem_bytes': (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None),
                 'train_only_s': train_only_s, 'sec_per_train_epoch': train_only_s / max(num_epochs_actual, 1)},
    }
    with open(os.path.join(root_path, 'train_time_diagnostics.pkl'), 'wb') as f:
        pickle.dump(train_time_diag, f)

    plots_params = {'class_names': label_list, 'save_path': plots_path}

    val_results_dict = model.evaluate(calibration_bins=num_bins, verbose=verbose, split='val')
    val_results_dict['val_loss'] = best_val_ce_loss
    with open(os.path.join(root_path, 'val_results.pkl'), 'wb') as f:
        pickle.dump(val_results_dict, f)

    model.test(test_loader, verbose=verbose)
    results_dict = model.evaluate(calibration_bins=num_bins, verbose=verbose, plots_params=plots_params, split='test')
    results_dict['val_loss'] = best_val_ce_loss
    results_dict['ncl_activation_epoch'] = activation_epoch

    if corruptions:
        corr = evaluate_corruptions(model, corruption_base, corruptions, corruption_severities,
                                    num_bins, data_root=data_root, batch_size=test_loader.batch_size)
        with open(os.path.join(root_path, 'corruption_results.pkl'), 'wb') as f:
            pickle.dump(corr, f)

    if run_ood:
        ood_res = evaluate_ood_separation(model, ood_base, data_root=data_root,
                                          batch_size=test_loader.batch_size)
        with open(os.path.join(root_path, 'ood_results.pkl'), 'wb') as f:
            pickle.dump(ood_res, f)

    model.save_model(root_path)
    with open(os.path.join(root_path, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    return model, val_results_dict, results_dict, diagnostics_history
